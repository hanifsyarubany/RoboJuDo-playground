"""ProtoMotions tracker policy for RoboJuDo.

Runs a unified ONNX model exported by
``deployment/export_bm_tracker_onnx.py`` with cached 50 fps motion from
``deployment/motion_utils.MotionPlayer``.

Key inputs:

- ``historical.processed_actions`` — action history feedback (previous PD
  targets are fed back as an ONNX input)
- ``mimic.future_anchor_rot`` — anchor-body-only rotation references

Heading alignment
-----------------
Yaw-only offset computed on first step to align motion heading with robot heading.

Sensor requirements (real G1)
-----------------------------
- ``env_data.dof_pos`` / ``env_data.dof_vel`` -- joint encoders
- ``env_data.base_quat`` (xyzw) -- pelvis IMU
- ``env_data.base_ang_vel`` -- pelvis IMU gyro (body-local frame)
- ``env_data.torso_quat`` (xyzw) -- FK-computed (requires ``update_with_fk=True``)
"""

import logging
import re

import numpy as np
import onnxruntime as ort
import yaml

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.motion_utils import (
    MotionPlayer,
    _extract_yaw_quat_np,
    apply_heading_offset_np,
    compute_yaw_offset_np,
)

logger = logging.getLogger(__name__)


@policy_registry.register
class ProtoMotionsTrackerPolicy(Policy):
    """Policy that drives a ProtoMotions tracker via unified ONNX model.

    The ONNX model bakes in: obs computation -> actor MLP -> action processing.
    Inputs are raw context tensors; outputs are absolute PD position targets.
    """

    cfg_policy: PolicyCfg

    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        # Load YAML metadata BEFORE calling super().__init__ so we can
        # build the DOF config from it.
        onnx_path = cfg_policy.policy_file
        yaml_path = onnx_path.replace(".onnx", ".yaml")

        with open(yaml_path) as f:
            self._meta = yaml.safe_load(f)

        robot_meta = self._meta["robot"]
        control_meta = self._meta["control"]
        motion_meta = self._meta["motion"]
        runtime = self._meta["_runtime"]

        joint_names = robot_meta["joint_names"]
        num_dofs = robot_meta["num_dofs"]
        stiffness = control_meta["stiffness"]
        damping = control_meta["damping"]
        effort_limits = control_meta.get("effort_limits")

        # Build DOF config from YAML metadata.
        dof_cfg = DoFConfig(
            joint_names=joint_names,
            default_pos=[0.0] * num_dofs,
            stiffness=stiffness,
            damping=damping,
            torque_limits=effort_limits,
        )
        cfg_policy_updated = cfg_policy.model_copy()
        cfg_policy_updated.obs_dof = dof_cfg
        cfg_policy_updated.action_dof = dof_cfg

        super().__init__(cfg_policy=cfg_policy_updated, device="cpu")

        # ONNX session
        logger.info(f"[TrackerPolicy] Loading ONNX: {onnx_path}")
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._onnx_in_names = [inp.name for inp in self._session.get_inputs()]
        self._onnx_out_names = [out.name for out in self._session.get_outputs()]
        self._onnx_name_to_key = runtime["onnx_name_to_in_key"]

        # Motion player (cached mode -- no protomotions import)
        motion_path = getattr(cfg_policy, "motion_path", None)
        if motion_path is None:
            raise ValueError("ProtoMotionsTrackerPolicyCfg must set motion_path")
        motion_index = getattr(cfg_policy, "motion_index", 0)
        timing = self._meta["timing"]
        self._control_dt = float(timing["control_dt"])
        self._player = MotionPlayer(
            motion_path, motion_index=motion_index, control_dt=timing["control_dt"]
        )

        # ONNX input config
        self._anchor_idx = robot_meta["anchor_body_index"]
        self._root_idx = robot_meta["root_body_index"]
        self._future_step_indices = motion_meta["future_step_indices"]

        # Determine how to read the anchor body rotation from env_data.
        # Use body name to look up in fk_info; pelvis uses base_quat directly.
        self._anchor_body_name = robot_meta.get("anchor_body_name")
        logger.info(
            f"[TrackerPolicy] anchor body: "
            f"{self._anchor_body_name or 'pelvis'} (idx={self._anchor_idx})"
        )

        # Action post-processing config
        self._pd_target_max_accel = control_meta.get("pd_target_max_accel")
        self._action_ema_alpha = control_meta.get("action_ema_alpha", 1.0)

        logger.info(
            f"[TrackerPolicy] {num_dofs} DOFs, "
            f"{self._player.total_frames} motion frames, "
            f"anchor_idx={self._anchor_idx}, root_idx={self._root_idx}"
        )

        # Resolve default standing pose from protomotions robot config.
        self._default_dof_pos = self._resolve_default_dof_pos(joint_names)

        self._heading_offset = None
        self.reset()

    # G1 default standing pose (from protomotions.robot_configs.g1)
    _G1_DEFAULT_JOINT_POS = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }

    def _resolve_default_dof_pos(self, joint_names: list[str]) -> np.ndarray:
        """Resolve default DOF positions via regex-pattern matching."""
        DEFAULT_JOINT_POS = self._G1_DEFAULT_JOINT_POS

        default_pos = np.zeros(len(joint_names), dtype=np.float32)
        for pattern, value in DEFAULT_JOINT_POS.items():
            for i, name in enumerate(joint_names):
                if re.fullmatch(pattern, name):
                    default_pos[i] = value
        logger.info(f"[TrackerPolicy] resolved default DOF pos: {default_pos}")
        return default_pos

    def set_default_pose_mode(self, enabled: bool):
        """Switch between tracking real motion and holding default pose.

        When enabled, the policy sees synthetic references for the default
        standing pose instead of the real motion.  Used during prepare/rampdown.
        """
        self._default_pose_mode = enabled
        if enabled:
            self._motion_done = False
        logger.info(f"[TrackerPolicy] default_pose_mode={'ON' if enabled else 'OFF'}")

    # Keyboard key → (vx, vy, yaw_rate) velocity contribution.
    # W/S: forward/backward, A/D: strafe left/right, Q/E: turn left/right.
    # 'k' release toggles the kick trigger (handled separately below).
    _KEY_CMD_MAP: dict[str, np.ndarray] = {
        "w": np.array([ 0.5,  0.0,  0.0], dtype=np.float32),
        "s": np.array([-0.3,  0.0,  0.0], dtype=np.float32),
        "a": np.array([ 0.0,  0.3,  0.0], dtype=np.float32),
        "d": np.array([ 0.0, -0.3,  0.0], dtype=np.float32),
        "q": np.array([ 0.0,  0.0,  0.5], dtype=np.float32),
        "e": np.array([ 0.0,  0.0, -0.5], dtype=np.float32),
    }

    # Gait-phase clock period (Stage-3 v11+ ONNX only). MUST match
    # GAIT_CYCLE_TIME in stage3_training/stage3_env.py -- a mismatch desyncs
    # the deployed gait's cadence from the one the policy was trained against.
    # Harmless for older (pre-v11) ONNX exports: cmd.phase is only forwarded
    # to the model if the loaded ONNX actually declares a cmd_phase input
    # (see the onnx_in_names-driven filter in get_observation).
    _GAIT_CYCLE_TIME = 0.9

    def reset(self):
        self._frame = 0
        self._prev_pd = None
        self._prev_prev_pd = None
        self._ema_prev = None
        self._stashed_pd_targets = np.zeros(self.num_actions, dtype=np.float32)
        self._prev_actions = np.zeros(self.num_actions, dtype=np.float32)
        self._motion_done = False
        self._paused = False
        self._default_pose_mode = False
        # Ball state for Stage-2 ONNX (ball.pos / ball.vel inputs).
        # Initialised on first step from initial robot position.
        # None until first get_observation call; zeros if no ball in scene.
        self._ball_pos_world: np.ndarray | None = None
        self._ball_vel_world: np.ndarray | None = None
        # Stage-3 cmd inputs: cmd_vel (vx, vy, yaw_rate) and kick trigger.
        # Updated each step via keyboard events or external ctrl_data.
        self._cmd_vel = np.zeros(3, dtype=np.float32)
        self._cmd_kick = np.zeros(1, dtype=np.float32)
        # Stage-3 v11+ gait-phase clock. Free-running, advanced once per
        # control step in get_observation regardless of walk/kick mode
        # (mirrors G1Stage3Env.post_physics_step's unconditional advance --
        # only the reward/obs terms that consume it are mode-gated, not the
        # clock itself). Reset to 0 here so a fresh episode starts in phase.
        self._gait_phase = 0.0
        # Set of currently held velocity-command keys (for hold-to-walk).
        self._held_keys: set[str] = set()

    def reset_alignment(self):
        self._heading_offset = None

    def _process_keyboard_for_cmd(self, ctrl_data) -> None:
        """Update self._cmd_vel and self._cmd_kick from keyboard events.

        Stage-3 ONNX has cmd_vel and cmd_kick as inputs.  This method reads raw
        KeyboardCtrl events (not yet consumed by process_triggers) so that:
          W/S → forward/backward   A/D → strafe left/right   Q/E → turn
          k (release) → toggle kick trigger (0→1→0)

        Velocity commands accumulate from all currently held keys (hold W to
        walk forward; release to stop).  cmd_vel is in the robot body frame,
        m/s / rad/s — matching the training distribution.

        For joystick integration: if ctrl_data already has a ``cmd_vel`` array
        under any controller key, it takes precedence over keyboard velocity.
        """
        kb_events = ctrl_data.get("KeyboardCtrl", {}).get("keyboard_event", [])
        for event in kb_events:
            if event.get("type") != "keyboard":
                continue
            key = str(event.get("name", ""))
            pressed = bool(event.get("pressed", False))
            if key in self._KEY_CMD_MAP:
                if pressed:
                    self._held_keys.add(key)
                else:
                    self._held_keys.discard(key)
            elif key == "k" and not pressed:
                new_kick = 1.0 - float(self._cmd_kick[0])
                self._cmd_kick[0] = new_kick
                if new_kick > 0.5:
                    # Rewind the kick clip so the FULL swing is tracked from the
                    # start.  Without this, a kick triggered after walking would
                    # read the reference at a playhead that has already run off
                    # the end of the clip (see post_step_callback) and no swing
                    # would be performed.
                    self._frame = 0
                    self._motion_done = False
                    # Re-align the kick clip to the robot's CURRENT heading.
                    # The heading offset is computed once at spawn and frozen.
                    # In walk mode the mimic reference is gated OFF inside the
                    # ONNX, so a stale heading offset is invisible — but the
                    # moment the kick turns on, the kick clip's anchor-rotation
                    # reference (mimic.future_anchor_rot) is fed through and, if
                    # the robot has since turned while walking, it points the
                    # torso at the OLD world heading.  The policy then twists to
                    # that stale heading instead of kicking forward and the kick
                    # fails.  Forcing a recompute makes the kick face wherever
                    # the robot currently faces — what the MuJoCo unit test gets
                    # for free by teleporting to frame 0 each episode.
                    self._heading_offset = None
                state = "ON" if new_kick > 0.5 else "OFF"
                logger.info(f"[TrackerPolicy] kick_trigger={state}")
        # Recompute cmd_vel from the union of currently held keys.
        vel = np.zeros(3, dtype=np.float32)
        for k in self._held_keys:
            delta = self._KEY_CMD_MAP.get(k)
            if delta is not None:
                vel += delta
        self._cmd_vel[:] = vel

    def post_step_callback(self, commands=None):
        # Only advance the kick-clip playhead while a kick is ACTIVE.  In walk
        # mode the mimic reference is gated OFF inside the ONNX (cmd_kick<0.5 →
        # reference zeroed), so a free-running playhead would silently run off
        # the end of the kick clip; by the time the user triggers a kick the
        # reference would be stuck on the last (post-kick) frame and no swing
        # would be performed.  Freezing the playhead during walk, rewinding on
        # the kick trigger (see _process_keyboard_for_cmd), and auto-returning
        # to walk when the swing completes makes kick a clean one-shot that can
        # be triggered any time — including after walking.
        in_kick = bool(self._cmd_kick[0] >= 0.5)
        if in_kick and not self._paused and not self._default_pose_mode:
            self._frame += 1
            if self._frame >= self._player.total_frames:
                # Kick swing finished → auto-return to walk and rewind so the
                # next kick starts fresh.
                self._frame = 0
                self._motion_done = False
                self._cmd_kick[0] = 0.0
                logger.info("[TrackerPolicy] kick complete → walk mode")
        for cmd in commands or []:
            if cmd in ("[MOTION_RESET]", "[MOTION_FADE_IN]"):
                self.reset()

    def get_observation(self, env_data, ctrl_data):
        # -- Stage-3: update cmd_vel / cmd_kick from keyboard events --
        self._process_keyboard_for_cmd(ctrl_data)

        # -- Heading alignment (first step after reset) --
        if self._heading_offset is None:
            motion_anchor_rot = self._player.get_state_at_frame(0)["body_rot"][self._anchor_idx]
            robot_anchor_rot = self._get_anchor_quat(env_data)
            self._heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)

        # -- State from env_data (already xyzw) --
        anchor_rot = self._get_anchor_quat(env_data)
        anchor_pos = self._get_anchor_pos(env_data)
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        # env_data.base_ang_vel comes from MuJoCo qvel[3:6] which is ALREADY
        # in the pelvis local frame (not world frame).  On the real G1, the
        # IMU gyroscope also reads in body-local frame.  So we use it directly
        # as root_local_ang_vel -- NO quat_rotate_inverse needed.
        root_local_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)

        # Ball state: compute a fixed nominal position on the first step
        # (0.6 m ahead, 0.15 m to the robot's right, in the root yaw frame).
        # On real hardware supply perception values here instead.
        if self._ball_pos_world is None:
            root_pos = np.asarray(env_data.base_pos, dtype=np.float32)
            x, y, z, w = anchor_rot   # xyzw; yaw = 2*atan2(z, w)
            yaw = 2.0 * float(np.arctan2(z, w))
            cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
            fwd, lat, bz = 0.6, -0.15, 0.115
            self._ball_pos_world = np.array([
                root_pos[0] + cy * fwd - sy * lat,
                root_pos[1] + sy * fwd + cy * lat,
                bz,
            ], dtype=np.float32)
            self._ball_vel_world = np.zeros(3, dtype=np.float32)

        if self._default_pose_mode:
            # -- Synthetic references: hold default standing pose --
            # Target DOFs = default standing pose, velocities = zero,
            # anchor rotation = yaw-only from robot's current anchor (hold
            # heading but neutral pitch/roll for stable upright standing).
            num_steps = len(self._future_step_indices)
            anchor_yaw_only = _extract_yaw_quat_np(anchor_rot)
            future_anchor_rot = np.tile(anchor_yaw_only, (num_steps, 1))
            future_dof_pos = np.tile(self._default_dof_pos, (num_steps, 1))
            future_dof_vel = np.zeros_like(future_dof_pos)
        else:
            # -- Future motion references with heading alignment --
            # Clamp each future step so it never exceeds the last valid frame.
            # This repeats the last frame's references at end-of-motion instead
            # of going out of bounds.
            last_frame = self._player.total_frames - 1
            clamped_steps = [min(self._frame + step, last_frame) - self._frame for step in self._future_step_indices]
            future_refs = self._player.get_future_references(self._frame, clamped_steps)
            future_body_rot = apply_heading_offset_np(self._heading_offset, future_refs["body_rot"])
            # Anchor-body-only rotation: [num_steps, 4]
            future_anchor_rot = future_body_rot[:, self._anchor_idx, :]
            future_dof_pos = future_refs["dof_pos"]
            future_dof_vel = future_refs["dof_vel"]

        # -- Build ONNX inputs --
        key_to_array = {
            "current.dof_pos": dof_pos[None],
            "current.dof_vel": dof_vel[None],
            "current.anchor_rot": anchor_rot[None],
            "current.anchor_pos": anchor_pos[None],
            "current.root_local_ang_vel": root_local_ang_vel[None],
            "mimic.future_anchor_rot": future_anchor_rot[None],
            "mimic.future_dof_pos": future_dof_pos[None],
            "mimic.future_dof_vel": future_dof_vel[None],
            "historical.processed_actions": self._prev_actions[None, None],
            # Stage-2 ball inputs (zeros / fixed pos for Stage-1 ONNX which won't request them).
            "ball.pos": self._ball_pos_world[None],
            "ball.vel": self._ball_vel_world[None],
            # Stage-3 teleoperation inputs: velocity command and kick trigger.
            # cmd_vel is raw body-frame (vx m/s, vy m/s, yaw rad/s); the ONNX
            # model normalises it internally.  cmd_kick is binary (0=walk, 1=kick).
            # Updated each step by _process_keyboard_for_cmd(); zero by default.
            "cmd.vel": self._cmd_vel[None],
            "cmd.kick": self._cmd_kick[None],
            # Stage-3 v11+ gait-phase clock (sin/cos computed in-graph). Only
            # picked up below if the loaded ONNX actually has a cmd_phase
            # input -- a no-op dict entry for older Stage-1/2 exports.
            "cmd.phase": np.array([self._gait_phase], dtype=np.float32)[None],
        }
        onnx_inputs = {}
        for onnx_name in self._onnx_in_names:
            sem_key = self._onnx_name_to_key.get(onnx_name)
            if sem_key and sem_key in key_to_array:
                onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

        # Advance the gait-phase clock for the NEXT step. Must happen exactly
        # once per control step, after this step's value was captured into
        # key_to_array above -- matches test_tracker_mujoco.py's capture-then-
        # advance ordering and G1Stage3Env's per-step phase increment.
        self._gait_phase = (self._gait_phase + self._control_dt / self._GAIT_CYCLE_TIME) % 1.0

        # -- ONNX inference --
        ort_out = self._session.run(self._onnx_out_names, onnx_inputs)
        pd_targets = ort_out[1].squeeze().copy()

        # -- PD target acceleration clamp --
        if (
            self._pd_target_max_accel is not None
            and self._prev_pd is not None
            and self._prev_prev_pd is not None
        ):
            delta = pd_targets - self._prev_pd
            prev_delta = self._prev_pd - self._prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(
                accel, -self._pd_target_max_accel, self._pd_target_max_accel
            )
            pd_targets = self._prev_pd + prev_delta + clamped_accel
        self._prev_prev_pd = self._prev_pd
        self._prev_pd = pd_targets.copy()

        # -- EMA action filter --
        alpha = self._action_ema_alpha
        if alpha < 1.0:
            if self._ema_prev is None:
                self._ema_prev = pd_targets.copy()
            pd_targets = alpha * pd_targets + (1.0 - alpha) * self._ema_prev
            self._ema_prev = pd_targets.copy()

        self._stashed_pd_targets = pd_targets
        self._prev_actions = pd_targets.copy()
        extras = {
            "CALLBACK": (
                ["[MOTION_DONE]"]
                if self._motion_done and not self._default_pose_mode
                else []
            ),
        }
        dummy_obs = np.zeros(1, dtype=np.float32)
        return dummy_obs, extras

    def _get_anchor_quat(self, env_data) -> np.ndarray:
        """Read the anchor body's quaternion from env_data.

        Uses the body name from YAML metadata to look up in fk_info.
        Falls back to base_quat for pelvis (root body).
        """
        name = self._anchor_body_name
        if name is not None and name not in (None, "pelvis"):
            # Named body -- look up in FK info
            fk = env_data.fk_info
            if fk is not None and name in fk:
                return np.asarray(fk[name]["quat"], dtype=np.float32)
            # Fallback: if the env exposes it as torso_quat and name matches
            if name == "torso_link" and env_data.torso_quat is not None:
                return np.asarray(env_data.torso_quat, dtype=np.float32)
        # Pelvis / root body -- always available as base_quat
        return np.asarray(env_data.base_quat, dtype=np.float32)

    def _get_anchor_pos(self, env_data) -> np.ndarray:
        """Read the anchor body's world position from env_data.

        Mirrors _get_anchor_quat: prefers fk_info, falls back to torso_pos,
        then base_pos (pelvis world XYZ).
        """
        name = self._anchor_body_name
        if name is not None and name not in (None, "pelvis"):
            fk = env_data.fk_info
            if fk is not None and name in fk:
                return np.asarray(fk[name]["pos"], dtype=np.float32)
            if name == "torso_link" and env_data.torso_pos is not None:
                return np.asarray(env_data.torso_pos, dtype=np.float32)
        return np.asarray(env_data.base_pos, dtype=np.float32)

    def get_action(self, obs):
        return self._stashed_pd_targets

    def get_init_dof_pos(self):
        return self._player.get_state_at_frame(0)["dof_pos"].copy()
