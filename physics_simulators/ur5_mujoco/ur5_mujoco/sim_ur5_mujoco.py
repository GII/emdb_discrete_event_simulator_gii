import os.path
import math
import time
import threading
import yaml
import yamlloader
import numpy as np

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig
from core.container import Container
from core.utils import class_from_classname, resolve_seed


# ----------------------------------------------------------------------------- MuJoCo backend
class UR5Scenario:
    """
    MuJoCo UR5e scenario: single arm, one ball, one box.
    Autograsp: ball attaches magnetically to the flange when close,
    releases (and drops) when the EE hovers over the box.
    Distances in meters, angles in degrees (like the 2D sim, wrap [-180, 180]).
    Thread-safe: physics mutation and state snapshots are guarded by a lock so
    the perception publisher can run concurrently with arm motions.
    """

    SITE_NAME = "attachment_site"
    magnet = True  # the ball follows the flange while grasped (see _physics_step)

    # IK / servo parameters
    DAMPING = 1e-2
    POS_GAIN = 8.0
    ROT_WEIGHT = 0.4
    MAX_DQ = 0.6

    def __init__(self, model_path, visualize=True, logger=None,
                 hover_z=0.16, grasp_range=0.07, place_range=0.07,
                 workspace_r=(0.30, 0.65), z_bounds=(0.032, 0.50),
                 realtime=False, speed=2.0, spawn_box=True):
        self.logger = logger
        self.hover_z = hover_z
        self.grasp_range = grasp_range
        self.place_range = place_range
        self.workspace_r = workspace_r
        self.z_bounds = z_bounds  # vertical clamp for z-enabled polar actions
        self.realtime = realtime  # pace physics to wall clock so motion is watchable
        self.speed = max(speed, 0.1)  # playback speed: 1.0 = real time, 2.0 = 2x, ...

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)  # scratch copy for kinematic IK
        self._last_view_sync = 0.0  # wall-clock time of the last viewer.sync()
        self._pace_debt = 0.0  # accumulated wall-clock seconds owed to realtime pacing
        self._instability_count = 0  # times the physics guard rolled back a bad step
        self.lock = threading.Lock()  # guards self.data mutation vs state reads

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.SITE_NAME)
        self.box_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "box_site")
        ball_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qposadr = self.model.jnt_qposadr[ball_joint]
        self.ball_dofadr = self.model.jnt_dofadr[ball_joint]
        self.nq_arm = 6

        # Ball collision bitmasks, saved so we can disable the ball's contacts
        # while it is magnet-grasped (the follow-the-flange teleport would
        # otherwise ram it into the arm/box geoms and spike QACC to NaN).
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._ball_contype = int(self.model.geom_contype[self.ball_geom_id])
        self._ball_conaffinity = int(self.model.geom_conaffinity[self.ball_geom_id])

        # Optional box: when spawn_box is False, park the box far off-scene so it
        # is invisible, unreachable and inert. box_pos() then returns that far
        # point, so ball_in_box(), the reset exclusion and the place logic all
        # naturally no-op -- no separate scene file or rebuild needed.
        self.has_box = spawn_box
        box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_body_id = box_body_id
        if not spawn_box and box_body_id >= 0:
            self.model.body_pos[box_body_id] = np.array([100.0, 100.0, -5.0])
            mujoco.mj_forward(self.model, self.data)  # refresh box_site xpos

        self.ball_radius = float(self.model.geom_size[self.ball_geom_id][0])

        self.grasped = False
        self.heading = 0.0  # degrees, like the 2D robots
        self._setup_extra()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if visualize else None
        self.reset()

    def _setup_extra(self):
        """Hook for subclasses: model lookups needed before the first reset."""
        pass

    # ------------------------------------------------------------- state access
    # NOTE: these raw accessors do NOT take the lock (they are used inside the
    # locked physics section). External readers should use get_state().
    def ee_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def ball_pos(self):
        return self.data.qpos[self.ball_qposadr:self.ball_qposadr + 3].copy()

    def box_pos(self):
        return self.data.site_xpos[self.box_site_id].copy()

    def get_state(self):
        """Thread-safe snapshot for the perception publisher."""
        with self.lock:
            return {
                "ee": self.ee_pos(),
                "ball": self.ball_pos(),
                "box": self.box_pos(),
                "grasped": self.grasped,
                "heading": self.heading,
                "ball_in_box": self.ball_in_box(),
            }

    def get_distance(self, a, b):
        """Planar (xy) distance in meters."""
        return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))

    def get_relative_angle(self, a, b):
        """Planar angle (deg) of the vector a->b."""
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    @staticmethod
    def wrap_angle(a):
        return (a + 180.0) % 360.0 - 180.0

    def ball_in_box(self):
        b, x = self.ball_pos(), self.box_pos()
        return (abs(b[0] - x[0]) < 0.07 and abs(b[1] - x[1]) < 0.07
                and b[2] < 0.09 and not self.grasped)

    # ------------------------------------------------------------- reset
    def reset(self, rng=None):
        with self.lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # "home" keyframe
            self.data.ctrl[:self.nq_arm] = self.data.qpos[:self.nq_arm]
            self.grasped = False
            self._set_ball_collisions(True)  # ensure contacts on if reset while grasped

            self._place_ball_randomly(rng)

            mujoco.mj_forward(self.model, self.data)
            ee = self.ee_pos()
            self.heading = math.degrees(math.atan2(ee[1], ee[0]))
        self._settle(0.3)
        # Drive to a canonical tool-down start pose so every trial begins sane
        self.heading = 0.0
        self.move_ee_to(0.45, 0.0, self.hover_z, 0.0, duration=3.0)

    def _place_ball_randomly(self, rng=None):
        """Random reachable ball position on the floor, in the front half-plane and away from the box.
        Call with the lock held."""
        if rng is None:
            rng = np.random.default_rng()
        while True:
            r = rng.uniform(self.workspace_r[0] + 0.05, self.workspace_r[1] - 0.05)
            th = rng.uniform(-math.pi / 2, math.pi / 2)  # front half-plane
            bx, by = r * math.cos(th), r * math.sin(th)
            # The box is static: read its place from the model. (box_pos() reads site_xpos, which right after
            # mj_resetDataKeyframe is all zeros until mj_forward, so the ball could spawn inside the box.)
            if self.get_distance((bx, by), self.model.body_pos[self.box_body_id]) > 0.20:
                break
        self.data.qpos[self.ball_qposadr:self.ball_qposadr + 3] = [bx, by, self.ball_radius]
        self.data.qpos[self.ball_qposadr + 3:self.ball_qposadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_dofadr:self.ball_dofadr + 6] = 0

    # ------------------------------------------------------------- control
    def _yaw_down_quat(self, yaw_rad):
        q_yaw = np.array([math.cos(yaw_rad / 2), 0, 0, math.sin(yaw_rad / 2)])
        q_flip = np.array([0.0, 1.0, 0.0, 0.0])  # tool z pointing down
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q_yaw, q_flip)
        return out

    def _solve_ik_from(self, p_des, seed_qpos, iters=300, tol=1e-3):
        """One DLS IK solve from a given seed. Position (3 DoF) + tool axis
        pointing down (2 DoF); yaw stays free so the problem stays redundant."""
        self.ik_data.qpos[:] = self.data.qpos          # carry ball/other DOFs
        self.ik_data.qpos[:self.nq_arm] = seed_qpos
        jnt_range = self.model.jnt_range[:self.nq_arm]
        a_des = np.array([0.0, 0.0, -1.0])  # desired tool axis: straight down
        ROT_W = 0.3
        p_err = np.zeros(3)
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)

            p_err = p_des - self.ik_data.site_xpos[self.site_id]
            a_cur = self.ik_data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
            rot_err = np.cross(a_cur, a_des)
            if np.dot(a_cur, a_des) < -0.9:
                # Tool nearly upside down: cross product vanishes, kick it
                # around a perpendicular axis to escape the singularity.
                perp = np.cross(a_cur, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(a_cur, np.array([0.0, 1.0, 0.0]))
                rot_err = perp / np.linalg.norm(perp) * 2.0

            if np.linalg.norm(p_err) < tol and np.linalg.norm(rot_err) < 0.1:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.site_id)
            J = np.vstack([jacp, ROT_W * jacr])[:, :self.nq_arm]
            err6 = np.concatenate([p_err, ROT_W * rot_err])

            dq = J.T @ np.linalg.solve(J @ J.T + self.DAMPING * np.eye(6), err6)
            dq = np.clip(dq, -0.2, 0.2)
            self.ik_data.qpos[:self.nq_arm] = np.clip(
                self.ik_data.qpos[:self.nq_arm] + dq,
                jnt_range[:, 0], jnt_range[:, 1])
        return self.ik_data.qpos[:self.nq_arm].copy(), float(np.linalg.norm(p_err))

    def solve_ik(self, p_des, iters=300, tol=1e-3):
        """Kinematic DLS IK, seeded from the CURRENT arm pose so the returned
        configuration is the one nearest to where the arm already is. This keeps
        motion continuous: the old home-keyframe seed could snap the redundant
        arm into a different elbow/wrist basin near the ball, making the servos
        slew violently to a contorted pose. Falls back to the home keyframe only
        when the current-seeded solve does not converge.
        Returns (joint config, position residual in meters)."""
        seed = self.data.qpos[:self.nq_arm].copy()
        # Testing hook (regression A/B): UR5_IK_HOME_SEED=1 forces the old
        # home-keyframe seed to reproduce the IK-jump bug on demand.
        if os.environ.get("UR5_IK_HOME_SEED") == "1" and self.model.nkey > 0:
            seed = self.model.key_qpos[0][:self.nq_arm].copy()
        q, res = self._solve_ik_from(p_des, seed, iters=iters, tol=tol)
        # Only fall back to the home seed if the current-seeded solve genuinely
        # fails to get close (>5 cm). A slightly worse reach is preferable to the
        # jump the home seed can cause; continuity is the priority here.
        if res > 0.05 and self.model.nkey > 0:
            q_home, res_home = self._solve_ik_from(
                p_des, self.model.key_qpos[0][:self.nq_arm], iters=iters, tol=tol)
            if res_home < res:
                q, res = q_home, res_home
        return q, res

    def _state_is_sane(self):
        """True if the sim state is finite and within sane bounds. Catches both
        NaN/Inf and finite-but-huge blowups before they integrate further."""
        qpos, qvel = self.data.qpos, self.data.qvel
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            return False
        # No arm joint or ball DOF should ever exceed this speed in normal use.
        if np.abs(qvel).max() > 1.0e3:
            return False
        return True

    def _physics_step(self):
        step_start = time.perf_counter()
        with self.lock:
            # Snapshot before stepping so we can roll back if the step diverges.
            # The position-controlled arm can crush the light ball against the
            # floor during the grasp approach, spiking contact torque on the
            # ball's tiny rotational inertia -> NaN/huge QACC. Rolling back keeps
            # that from ever propagating into the arm pose or the episode.
            qpos_backup = self.data.qpos.copy()
            mujoco.mj_step(self.model, self.data)
            if not self._state_is_sane():
                self.data.qpos[:] = qpos_backup      # arm & ball back to last good pose
                self.data.qvel[:] = 0.0
                self.data.qacc[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self._instability_count += 1
                # Throttle: a sustained crush would otherwise warn every step.
                if self.logger and self._instability_count % 100 == 1:
                    self.logger.warn(
                        f"Physics instability caught and rolled back "
                        f"(total={self._instability_count})")
            if self.grasped and self.magnet:
                # Magnet grasp: ball follows the flange (like set_pos in the 2D sim)
                ee = self.ee_pos()
                hold = ee - np.array([0, 0, 0.05])
                hold[2] = max(hold[2], 0.031)  # never push the held ball into the floor
                self.data.qpos[self.ball_qposadr:self.ball_qposadr + 3] = hold
                self.data.qvel[self.ball_dofadr:self.ball_dofadr + 6] = 0
        # Render at most ~30 FPS of wall clock: viewer.sync() is expensive
        # (software GL in Docker), so throttling by time instead of step count
        # keeps the render cost independent of timestep and speed.
        now = time.perf_counter()
        if (self.viewer is not None and self.viewer.is_running()
                and now - self._last_view_sync >= 1.0 / 30.0):
            self.viewer.sync()
            self._last_view_sync = time.perf_counter()
        if self.realtime and self.viewer is not None and self.viewer.is_running():
            # Pace against an accumulated budget rather than sleeping every
            # step: the per-step budget (timestep / speed) is sub-millisecond,
            # below time.sleep() granularity, so per-step sleeps overshoot and
            # drag the sim far below the requested speed.
            budget = self.model.opt.timestep / self.speed
            self._pace_debt += budget - (time.perf_counter() - step_start)
            if self._pace_debt >= 0.002:
                sleep_start = time.perf_counter()
                time.sleep(self._pace_debt)
                self._pace_debt -= time.perf_counter() - sleep_start
            elif self._pace_debt < -0.1:
                # We fell behind (e.g. a slow render); don't bank an unbounded
                # deficit that would suppress pacing long after we catch up.
                self._pace_debt = -0.1

    def _settle(self, seconds):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._physics_step()

    def move_ee_to(self, x, y, z, yaw_deg=0.0, duration=1.5, tol=0.01):
        """Move the EE to (x, y, z): solve IK kinematically, command the joint
        target to the position actuators, step physics until tracked.
        Yaw around the vertical is left free by the IK."""
        # Clamp target to the reachable annulus
        r = math.hypot(x, y)
        r_cl = min(max(r, self.workspace_r[0]), self.workspace_r[1])
        if r > 1e-6 and r_cl != r:
            x, y = x * r_cl / r, y * r_cl / r
        p_des = np.array([x, y, z])

        q_target, ik_err = self.solve_ik(p_des)
        if ik_err > 0.02 and self.logger:
            self.logger.warn(
                f"Target unreachable: IK residual {ik_err*100:.1f} cm at {np.round(p_des, 3)}")
        lo = self.model.actuator_ctrlrange[:self.nq_arm, 0]
        hi = self.model.actuator_ctrlrange[:self.nq_arm, 1]
        with self.lock:
            self.data.ctrl[:self.nq_arm] = np.clip(q_target, lo, hi)

        for _ in range(int(duration / self.model.opt.timestep)):
            self._physics_step()
            if np.linalg.norm(p_des - self.ee_pos()) < tol:
                break
        track_err = float(np.linalg.norm(p_des - self.ee_pos()))
        if track_err > 0.02 and ik_err <= 0.02 and self.logger:
            self.logger.warn(
                f"Tracking residual {track_err*100:.1f} cm (IK was fine) at {np.round(p_des, 3)}")
        return self.ee_pos()

    def _rescue_ball(self):
        """If the ball rolled out of the reachable annulus after a push, clamp
        it back to the nearest reachable point (same bearing, corrected radius)
        so the trial does not go dead until the next world reset."""
        if self.grasped:
            return
        ball = self.ball_pos()
        r = math.hypot(ball[0], ball[1])
        lo, hi = self.workspace_r[0] + 0.03, self.workspace_r[1] - 0.03
        r_cl = min(max(r, lo), hi)
        if abs(r_cl - r) < 1e-6:
            return
        scale = r_cl / max(r, 1e-9)
        with self.lock:
            self.data.qpos[self.ball_qposadr:self.ball_qposadr + 3] = \
                [ball[0] * scale, ball[1] * scale, self.ball_radius]
            self.data.qvel[self.ball_dofadr:self.ball_dofadr + 6] = 0
        self._settle(0.1)
        if self.logger:
            self.logger.info("Ball left the reachable workspace - clamped back in")

    def apply_action_polar(self, angle_deg, dist, dz=None):
        """2D-sim-like action: rotate heading, then advance dist meters in the plane.
        dz, when given, moves the EE vertically relative to its current height
        (clamped to z_bounds); when None the height stays pinned to hover_z."""
        self.heading = self.wrap_angle(self.heading + angle_deg)
        ee = self.ee_pos()
        tx = ee[0] + dist * math.cos(math.radians(self.heading))
        ty = ee[1] + dist * math.sin(math.radians(self.heading))
        if dz is None:
            tz = self.hover_z
        else:
            tz = min(max(ee[2] + dz, self.z_bounds[0]), self.z_bounds[1])
        self.move_ee_to(tx, ty, tz, self.heading)
        self.update_grasp()
        self._rescue_ball()

    def apply_action_absolute(self, x, y, z, yaw_deg):
        self.heading = self.wrap_angle(yaw_deg)
        self.move_ee_to(x, y, z, yaw_deg)
        self.update_grasp()
        self._rescue_ball()

    def _set_ball_collisions(self, enabled):
        """Enable/disable the ball's contacts. Disabled while magnet-grasped so
        the follow-the-flange teleport can't drive it into the arm/box geoms
        and produce NaN/huge QACC on its rotation DOFs."""
        if enabled:
            self.model.geom_contype[self.ball_geom_id] = self._ball_contype
            self.model.geom_conaffinity[self.ball_geom_id] = self._ball_conaffinity
        else:
            self.model.geom_contype[self.ball_geom_id] = 0
            self.model.geom_conaffinity[self.ball_geom_id] = 0

    # ------------------------------------------------------------- autograsp
    def update_grasp(self):
        ee, ball, box = self.ee_pos(), self.ball_pos(), self.box_pos()
        if not self.grasped:
            # 3D distance: with z as a free action, planar proximity is not enough
            if np.linalg.norm(ee - ball) < self.grasp_range and not self.ball_in_box():
                with self.lock:
                    self.grasped = True
                    self._set_ball_collisions(False)  # held: no physical contacts
                if self.logger:
                    self.logger.info("AUTOGRASP - Ball grasped")
        else:
            if self.get_distance(ee, box) < self.place_range:
                with self.lock:
                    self.grasped = False
                    self._set_ball_collisions(True)   # released: contacts back on
                self._settle(0.5)  # let the ball drop into the box
                if self.logger:
                    self.logger.info("AUTOGRASP - Ball released over box")


class UR5GripperScenario(UR5Scenario):
    """
    UR5e + Robotiq 2F-85 (both from MuJoCo Menagerie, unchanged): physical grasp, no magnet. The ball is held
    only by the friction of the finger pads.

    - Control point: the gripper's pinch site (between the fingertips). The tool points down and the fingers'
      closing axis is kept perpendicular to the heading, so the gripper orientation is a function of the heading
      that the perceptions are relative to. The gripper is symmetric, so of the two yaws that do this the one
      closer to the current one is used (at most a 90 degree turn), unless it would wind wrist 3 too far.
    - Motions are straight Cartesian lines (like URScript movel), split into waypoints solved with IK and run
      at the UR5e's limits (joint speed MAX_JOINT_SPEED, tool speed TOOL_SPEED).
    - The grip command is a grasp attempt, run after the motion: close until the fingers stop; if both pads touch
      the ball, lift ``verify_lift`` and check that both still touch it and the ball left the floor (like checking
      Robotiq's object-detected status after a pick). If the check fails, open again. A grip command at or below
      0.5 while closed opens the gripper (release).
    """

    SITE_NAME = "gripper_pinch"
    magnet = False
    GRIPPER_ACTUATOR = "gripper_fingers_actuator"
    DRIVER_JOINT = "gripper_right_driver_joint"
    PAD_GEOMS = {"left": ("gripper_left_pad1", "gripper_left_pad2"),
                 "right": ("gripper_right_pad1", "gripper_right_pad2")}
    CTRL_OPEN, CTRL_CLOSED = 0.0, 255.0
    MAX_JOINT_SPEED = math.pi  # rad/s: UR5e limit, all joints (180 deg/s)
    TOOL_SPEED = 0.5           # m/s along the line (URScript movel allows up to ~1 m/s)
    WAYPOINT_STEP = 0.01       # m between IK waypoints
    WAYPOINT_YAW = math.radians(5)
    ROT_W = 0.3                # weight of the orientation error in the IK
    WRIST3_LIMIT = 1.5 * math.pi  # keep wrist 3 within +-270 deg (its range is +-360 deg)
    SETTLE_MAX = 3.0           # s of simulated time to wait for the arm to settle after a motion

    def __init__(self, model_path, verify_lift=0.03, z_bounds=(0.008, 0.40), **kwargs):
        self.verify_lift = verify_lift
        super().__init__(model_path, z_bounds=z_bounds, **kwargs)

    def _setup_extra(self):
        m = self.model
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.GRIPPER_ACTUATOR)
        drv = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, self.DRIVER_JOINT)
        if self.grip_act < 0 or drv < 0:
            raise ValueError(f"{self.GRIPPER_ACTUATOR} / {self.DRIVER_JOINT} not in the model: is it the 2F-85 scene?")
        self.driver_dofadr = m.jnt_dofadr[drv]
        self.pad_ids = {side: {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names}
                        for side, names in self.PAD_GEOMS.items()}
        self.wrist3_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist_3_joint")]
        self.closed = False  # gripper commanded closed

    def _physics_step(self):
        """Step with gravity (and Coriolis) compensation on the arm joints, as the UR controller does: the
        menagerie position servos alone sag about 1 cm at the pinch point under the 0.9 kg gripper, more than
        the finger pads tolerate."""
        with self.lock:
            self.data.qfrc_applied[:self.nq_arm] = self.data.qfrc_bias[:self.nq_arm]
        super()._physics_step()

    # ------------------------------------------------------------- state
    def get_state(self):
        state = super().get_state()
        with self.lock:
            state["closed"] = self.closed
        return state

    def pads_touching_ball(self):
        """Which finger pads touch the ball, as {"left": bool, "right": bool}."""
        touching = {"left": False, "right": False}
        ncon = self.data.ncon
        if ncon == 0:
            return touching
        b = self.ball_geom_id
        for g1, g2 in self.data.contact.geom[:ncon]:
            other = g2 if g1 == b else g1 if g2 == b else None
            if other is None:
                continue
            for side, ids in self.pad_ids.items():
                if other in ids:
                    touching[side] = True
        return touching

    def is_holding(self):
        t = self.pads_touching_ball()
        return t["left"] and t["right"]

    def tool_yaw(self):
        """Yaw (rad) of the pinch frame's x axis (the heading direction when aligned)."""
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    @staticmethod
    def down_rotation(yaw):
        """Pinch frame for a top-down tool, as a rotation matrix (columns x, y, z): x = (cos, sin, 0) (heading),
        z = down, y = z cross x (the closing axis)."""
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])

    # ------------------------------------------------------------- IK
    def _solve_ik_pose(self, p_des, R_des, seed_qpos, iters=200, tol=5e-4, rot_tol=5e-3):
        """DLS IK for the full pinch pose (position + orientation), from a seed.
        Returns (joint config, position residual in m, rotation residual in rad)."""
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qpos[:self.nq_arm] = seed_qpos
        jnt_range = self.model.jnt_range[:self.nq_arm]
        quat, rot_err = np.zeros(4), np.zeros(3)
        p_err = np.zeros(3)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)
            p_err = p_des - self.ik_data.site_xpos[self.site_id]
            R_cur = self.ik_data.site_xmat[self.site_id].reshape(3, 3)
            mujoco.mju_mat2Quat(quat, (R_des @ R_cur.T).flatten())
            mujoco.mju_quat2Vel(rot_err, quat, 1.0)  # world-frame rotation vector
            if np.linalg.norm(p_err) < tol and np.linalg.norm(rot_err) < rot_tol:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.site_id)
            J = np.vstack([jacp, self.ROT_W * jacr])[:, :self.nq_arm]
            err6 = np.concatenate([p_err, self.ROT_W * rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + self.DAMPING * np.eye(6), err6)
            dq = np.clip(dq, -0.2, 0.2)
            self.ik_data.qpos[:self.nq_arm] = np.clip(
                self.ik_data.qpos[:self.nq_arm] + dq, jnt_range[:, 0], jnt_range[:, 1])
        return (self.ik_data.qpos[:self.nq_arm].copy(), float(np.linalg.norm(p_err)),
                float(np.linalg.norm(rot_err)))

    def _yaw_change(self, target_yaw):
        """Tool yaw change (rad) to reach target_yaw or target_yaw + 180 deg (the same grasp, the gripper is
        symmetric): the smaller turn, unless it would take wrist 3 beyond WRIST3_LIMIT. Call with the lock held."""
        delta = self.wrap_angle(math.degrees(target_yaw - self.tool_yaw()))
        if delta > 90.0:
            delta -= 180.0
        elif delta < -90.0:
            delta += 180.0
        alt = delta - 180.0 if delta >= 0.0 else delta + 180.0
        # How wrist 3 moves with the tool yaw: the z component of its angular Jacobian column (about +-1 tool down)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        dyaw_dq3 = jacr[2, self.wrist3_dof]
        q3 = self.data.qpos[self.wrist3_dof]
        if abs(dyaw_dq3) > 0.5 and abs(q3 + math.radians(delta) / dyaw_dq3) > self.WRIST3_LIMIT:
            delta = alt
        return math.radians(delta)

    # ------------------------------------------------------------- motion
    def move_ee_to(self, x, y, z, yaw_deg=0.0, duration=None, tol=0.001):
        """Move the pinch point to (x, y, z) along a straight line, tool down, closing axis perpendicular to the
        heading yaw_deg. The line is solved with IK every WAYPOINT_STEP (and WAYPOINT_YAW) and run at the UR5e
        limits; then the arm settles on the last waypoint. ``duration`` is unused (kept for the interface)."""
        r = math.hypot(x, y)
        r_cl = min(max(r, self.workspace_r[0]), self.workspace_r[1])
        if r > 1e-6 and r_cl != r:
            x, y = x * r_cl / r, y * r_cl / r
        z = min(max(z, self.z_bounds[0]), self.z_bounds[1])
        p_goal = np.array([x, y, z])

        with self.lock:
            p0 = self.ee_pos()
            yaw0 = self.tool_yaw()
            dyaw = self._yaw_change(math.radians(yaw_deg))
            q_prev = self.data.qpos[:self.nq_arm].copy()
        n = max(1, math.ceil(np.linalg.norm(p_goal - p0) / self.WAYPOINT_STEP),
                math.ceil(abs(dyaw) / self.WAYPOINT_YAW))
        waypoints = []
        q = q_prev
        res = rot_res = 0.0
        for i in range(1, n + 1):
            s = i / n
            q, res, rot_res = self._solve_ik_pose(p0 + s * (p_goal - p0), self.down_rotation(yaw0 + s * dyaw), q)
            waypoints.append(q)
        if (res > 0.01 or rot_res > 0.05) and self.logger:
            self.logger.warn(f"Target not reachable: IK residual {res*100:.1f} cm, "
                             f"{math.degrees(rot_res):.1f} deg at {np.round(p_goal, 3)}")

        lo = self.model.actuator_ctrlrange[:self.nq_arm, 0]
        hi = self.model.actuator_ctrlrange[:self.nq_arm, 1]
        dt = self.model.opt.timestep
        seg_len = np.linalg.norm(p_goal - p0) / n
        for q_next in waypoints:
            T = max(np.max(np.abs(q_next - q_prev)) / self.MAX_JOINT_SPEED, seg_len / self.TOOL_SPEED, dt)
            steps = max(1, int(round(T / dt)))
            for k in range(1, steps + 1):
                with self.lock:
                    self.data.ctrl[:self.nq_arm] = np.clip(q_prev + (q_next - q_prev) * k / steps, lo, hi)
                self._physics_step()
            q_prev = q_next
        # Settle on the last waypoint: the menagerie servos are overdamped (kd/kp = 0.2 s), so they lag behind a
        # moving target. Simulated time is cheap (~25 us per step), so wait until the tool is there and still.
        for _ in range(int(self.SETTLE_MAX / dt)):
            self._physics_step()
            if (np.linalg.norm(p_goal - self.ee_pos()) < tol
                    and np.max(np.abs(self.data.qvel[:self.nq_arm])) < 0.01):
                break
        return self.ee_pos()

    # ------------------------------------------------------------- gripper
    def set_gripper(self, close):
        """Command the 2F-85 fully open or closed and step until the fingers stop (on an object, or at the end)."""
        with self.lock:
            self.data.ctrl[self.grip_act] = self.CTRL_CLOSED if close else self.CTRL_OPEN
        dt = self.model.opt.timestep
        still = 0
        for k in range(int(1.5 / dt)):
            self._physics_step()
            if k * dt < 0.05:
                continue  # let the fingers start moving
            still = still + 1 if abs(self.data.qvel[self.driver_dofadr]) < 0.02 else 0
            if still * dt >= 0.03:
                break
        self.closed = close

    def grasp_attempt(self):
        """Close; if both pads touch the ball, lift verify_lift and check it is still held off the floor.
        On failure, open again. Returns True if the ball is held."""
        self.set_gripper(True)
        if self.is_holding():
            p = self.ee_pos()
            self.move_ee_to(p[0], p[1], p[2] + self.verify_lift, self.heading)
            if self.is_holding() and self.ball_pos()[2] > self.ball_radius + 0.5 * self.verify_lift:
                if self.logger:
                    self.logger.info("GRASP - Ball held (both pads, lifted)")
                return True
            if self.logger:
                self.logger.info("GRASP - Ball slipped during the verification lift")
        self.set_gripper(False)
        return False

    # ------------------------------------------------------------- actions
    def apply_action_polar(self, angle_deg, dist, dz=None, grip=None):
        """Turn the heading, move dist along it (and dz vertically, or to hover_z), then run the grip command."""
        self.heading = self.wrap_angle(self.heading + angle_deg)
        ee = self.ee_pos()
        tx = ee[0] + dist * math.cos(math.radians(self.heading))
        ty = ee[1] + dist * math.sin(math.radians(self.heading))
        tz = self.hover_z if dz is None else ee[2] + dz
        self.move_ee_to(tx, ty, tz, self.heading)
        if grip is not None:
            if grip > 0.5 and not self.closed:
                self.grasp_attempt()
            elif grip <= 0.5 and self.closed:
                self.set_gripper(False)
        with self.lock:
            self.grasped = self.closed and self.is_holding()
        self._rescue_ball()

    def apply_action_absolute(self, x, y, z, yaw_deg):
        self.heading = self.wrap_angle(yaw_deg)
        self.move_ee_to(x, y, z, yaw_deg)
        with self.lock:
            self.grasped = self.closed and self.is_holding()
        self._rescue_ball()

    def update_grasp(self):
        """No magnet: holding is decided by the pad contacts (see grasp_attempt)."""
        pass

    # ------------------------------------------------------------- reset
    def reset(self, rng=None):
        with self.lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # "home" keyframe, gripper open
            self.data.ctrl[:self.nq_arm] = self.data.qpos[:self.nq_arm]
            self.data.ctrl[self.grip_act] = self.CTRL_OPEN
            self.grasped = False
            self.closed = False
            self._place_ball_randomly(rng)
            mujoco.mj_forward(self.model, self.data)
        self._settle(0.3)
        self.heading = 0.0
        self.move_ee_to(0.45, 0.0, self.hover_z, 0.0)


# ----------------------------------------------------------------------------- 2FG7, smooth motion
class UR5TwoFG7Scenario(UR5GripperScenario):
    """
    UR5e + OnRobot 2FG7, the lab's gripper (OnRobot's official model; silicone fingertips mounted outwards,
    35-73 mm), with motions like the real controller's and no grasp macro.

    Motion. Every physics step (2 ms, the UR5e's 500 Hz control rate) a jerk-limited online trajectory generator
    (Ruckig) moves a setpoint for the pinch point (x, y, z and the tool yaw; the tool always points down) towards
    the current target, within the tool speed and acceleration limits (by default URScript movel's 0.25 m/s and
    1.2 m/s^2). The setpoint is solved with IK and sent to the joint servos with velocity feedforward, so they
    track it like the UR's joint controllers instead of lagging 0.2 s behind, plus gravity compensation. A new
    target can be set while the arm moves: the setpoint bends into it from its current velocity, it does not jump
    or stop.

    Decisions. An action sets the next target, relative to where the pinch point is when the action starts (so an
    action means what it did before: "turn the heading, move this much along it and up/down"). The action ends,
    and e-MDB decides the next one, ``blend_time`` before the end of the motion, while the arm still moves (like
    the blend radius of a UR program). An action that closes the gripper ends at rest instead, as in a pick: the
    arm stops at the target, the fingers close, and the action ends when they stop (on the ball or at the closed
    limit).

    Gripper. The 2FG7 opens and closes at ``grip_speed`` and grips with ``grip_force``: its position servo is
    limited to that force, so a blocked finger presses with it, like the real force-controlled grip. arm:grip >
    0.5 closes, otherwise it opens (an opening starts with the motion). Nothing is automatic: to lift the ball the
    learner closes and then moves up, and after a miss it has to open. ``ball_lifted`` is 1 when both pads touch
    the ball and its bottom is ``lift_height`` above the table.

    Clock. realtime False (default): physics runs only during actions and resets, as fast as it can; between
    actions the simulation is frozen, arm velocity included, so the motion is continuous in simulated time.
    realtime True: physics runs all the time in its own thread at ``speed`` times real time; while e-MDB decides,
    the arm goes on towards the last target (and stops there smoothly if the decision comes late), as the real
    robot would. ``video_dir``: record every ``video_every``-th trial to an mp4, one frame per 1/30 s of
    simulated time (real speed, whatever the clock). The recorder starts before the viewer window opens; if it
    cannot record, the run goes on without video (ERROR "Video recording disabled" in the log).
    """

    SITE_NAME = "gripper_pinch"
    GRIPPER_ACTUATOR = "gripper_fingers_actuator"
    PAD_GEOMS = {"left": ("gripper_left_pad",), "right": ("gripper_right_pad",)}
    FINGER_JOINTS = ("gripper_left_finger_joint", "gripper_right_finger_joint")  # each = half the aperture
    TOOL_JERK = 10.0              # m/s^3 (UR does not publish it; this gives 0.1 s acceleration ramps)
    YAW_SPEED, YAW_ACCEL, YAW_JERK = 1.5, 4.0, 40.0  # tool rotation: rad/s, rad/s^2, rad/s^3
    AXIS_FLOOR = 0.3              # per-axis limits are the tool limits times the direction, at least this fraction
    SETTLE_TOL = 0.001            # m: "arrived" for the actions that end at rest
    SETTLE_MAX = 1.0              # s of simulated time to wait for that after the setpoint arrived (blocked arm)
    GRIP_TIMEOUT = 2.0            # s
    ACTION_TIMEOUT = 8.0          # s of simulated time
    FLOOR_MARGIN = 0.002          # m between the fingertips and the table at the lowest allowed pinch height
    VIDEO_FPS, VIDEO_SIZE = 30, (1280, 720)
    VIDEO_STALL = 30.0            # s (wall clock): a recorder that takes no frame for this long is given up

    def __init__(self, model_path, grip_force=20.0, grip_speed=0.1, lift_height=0.05, tool_speed=0.25,
                 tool_accel=1.2, blend_time=0.15, video_dir="", video_every=1, **kwargs):
        from ruckig import InputParameter, OutputParameter, Ruckig, Synchronization  # only this scenario needs it
        self._ruckig = (InputParameter, OutputParameter, Ruckig, Synchronization)
        self.grip_force, self.grip_speed, self.lift_height = float(grip_force), float(grip_speed), float(lift_height)
        self.tool_speed, self.tool_accel, self.blend_time = float(tool_speed), float(tool_accel), float(blend_time)
        self.video_dir, self.video_every = video_dir, max(1, int(video_every))
        self._physics_thread = None   # realtime: the thread that steps the physics
        self._video_q = None          # video: queue of the recording thread
        self._video_ok = False        # video: the recorder is running (only then are frames queued)
        self._recording = False       # video: the current trial is being recorded
        self._trial = -1
        self._q_start = None          # arm configuration of the start pose (computed once)
        kwargs.pop("verify_lift", None)
        super().__init__(model_path, **kwargs)  # the base constructor runs the first reset
        if self.realtime:
            self._running = True
            self._physics_thread = threading.Thread(target=self._physics_loop, daemon=True, name="ur5_physics")
            self._physics_thread.start()

    def _setup_extra(self):
        m = self.model
        InputParameter, OutputParameter, Ruckig, Synchronization = self._ruckig
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.GRIPPER_ACTUATOR)
        if self.grip_act < 0:
            raise ValueError(f"{self.GRIPPER_ACTUATOR} not in the model: is it the 2FG7 scene?")
        self.pad_ids = {side: {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names}
                        for side, names in self.PAD_GEOMS.items()}
        if any(-1 in ids for ids in self.pad_ids.values()):
            raise ValueError(f"Pad geoms {self.PAD_GEOMS} not in the model")
        self.wrist3_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wrist_3_joint")]
        self.aperture_min, self.aperture_max = (float(v) for v in m.actuator_ctrlrange[self.grip_act])
        self.finger_qposadr = [m.jnt_qposadr[j] for j in
                               (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.FINGER_JOINTS) if j >= 0]
        m.actuator_forcelimited[self.grip_act] = 1
        m.actuator_forcerange[self.grip_act] = [-self.grip_force, self.grip_force]
        # Arm servos: force = kp (ctrl - q) - kd qdot. ctrl = q_des + (kd/kp) qdot_des makes them PD trackers
        self.ff_tau = -m.actuator_biasprm[:self.nq_arm, 2] / m.actuator_gainprm[:self.nq_arm, 0]
        self.ctrl_lo = m.actuator_ctrlrange[:self.nq_arm, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:self.nq_arm, 1].copy()
        self.closed = False
        self.grip_cmd = self.grip_ctrl = self.aperture_max
        self._grip_cmd_time = 0.0
        self._grip_still = 0.0
        self._last_aperture = self.aperture_max
        self.otg = Ruckig(4, m.opt.timestep)
        self.otg_in, self.otg_out = InputParameter(4), OutputParameter(4)
        self.otg_in.synchronization = Synchronization.Phase  # straight lines when starting at rest
        self._otg_active = False
        self._otg_new = False
        self._finished_at = 0.0
        self._ik_worst = 0.0
        self.q_des = self.data.qpos[:self.nq_arm].copy()
        self.qd_des = np.zeros(self.nq_arm)
        # Lowest allowed pinch height: the fingertips' lowest point (tool down) FLOOR_MARGIN above the table
        mujoco.mj_forward(m, self.data)
        R_site = self.data.site_xmat[self.site_id].reshape(3, 3)
        below = 0.0
        for ids in self.pad_ids.values():
            for g in ids:
                Rg, half = self.data.geom_xmat[g].reshape(3, 3), m.geom_size[g]
                for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                    p = self.data.geom_xpos[g] + Rg @ (corner * half)
                    below = max(below, float((R_site.T @ (p - self.data.site_xpos[self.site_id]))[2]))
        self.z_bounds = (below + self.FLOOR_MARGIN, self.z_bounds[1])
        if self.video_dir:
            self._start_recorder()  # here, because the base constructor opens the viewer right after this

    # ------------------------------------------------------------- physics and control
    def _control_tick(self):
        """One control period (call with the lock held): advance the setpoint, IK, joint and gripper commands."""
        dt = self.model.opt.timestep
        if self._otg_active:
            result = self.otg.update(self.otg_in, self.otg_out)
            self.otg_out.pass_to_input(self.otg_in)
            self._otg_new = False  # otg_out now describes the trajectory to the new target
            if int(result) < 0:
                if self.logger:
                    self.logger.error(f"Trajectory generator error {result}: stopping the motion")
                self._otg_active = False
            elif int(result) == 1:  # Finished
                self._otg_active = False
                self._finished_at = self.data.time
            sp = self.otg_out.new_position
            q, res, _ = self._solve_ik_pose(np.array(sp[:3]), self.down_rotation(sp[3]), self.q_des,
                                            iters=20, tol=1e-4, rot_tol=1e-3)
            self._ik_worst = max(self._ik_worst, res)
            step = self.MAX_JOINT_SPEED * dt
            q = self.q_des + np.clip(q - self.q_des, -step, step)
            self.qd_des = (q - self.q_des) / dt
            self.q_des = q
        else:
            self.qd_des[:] = 0.0
        self.data.ctrl[:self.nq_arm] = np.clip(self.q_des + self.ff_tau * self.qd_des, self.ctrl_lo, self.ctrl_hi)
        # Gripper: the servo setpoint moves at grip_speed; a blocked finger presses with the force limit
        g = self.grip_speed * dt
        self.grip_ctrl += float(np.clip(self.grip_cmd - self.grip_ctrl, -g, g))
        self.data.ctrl[self.grip_act] = self.grip_ctrl
        aperture = float(self.data.actuator_length[self.grip_act])
        self._grip_still = self._grip_still + dt if abs(aperture - self._last_aperture) < 0.001 * dt else 0.0
        self._last_aperture = aperture
        # Gravity (and Coriolis) compensation on the arm, as the UR controller does
        self.data.qfrc_applied[:self.nq_arm] = self.data.qfrc_bias[:self.nq_arm]

    def _physics_step(self):
        step_start = time.perf_counter()
        frame = None
        with self.lock:
            self._control_tick()
            qpos_backup = self.data.qpos.copy()
            mujoco.mj_step(self.model, self.data)
            if not self._state_is_sane():  # as in UR5Scenario: roll back a diverged step
                self.data.qpos[:] = qpos_backup
                self.data.qvel[:] = 0.0
                self.data.qacc[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self._instability_count += 1
                if self.logger and self._instability_count % 100 == 1:
                    self.logger.warn(f"Physics instability caught and rolled back (total={self._instability_count})")
            if self._recording and self.data.time >= self._next_frame:
                self._next_frame += 1.0 / self.VIDEO_FPS
                frame = self.data.qpos.copy()
        if frame is not None:
            self._video_put(("frame", frame))  # waits while the recorder is behind, never for a stopped one
        now = time.perf_counter()
        if self.viewer is not None and self.viewer.is_running() and now - self._last_view_sync >= 1.0 / 30.0:
            self.viewer.sync()
            self._last_view_sync = time.perf_counter()
        if self.realtime:
            self._pace_debt += self.model.opt.timestep / self.speed - (time.perf_counter() - step_start)
            if self._pace_debt >= 0.002:
                sleep_start = time.perf_counter()
                time.sleep(self._pace_debt)
                self._pace_debt -= time.perf_counter() - sleep_start
            elif self._pace_debt < -0.1:
                self._pace_debt = -0.1

    def _physics_loop(self):
        while self._running:
            self._physics_step()

    def _run_until(self, done, max_time):
        """Run the simulation until done() (called with the lock held) or max_time of simulated time.
        Without the physics thread this steps the physics itself; with it, it waits for the thread."""
        with self.lock:
            t_end = self.data.time + max_time
        while True:
            if self._physics_thread is None:
                self._physics_step()
            else:
                time.sleep(0.001)
            with self.lock:
                if done():
                    return True
                if self.data.time >= t_end:
                    return False

    def _settle(self, seconds):
        self._run_until(lambda: False, seconds)

    # ------------------------------------------------------------- motion commands (lock held)
    def _sync_setpoint_to_robot(self):
        """Setpoint = where the tool is, at rest (after a reset)."""
        ee, yaw = self.ee_pos(), self.tool_yaw()
        pose = [float(ee[0]), float(ee[1]), float(ee[2]), yaw]
        self.otg_in.current_position = pose
        self.otg_in.current_velocity = [0.0] * 4
        self.otg_in.current_acceleration = [0.0] * 4
        self.otg_in.target_position = pose
        self.otg_in.target_velocity = [0.0] * 4
        self.otg_in.target_acceleration = [0.0] * 4
        self.q_des = self.data.qpos[:self.nq_arm].copy()
        self.qd_des = np.zeros(self.nq_arm)
        self._otg_active = False

    def _command_move(self, x, y, z, heading_deg):
        """New target for the pinch point (clamped to the workspace), closing axis perpendicular to heading_deg."""
        r = math.hypot(x, y)
        r_cl = min(max(r, self.workspace_r[0]), self.workspace_r[1])
        if r > 1e-6 and r_cl != r:
            x, y = x * r_cl / r, y * r_cl / r
        z = min(max(z, self.z_bounds[0]), self.z_bounds[1])
        cur = np.array(self.otg_in.current_position)
        yaw = float(cur[3] + self._yaw_change(math.radians(heading_deg)))
        d = np.array([x, y, z]) - cur[:3]
        L = float(np.linalg.norm(d))
        frac = np.maximum(np.abs(d) / L if L > 1e-9 else np.ones(3), self.AXIS_FLOOR)
        self.otg_in.max_velocity = [*(self.tool_speed * frac), self.YAW_SPEED]
        self.otg_in.max_acceleration = [*(self.tool_accel * frac), self.YAW_ACCEL]
        self.otg_in.max_jerk = [*(self.TOOL_JERK * frac), self.YAW_JERK]
        self.otg_in.target_position = [float(x), float(y), float(z), yaw]
        self.otg_in.target_velocity = [0.0] * 4
        self.otg_in.target_acceleration = [0.0] * 4
        self._otg_active = True
        self._otg_new = True
        self._ik_worst = 0.0
        self._target = np.array([x, y, z])

    def _command_gripper(self, close):
        self.grip_cmd = self.aperture_min if close else self.aperture_max
        self.closed = close
        self._grip_cmd_time = self.data.time
        self._grip_still = 0.0

    # ------------------------------------------------------------- end-of-action conditions (lock held)
    def _near_end(self):
        if self._otg_new:  # the generator has not taken the new target yet (realtime: the physics thread)
            return False
        return (not self._otg_active
                or self.otg_out.trajectory.duration - self.otg_out.time <= self.blend_time)

    def _arrived(self):
        if self._otg_active:
            return False
        still = (np.linalg.norm(self._target - self.ee_pos()) < self.SETTLE_TOL
                 and np.max(np.abs(self.data.qvel[:self.nq_arm])) < 0.01)
        return still or self.data.time - self._finished_at > self.SETTLE_MAX  # blocked (e.g. by the box)

    def _gripper_stopped(self):
        """Fingers still, and either pressing with the grip force (on an object) or at the commanded end."""
        if self.data.time - self._grip_cmd_time < 0.05 or self._grip_still < 0.03:
            return False
        pressing = abs(self.data.actuator_force[self.grip_act]) >= 0.95 * self.grip_force
        return pressing or abs(self.grip_ctrl - self.grip_cmd) < 1e-9

    # ------------------------------------------------------------- state
    def ball_lifted(self):
        """Both pads on the ball and its bottom lift_height above the table (call with the lock held)."""
        return self.is_holding() and self.ball_pos()[2] - self.ball_radius >= self.lift_height

    def get_state(self):
        state = UR5Scenario.get_state(self)
        with self.lock:
            state.update(closed=self.closed,
                         aperture=float(self.data.actuator_length[self.grip_act]),
                         grip_force=float(abs(self.data.actuator_force[self.grip_act])),
                         lifted=self.ball_lifted())
        return state

    # ------------------------------------------------------------- actions
    def _end_action(self):
        if self._ik_worst > 0.01 and self.logger:
            self.logger.warn(f"Target not reachable: IK residual {self._ik_worst*100:.1f} cm at "
                             f"{np.round(self._target, 3)}")
        with self.lock:
            self.grasped = self.is_holding()  # (UR5Scenario._rescue_ball leaves a held ball alone)
        self._rescue_ball()

    def apply_action_polar(self, angle_deg, dist, dz=None, grip=None):
        """Turn the heading, move dist along it (and dz up/down, or to hover_z) and open/close the gripper."""
        close = grip is not None and grip > 0.5
        with self.lock:
            self.heading = self.wrap_angle(self.heading + angle_deg)
            ee = self.ee_pos()
            tx = ee[0] + dist * math.cos(math.radians(self.heading))
            ty = ee[1] + dist * math.sin(math.radians(self.heading))
            tz = self.hover_z if dz is None else ee[2] + dz
            if grip is not None and not close and self.closed:
                self._command_gripper(False)  # opening starts with the motion
            self._command_move(tx, ty, tz, self.heading)
        if close and not self.closed:
            # A pick: stop at the target, close, and end when the fingers stop
            self._run_until(self._arrived, self.ACTION_TIMEOUT)
            with self.lock:
                self._command_gripper(True)
            self._run_until(self._gripper_stopped, self.GRIP_TIMEOUT)
        else:
            self._run_until(self._near_end, self.ACTION_TIMEOUT)
        self._end_action()

    def move_ee_to(self, x, y, z, yaw_deg=0.0, duration=None, tol=None):
        """Move to (x, y, z) with heading yaw_deg and wait until the arm is there and at rest."""
        with self.lock:
            self._command_move(x, y, z, yaw_deg)
        self._run_until(self._arrived, self.ACTION_TIMEOUT)
        return self.ee_pos()

    def apply_action_absolute(self, x, y, z, yaw_deg):
        self.heading = self.wrap_angle(yaw_deg)
        self.move_ee_to(x, y, z, yaw_deg)
        self._end_action()

    # ------------------------------------------------------------- reset
    def reset(self, rng=None):
        """New trial: arm at the start pose (tool down, pinch point at (0.45, 0, hover_z)), gripper open, ball
        placed at random. The start pose is set directly (the ball is teleported too), not driven to."""
        self._end_video()
        with self.lock:
            if self.model.nkey > 0 and self.model.key_qpos.shape[1] == self.model.nq:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            else:
                mujoco.mj_resetData(self.model, self.data)
            if self._q_start is None:
                self._q_start = self._solve_ik_pose(np.array([0.45, 0.0, self.hover_z]), self.down_rotation(0.0),
                                                    self.data.qpos[:self.nq_arm].copy(), iters=500)[0]
            self.data.qpos[:self.nq_arm] = self._q_start
            self.data.ctrl[:self.nq_arm] = self._q_start
            for adr in self.finger_qposadr:  # open (the keyframe has them closed)
                self.data.qpos[adr] = self.aperture_max / 2
            self.grip_cmd = self.grip_ctrl = self.aperture_max
            self.data.ctrl[self.grip_act] = self.aperture_max
            self.grasped = False
            self.closed = False
            self.heading = 0.0
            self._place_ball_randomly(rng)
            mujoco.mj_forward(self.model, self.data)
            self._sync_setpoint_to_robot()
            self._target = self.ee_pos()
        self._settle(0.3)  # fingers open, ball settles on the table
        self._trial += 1
        self._start_video()

    # ------------------------------------------------------------- video
    def _start_recorder(self):
        """Start the recording thread and wait until its offscreen renderer is ready. It must run before the viewer
        window opens: once the viewer's OpenGL context exists, the recorder's EGL context cannot be made current
        (it fails every time; the other order works). If the recorder cannot start, the run
        goes on without video."""
        import queue
        os.makedirs(self.video_dir, exist_ok=True)
        self._video_q = queue.Queue(maxsize=60)
        self._video_abandoned = False
        ready = threading.Event()
        threading.Thread(target=self._video_loop, args=(ready,), daemon=True, name="ur5_video").start()
        if not ready.wait(30.0):
            self._video_abandoned = True
            self._video_failed("the offscreen renderer did not start within 30 s")

    def _video_failed(self, why):
        """Recording off for the rest of the run: no more frames are queued, so the physics never waits for it."""
        self._video_ok = False
        self._recording = False
        if self.logger:
            self.logger.error(f"Video recording disabled: {why}")
        else:
            print(f"Video recording disabled: {why}", flush=True)

    def _video_put(self, item):
        """Queue an item for the recorder. While the queue is full this waits (no frame is lost when the recorder
        is behind), but never for a recorder that has stopped, and at most VIDEO_STALL s without progress."""
        import queue
        waited = 0.0
        while self._video_ok:
            try:
                self._video_q.put(item, timeout=0.5)
                return
            except queue.Full:
                waited += 0.5
                if waited >= self.VIDEO_STALL:
                    self._video_failed(f"the recorder took no frame for {self.VIDEO_STALL:.0f} s")

    def _start_video(self):
        if not self._video_ok or self._trial % self.video_every:
            self._recording = False
            return
        with self.lock:
            self._next_frame = self.data.time
            self._recording = True
        self._video_put(("start", os.path.join(self.video_dir, f"trial_{self._trial:05d}.mp4")))

    def _end_video(self):
        if self._recording:
            with self.lock:
                self._recording = False
            self._video_put(("end", None))

    def _video_loop(self, ready):
        """Recording thread: renders the queued configurations offscreen and pipes them to ffmpeg. An error stops
        the recording (logged), never the simulation."""
        import subprocess
        w, h = self.VIDEO_SIZE
        try:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
            renderer = mujoco.Renderer(self.model, height=h, width=w)
        except Exception as e:  # e.g. "Failed to make the EGL context current."
            self._video_failed(f"no offscreen renderer: {e} (MUJOCO_GL={os.environ.get('MUJOCO_GL', '')!r})")
            ready.set()
            return
        if self._video_abandoned:  # _start_recorder gave up waiting
            return
        self._video_ok = True
        ready.set()
        data = mujoco.MjData(self.model)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.38, -0.08, 0.12]  # the workspace in front of the robot, box included
        cam.distance, cam.azimuth, cam.elevation = 1.05, 150.0, -22.0
        proc, path = None, None
        try:
            while True:
                kind, item = self._video_q.get()
                if kind == "start":
                    path = item  # the file is created with the first frame (a trial without motion has none)
                elif kind == "frame" and proc is None and path is not None:
                    proc = subprocess.Popen(
                        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                         f"{w}x{h}", "-r", str(self.VIDEO_FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         "-preset", "veryfast", "-crf", "23", path], stdin=subprocess.PIPE)
                if kind == "frame" and proc is not None:
                    data.qpos[:] = item
                    mujoco.mj_forward(self.model, data)
                    renderer.update_scene(data, camera=cam)
                    proc.stdin.write(renderer.render().tobytes())
                elif kind in ("end", "close"):
                    if proc is not None:
                        proc.stdin.close()
                        proc.wait()
                    proc, path = None, None
                if kind == "close":
                    return
        except Exception as e:  # ffmpeg missing or gone, a rendering error...
            self._video_failed(f"{type(e).__name__}: {e}")
            if proc is not None:
                proc.kill()

    def close(self):
        """Stop the physics thread and finish the video being recorded (call on shutdown)."""
        self._running = False
        if self._physics_thread is not None:
            self._physics_thread.join(timeout=1.0)
        self._end_video()
        self._video_put(("close", None))


# ----------------------------------------------------------------------------- ROS2 node
class SimUR5Mujoco(Node):
    """MuJoCo UR5e simulator node for e-MDB. Mirrors Sim2DSimple's interface."""

    def __init__(self):
        super().__init__("sim_ur5_mujoco")

        self.random_seed = (
            self.declare_parameter("random_seed", value=0)
            .get_parameter_value().integer_value
        )
        self.config_file = (
            self.declare_parameter("config_file",
                                   descriptor=ParameterDescriptor(dynamic_typing=True))
            .get_parameter_value().string_value
        )
        visualize = (
            self.declare_parameter("visualize", value=True)
            .get_parameter_value().bool_value
        )
        self.standalone = (
            self.declare_parameter("standalone", value=False)
            .get_parameter_value().bool_value
        )
        realtime = (
            self.declare_parameter("realtime", value=False)
            .get_parameter_value().bool_value
        )
        sim_speed = (
            self.declare_parameter("sim_speed", value=2.0)
            .get_parameter_value().double_value
        )
        spawn_box = (
            self.declare_parameter("spawn_box", value=True)
            .get_parameter_value().bool_value
        )
        # 20 Hz is plenty: perception consumers do per-message Python/xarray
        # work, and 200 Hz across 10 sensor topics saturated their processes.
        self.perception_rate = (
            self.declare_parameter("perception_rate", value=20.0)
            .get_parameter_value().double_value
        )
        # Empty: the scene of the gripper chosen in the experiment yaml (see create_scenario)
        self.model_path = (
            self.declare_parameter("model_path", value="")
            .get_parameter_value().string_value
        )
        # Video of the trials (2FG7 scenario only): folder ("" = off) and every how many trials
        self.video_dir = (
            self.declare_parameter("video_dir", value="")
            .get_parameter_value().string_value
        )
        self.video_every = (
            self.declare_parameter("video_every", value=1)
            .get_parameter_value().integer_value
        )
        self.scenario_options = dict(visualize=visualize, realtime=realtime, speed=sim_speed, spawn_box=spawn_box)

        self.cbgroup_server = MutuallyExclusiveCallbackGroup()
        self.cbgroup_client = MutuallyExclusiveCallbackGroup()
        self.cbgroup_pub = MutuallyExclusiveCallbackGroup()  # perceptions: own group

        self.sim_publishers = {}
        self.perceptions = {}
        self.base_messages = {}
        self.service_world_reset = False

        self.sim = None  # created by load_configuration, once the yaml says which gripper
        self.arm_name = "left_arm"  # actuator name used in the yaml actuation_config

    SCENES = {
        "none": "ur5e_ball_box_scene.xml",          # magnet autograsp (the original experiment)
        "robotiq_2f85": "ur5e_2f85_ball_box_scene.xml",  # physical Robotiq 2F-85, grasp macro
        "onrobot_2fg7": "ur5e_2fg7_ball_box_scene.xml",  # the lab's OnRobot 2FG7, smooth motion, no macro
    }
    # Robot keys of the 2FG7 scenario (see UR5TwoFG7Scenario)
    TWOFG7_KEYS = ("grip_force", "grip_speed", "lift_height", "tool_speed", "tool_accel", "blend_time")

    def create_scenario(self, robot_cfg):
        """Create the MuJoCo scenario. ``robot_cfg`` is the ``Robot`` block of the simulator section of the
        experiment yaml: ``gripper`` (none | robotiq_2f85 | onrobot_2fg7, default none), and optionally
        ``hover_z``; with the Robotiq ``verify_lift`` (m); with the 2FG7 the TWOFG7_KEYS."""
        gripper = str(robot_cfg.get("gripper", "none"))
        if gripper not in self.SCENES:
            raise ValueError(f"Unknown gripper '{gripper}' (one of {list(self.SCENES)})")
        model_path = self.model_path or os.path.join(
            get_package_share_directory("ur5_mujoco"), "models", "universal_robots_ur5e", self.SCENES[gripper])
        options = dict(self.scenario_options, logger=self.get_logger())
        if "hover_z" in robot_cfg:
            options["hover_z"] = float(robot_cfg["hover_z"])
        if gripper == "none":
            self.sim = UR5Scenario(model_path, **options)
        elif gripper == "robotiq_2f85":
            self.sim = UR5GripperScenario(model_path, verify_lift=float(robot_cfg.get("verify_lift", 0.03)), **options)
        else:
            extra = {k: float(robot_cfg[k]) for k in self.TWOFG7_KEYS if k in robot_cfg}
            self.sim = UR5TwoFG7Scenario(model_path, video_dir=self.video_dir, video_every=self.video_every,
                                         **extra, **options)
        self.gripper = gripper
        self.get_logger().info(f"Scenario: {os.path.basename(model_path)} (gripper: {gripper})")

    # ------------------------------------------------------------- perceptions
    def get_perceptions(self):
        state = self.sim.get_state()  # thread-safe snapshot
        ee = state["ee"]
        heading = state["heading"]
        ball = state["ball"]
        box = state["box"]
        grasped = state["grasped"]
        in_box = state["ball_in_box"]

        def set_if(name, fn):
            if name in self.perceptions:
                fn(self.perceptions[name])

        def arm_msg(p):
            p.data[0].x = float(ee[0]); p.data[0].y = float(ee[1])
            p.data[0].angle = float(heading)
        set_if("left_arm", arm_msg)

        set_if("ball_in_left_hand", lambda p: setattr(p, "data", bool(grasped)))
        set_if("grasped_ball", lambda p: setattr(p, "data", 1.0 if grasped else 0.0))
        set_if("ball_in_box", lambda p: setattr(p, "data", 1.0 if in_box else 0.0))
        # End-effector height: makes the vertical gap observable now that z is
        # a free action (the 2D sim only had planar x/y perceptions).
        set_if("arm_height", lambda p: setattr(p, "data", float(ee[2])))

        def ball_msg(p):
            p.data[0].x = float(ball[0]); p.data[0].y = float(ball[1])
        set_if("ball", ball_msg)

        def box_msg(p):
            p.data[0].x = float(box[0]); p.data[0].y = float(box[1])
        set_if("box", box_msg)

        def dist_arm_ball(p):
            # 3D distance: z is a free action dimension and the autograsp
            # condition is 3D, so a planar distance would hide the height gap.
            p.data[0].distance = float(np.linalg.norm(np.array(ee) - np.array(ball)))
            p.data[0].angle = self.sim.wrap_angle(
                heading - self.sim.get_relative_angle(ee, ball))
        set_if("dist_left_arm_ball", dist_arm_ball)

        def dist_ball_box(p):
            p.data[0].distance = self.sim.get_distance(ball, box)
            p.data[0].angle = self.sim.get_relative_angle(ball, box)
        set_if("dist_ball_box", dist_ball_box)

        set_if("ball_angle", lambda p: setattr(p, "data", float(ball[0])))  # (it is the ball's x, old name)
        set_if("ball_x", lambda p: setattr(p, "data", float(ball[0])))
        set_if("box_angle", lambda p: setattr(p, "data", float(box[0])))

        # Gripper perceptions (with the magnet scene they refer to the flange)
        def dist_gripper_ball(p):
            # Planar distance and bearing relative to the heading; the height gap is gripper_ball_dz
            p.data[0].distance = self.sim.get_distance(ee, ball)
            p.data[0].angle = self.sim.wrap_angle(heading - self.sim.get_relative_angle(ee, ball))
        set_if("dist_gripper_ball", dist_gripper_ball)
        set_if("gripper_ball_dz", lambda p: setattr(p, "data", float(ee[2] - ball[2])))
        set_if("gripper_closed", lambda p: setattr(p, "data", 1.0 if state.get("closed", grasped) else 0.0))
        # Robot-side readings, as on the real UR5e + 2FG7: pinch height from the arm's kinematics, and the
        # gripper's aperture (m) and finger force (N) as OnRobot's driver reports them
        set_if("gripper_height", lambda p: setattr(p, "data", float(ee[2])))
        set_if("gripper_width", lambda p: setattr(p, "data", float(state.get("aperture", 0.0))))
        set_if("grip_force", lambda p: setattr(p, "data", float(state.get("grip_force", 0.0))))
        set_if("ball_lifted", lambda p: setattr(p, "data", 1.0 if state.get("lifted", False) else 0.0))

    def reset_world(self):
        self.sim.reset(self.rng)

    # ------------------------------------------------------------- actions
    def denormalize_actuation(self, action: Container, actuation_config):
        """Identical to Sim2DSimple.denormalize_actuation."""
        action_dims = action.feature_labels
        data = action.read()
        for dim in action_dims:
            actuator, param = dim.split(":", 1)
            if actuation_config[actuator][param]["type"] == "float":
                bounds = actuation_config[actuator][param]["bounds"]
                value = data.sel(features=dim).values
                data.loc[{"features": dim}] = bounds[0] + (value * (bounds[1] - bounds[0]))
        return data

    def execute_action(self, action):
        vec = self.denormalize_actuation(action, self.actuation_config)
        features = list(vec.coords["features"].values)

        if f"{self.arm_name}:dist" in features:
            dist = float(vec.sel(features=f"{self.arm_name}:dist").values)
            angle = float(vec.sel(features=f"{self.arm_name}:angle").values)
            dz = float(vec.sel(features=f"{self.arm_name}:z").values) \
                if f"{self.arm_name}:z" in features else None
            if f"{self.arm_name}:grip" in features:
                if self.gripper == "none":
                    self.get_logger().error("grip action without a gripper (set Robot: gripper in the yaml)")
                grip = float(vec.sel(features=f"{self.arm_name}:grip").values)
                self.sim.apply_action_polar(angle, dist, dz, grip)
            else:
                self.sim.apply_action_polar(angle, dist, dz)
        elif f"{self.arm_name}:x" in features:
            x = float(vec.sel(features=f"{self.arm_name}:x").values)
            y = float(vec.sel(features=f"{self.arm_name}:y").values)
            z = float(vec.sel(features=f"{self.arm_name}:z").values) \
                if f"{self.arm_name}:z" in features else self.sim.hover_z
            yaw = float(vec.sel(features=f"{self.arm_name}:yaw").values) \
                if f"{self.arm_name}:yaw" in features else self.sim.heading
            self.sim.apply_action_absolute(x, y, z, yaw)
        else:
            self.get_logger().error(f"Unknown action features: {features}")

    # ------------------------------------------------------------- callbacks (as in Sim2DSimple)
    def world_reset_service_callback(self, request, response):
        self.reset_world()
        response.success = True
        return response

    def new_command_callback(self, data):
        self.get_logger().debug(f"Command received... ITERATION: {data.iteration}")
        if data.command == "reset_world" and not self.service_world_reset:
            self.reset_world()
        elif data.command == "end":
            self.get_logger().info("Ending simulator as requested by LTM...")
            rclpy.shutdown()

    def new_action_service_callback(self, request, response):
        action = Container.from_msg(request.action)
        self.get_logger().info("Executing action " + str(action))
        self.execute_action(action)
        response.success = True
        return response

    def publish_perceptions(self):
        self.get_perceptions()
        for ident, publisher in self.sim_publishers.items():
            publisher.publish(self.perceptions[ident])

    # ------------------------------------------------------------- yaml config (as in Sim2DSimple)
    def load_experiment_file_in_commander(self):
        self.load_client = ServiceClient(LoadConfig, 'commander/load_experiment')
        return self.load_client.send_request(file=self.config_file)

    def load_configuration(self):
        if self.config_file is None or not os.path.isfile(self.config_file):
            self.get_logger().error(f"Config file '{self.config_file}' not found!")
            rclpy.shutdown()
            return
        config = yaml.load(open(self.config_file, "r", encoding="utf-8"),
                           Loader=yamlloader.ordereddict.CLoader)
        sim_cfg = config.get("SimulatedUR5", config.get("SimulatedBaxter"))
        self.create_scenario(sim_cfg.get("Robot", {}) or {})
        self.setup_perceptions(sim_cfg["Perceptions"])
        self.actuation_config = sim_cfg["Actuation"]
        self.arm_name = next(iter(self.actuation_config))  # the (only) actuator of the yaml, e.g. left_arm or arm
        self.setup_control_channel(config["Control"])

        self.random_seed = resolve_seed(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.get_logger().info(f"RNG seed {self.random_seed}")
        self.reset_world()
        if not self.standalone:
            self.load_experiment_file_in_commander()
        else:
            self.get_logger().info("STANDALONE mode: not contacting the commander")

    def setup_control_channel(self, simulation):
        self.ident = simulation["id"]
        message = class_from_classname(simulation["control_msg"])
        self.create_subscription(message, simulation["control_topic"],
                                 self.new_command_callback, 0)
        service_action = simulation.get("executed_action_service")
        service_world_reset = simulation.get("world_reset_service")
        if simulation.get("executed_policy_topic"):
            raise RuntimeError("Topic-triggered policies not supported; use executed_action_service")
        if service_action:
            msg_srv = class_from_classname(simulation["executed_action_msg"])
            self.create_service(msg_srv, service_action,
                                self.new_action_service_callback,
                                callback_group=self.cbgroup_server)
            # Perception timer in its OWN callback group so it keeps firing
            # while an action is being executed in the server group.
            self.perceptions_timer = self.create_timer(
                1.0 / max(self.perception_rate, 1.0), self.publish_perceptions,
                callback_group=self.cbgroup_pub)
        if service_world_reset:
            self.service_world_reset = True
            msg_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(msg_reset, service_world_reset,
                                self.world_reset_service_callback,
                                callback_group=self.cbgroup_server)

    def setup_perceptions(self, perceptions):
        for perception in perceptions:
            sid = perception["name"]
            topic = perception["perception_topic"]
            classname = perception["perception_msg"]
            message = class_from_classname(classname)
            self.perceptions[sid] = message()
            if "List" in classname:
                self.perceptions[sid].data = []
                self.base_messages[sid] = class_from_classname(classname.replace("List", ""))
                self.perceptions[sid].data.append(self.base_messages[sid]())
            elif "Float" in classname:
                self.perceptions[sid].data = 0.0
            else:
                self.perceptions[sid].data = False
            self.get_logger().info("I will publish to... " + str(topic))
            self.sim_publishers[sid] = self.create_publisher(message, topic, 0)


def main(args=None):
    rclpy.init(args=args)
    sim = SimUR5Mujoco()
    sim.load_configuration()
    # Multi-threaded executor: perception timer runs concurrently with the
    # action service, so publishing keeps its rate during arm motions.
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(sim, executor=executor)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Detected: Shutting down simulator...')
    finally:
        if sim.sim is not None and hasattr(sim.sim, "close"):
            sim.sim.close()  # stops the physics thread, finishes the video being recorded
        sim.destroy_node()


if __name__ == '__main__':
    main()