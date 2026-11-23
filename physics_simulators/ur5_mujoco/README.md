# ur5_mujoco: MuJoCo UR5e simulator

A UR5e arm on a table with a 5 cm ball (and a box), simulated in MuJoCo for experiments with low-level actions.
Unlike the discrete event simulators of this repository, it has physics.

This package holds the simulation: the scenario classes of `ur5_mujoco.scenarios` and the robot models. It does not
use ROS or e-MDB. The e-MDB node that runs it, with its launch file and the scripts to run the experiments, is the
*ur5_mujoco_emdb* package of [emdb_experiments_gii](https://github.com/GII/emdb_experiments_gii), whose experiment
`ur5_2fg7_lift_pl_only_experiment.yaml` uses the 2FG7 scene.

There is one scenario per gripper, with its scene in `models/universal_robots_ur5e/`:

| Class | Gripper | Scene | Grasp |
|---|---|---|---|
| `UR5Scenario` | none | `ur5e_ball_box_scene.xml` | Magnet: the ball sticks to the flange within 7 cm |
| `UR5GripperScenario` | Robotiq 2F-85 | `ur5e_2f85_ball_box_scene.xml` | Grasp macro (close, check with a short lift, reopen) |
| `UR5TwoFG7Scenario` | OnRobot 2FG7 | `ur5e_2fg7_ball_box_scene.xml` | No macro, smooth motion: the learner has to close, lift and reopen by itself |

## Setup

MuJoCo and Ruckig (the trajectory generator of the 2FG7 scene) are Python packages that Ubuntu 24.04 does not ship;
with ROS 2 Jazzy, numpy must stay below 2 (the README of *ur5_mujoco_emdb* installs them in a virtual environment):

```bash
pip install "mujoco>=3.3" ruckig
sudo apt install ffmpeg   # optional: videos of the trials (2FG7 scene)
colcon build --symlink-install --packages-select ur5_mujoco
```

## Use

```python
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
from ur5_mujoco.scenarios import UR5TwoFG7Scenario

scene = os.path.join(get_package_share_directory("ur5_mujoco"), "models", "universal_robots_ur5e",
                     "ur5e_2fg7_ball_box_scene.xml")
sim = UR5TwoFG7Scenario(scene, visualize=False)
sim.reset(np.random.default_rng(1))   # arm to its start pose, ball placed at random
sim.apply_action_polar(angle_deg=20.0, dist=0.1, dz=-0.05, grip=0.0)
print(sim.get_state())
sim.close()
```

- `apply_action_polar(angle_deg, dist, dz=None, grip=None)` turns the heading by `angle_deg` and moves the pinch
  point (the flange in the magnet scene) `dist` along it and `dz` up or down (`None`: to `hover_z`), relative to
  where it is when the action starts. With a gripper, `grip > 0.5` closes it and a lower value opens it.
  `apply_action_absolute(x, y, z, yaw_deg)` moves to a pose instead. Both return when the action ends.
- `get_state()` is a thread-safe snapshot: `ee` (the pinch point), `heading`, `ball`, `box`, `grasped`,
  `ball_in_box`, and with the 2FG7 `closed`, `aperture`, `grip_force` and `lifted`.
- Options of the constructors: `visualize` (MuJoCo viewer), `realtime` and `speed` (physics paced to the wall clock,
  at that speed), `spawn_box` (False parks the box off the scene), `hover_z`, and `logger` (an object with `info`,
  `warn` and `error`, such as the logger of a ROS node); with the 2F-85 `verify_lift`; with the 2FG7 `grip_force`,
  `grip_speed`, `lift_height`, `tool_speed`, `tool_accel`, `blend_time`, and `video_dir` and `video_every` (an mp4
  of one trial in `video_every`). `close()` stops the physics thread and finishes the video being recorded.

## The 2FG7 scene

- **Gripper:** OnRobot's official 2FG7 model (`models/onrobot_2fg7`, see `SOURCE.txt`), silicone fingertips mounted
  outwards, directly on the flange. Aperture 33-71 mm (the silicone adds 1 mm per side to the datasheet's 35-73 mm).
  Its servo is limited to `grip_force` (20-140 N), so a finger blocked by the ball presses with that force, like the
  real force-controlled grip; it opens and closes at `grip_speed`.
- **Actions:** turn the heading, move along it and up or down, relative to where the pinch point is when the
  action starts. `grip > 0.5` closes the gripper at the end of the motion, with the arm at rest (as in a pick);
  otherwise it opens. Nothing is automatic: the learner has to close, lift and, after a miss, open again.
- **Goal:** `lifted` is true when both pads touch the ball and its bottom is `lift_height` (5 cm) above the table.
- **Motion:** every 2 ms (the UR5e's 500 Hz) a jerk-limited trajectory generator (Ruckig) moves a setpoint of the
  pinch point towards the current target at URScript movel's defaults (`tool_speed` 0.25 m/s, `tool_accel`
  1.2 m/s^2). IK turns it into joint targets, sent with velocity feedforward (tracking error below 1.5 mm) and
  gravity compensation. An action ends `blend_time` (0.15 s) before its motion does, while the arm still moves,
  like a UR blend radius, so the motion does not stop between actions; only closing the gripper ends at rest.
- **Clock:** by default physics runs only during actions, as fast as it can, and is frozen (velocities included)
  while e-MDB decides, so the motion is continuous in simulated time. With `realtime=True` physics runs all the
  time in its own thread; while e-MDB decides, the arm goes on to its last target and stops there smoothly, as the
  real robot would.
- **Tolerance** (scripted check): the closing fingers centre the ball. A grasp survives the lift, a 90 degree turn
  and a carry from about +-20 mm along the heading, +-36 mm across the fingers, and a pinch height of -4 to +26 mm
  from the ball centre.

## The 2F-85 and magnet scenes

- **2F-85:** UR5e and 2F-85 from MuJoCo Menagerie, unchanged; `ur5e_2f85.xml` attaches the gripper at the tool flange
  (names with the prefix `gripper_`). Motions are straight lines at the UR5e limits with gravity compensation, and
  the arm settles to within 1 mm before the next step. `grip > 0.5` is a grasp attempt after the motion: close until
  the fingers stop; if both pads touch the ball, lift 3 cm and check that it is still held (like Robotiq's
  object-detected status), otherwise open again. `grasped` is true only for a verified grasp.
- **Magnet:** the ball attaches to the flange when it is within 7 cm.

## Models and licenses

The package is Apache-2.0. The robot models keep their own licenses, in the `LICENSE` file of each folder:
`models/universal_robots_ur5e` (MuJoCo Menagerie, BSD-3-Clause, see its `README.md`), `models/robotiq_2f85`
(MuJoCo Menagerie, BSD-2-Clause, see `SOURCE.txt`) and `models/onrobot_2fg7` (OnRobot's onrobot-ros2,
BSD-3-Clause, see `SOURCE.txt`). The scene files (`ur5e_*_scene.xml`) and `ur5e_2fg7.xml`, `ur5e_2f85.xml` were
written for this package.
