# ur5_mujoco: MuJoCo UR5e simulator for e-MDB

A UR5e arm on a table with a 5 cm ball (and a box), simulated in MuJoCo for e-MDB experiments with low-level
actions. Unlike the discrete event simulators of this repository, it has physics. It runs on ROS 2 Jazzy.

The gripper is chosen by the `Robot: gripper` entry of the `SimulatedUR5` section of the experiment yaml:

| `gripper` | Scene | Grasp |
|---|---|---|
| `none` (default) | `ur5e_ball_box_scene.xml` | Magnet: the ball sticks to the flange within 7 cm |
| `robotiq_2f85` | `ur5e_2f85_ball_box_scene.xml` | Robotiq 2F-85 with a grasp macro (close, check with a short lift, reopen) |
| `onrobot_2fg7` | `ur5e_2fg7_ball_box_scene.xml` | OnRobot 2FG7, no macro, smooth motion: the learner has to close, lift and reopen by itself |

The experiment that uses it is `ur5_2fg7_lift_pl_only_experiment.yaml` of
[emdb_experiments_gii](https://github.com/GII/emdb_experiments_gii) (2FG7, grasp and lift the ball 5 cm), started
with its `ur5_launch.py`.

## Setup

The Python packages that Ubuntu 24.04 does not ship can go in a virtual environment that also sees the system
packages (numpy must stay below 2 for ROS 2 Jazzy):

```bash
python3 -m venv --system-site-packages ~/venvs/emdb
echo "numpy<2" > ~/venvs/emdb/constraints.txt
~/venvs/emdb/bin/pip install -c ~/venvs/emdb/constraints.txt torch --index-url https://download.pytorch.org/whl/cpu
~/venvs/emdb/bin/pip install -c ~/venvs/emdb/constraints.txt "stable-baselines3>=2.3" "gymnasium>=0.29" \
    "mujoco>=3.3" ruckig yamlloader
sudo apt install ffmpeg   # optional: videos of the trials
```

`ruckig` is the trajectory generator of the 2FG7 scene; torch, stable-baselines3 and gymnasium are used by the
learned policies of e-MDB (`PolicyLearned`). Build from the colcon workspace that holds the e-MDB repositories:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to core cognitive_nodes cognitive_processes dummy_nodes \
    simulators sim_2d_emdb experiments ur5_mujoco
```

Per-machine settings can go in `<workspace>/emdb_env.sh`, which `scripts/run_ur5.sh` sources (instead of ROS and
the workspace) when it exists. For example:

```bash
source /opt/ros/jazzy/setup.bash
source "$HOME/<workspace>/install/setup.bash"
export PYTHONPATH="$HOME/venvs/emdb/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID=50              # keep e-MDB runs apart from other ROS 2 systems on the network
export OMP_NUM_THREADS=1             # torch on CPU: one thread per process (e-MDB runs many small networks)
```

## Running an experiment

```bash
scripts/run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 1                 # seed 1, viewer open
VISUALIZE=False scripts/run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 2
VIDEO_EVERY=50 scripts/run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 1  # an mp4 of one trial in 50
scripts/run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 1 realtime:=true sim_speed:=1.0   # watch at real speed
```

Each run gets its own folder, `<workspace>/results/<yaml name>/seed<N>_<date-time>/` (`EMDB_RESULTS` changes the
results folder), with `launch.log`, the e-MDB files (`trials_0.txt`, `goodness_0.txt`, ...), the logs of the
learned policies (`logs/policy_*/`: `traces.csv`, `turns.csv`, `progress.csv`, and the saved model
`sac_model.zip` with `sac_model.json`) and the videos (`videos/trial_XXXXX.mp4`). Use seeds 1, 2, 3...: seed 0
means "random from the clock" (the seed used is printed in the log).

Scripts to look at the runs (with no argument they take the newest run):

| Script | What |
|---|---|
| `scripts/progress.py [RUN_DIR]` | Progress of a run: pace, estimated end, success rate, actions per success, policy learning |
| `scripts/plot_run.py [RUN_DIR]` | One run in detail: `<run>/run_summary.png` (successes over time and who reached them, success rate, actions per trial, SAC training, frames of a recorded trial) |
| `scripts/compare_runs.py [RUN_DIR ...]` | Several runs side by side: a text summary per run and one figure |

The simulation runs as fast as it can (about 0.02 s of wall clock per action), so the architecture sets the pace,
and with the viewer each motion flashes by. `realtime:=true sim_speed:=1.0` paces it to the wall clock; videos
(`VIDEO_EVERY=N`) always play at real speed, one frame per 1/30 s of simulated time, whatever the clock. Video
recording works with or without the viewer: the offscreen recorder starts before the viewer window opens (in the
other order its EGL context cannot be made current). If it cannot record, the log shows
`Video recording disabled: ...` and the run goes on without video.

## Simulator node

`sim_ur5` (launch file `launch/ur5_sim_launch.py` to start it alone) reads the `SimulatedUR5` section of the
experiment yaml (`Robot`, `Actuation`, `Perceptions`) and its `Control` section (the action and world reset
services), and publishes the perceptions on `/mdb/ur5/sensor/<name>`. Parameters:

| Parameter | Default | What |
|---|---|---|
| `config_file` | | Experiment yaml |
| `random_seed` | 0 | Seed of the ball placement (0: from the clock) |
| `visualize` | True | MuJoCo viewer |
| `realtime`, `sim_speed` | False, 2.0 | Physics paced to the wall clock, at that speed |
| `spawn_box` | True | False parks the box off the scene |
| `perception_rate` | 20.0 | Hz of the perception topics |
| `video_dir`, `video_every` | "", 1 | 2FG7 scene: record one trial in `video_every` to `video_dir` |
| `model_path` | "" | Scene file (default: the one of the gripper) |
| `standalone` | False | Do not load the experiment in the commander (to test the node alone) |

## The 2FG7 scene

- **Gripper:** OnRobot's official 2FG7 model (`models/onrobot_2fg7`, see `SOURCE.txt`), silicone fingertips mounted
  outwards, directly on the flange. Aperture 33-71 mm (the silicone adds 1 mm per side to the datasheet's 35-73 mm).
  Its servo is limited to `grip_force` (20-140 N), so a finger blocked by the ball presses with that force, like the
  real force-controlled grip; it opens and closes at `grip_speed`.
- **Actions:** `arm: dist, angle, z, grip`: turn the heading, move along it and up or down, relative to where the
  pinch point is when the action starts. `grip > 0.5` closes the gripper at the end of the motion, with the arm at
  rest (as in a pick); otherwise it opens. Nothing is automatic: the learner has to close, lift and, after a miss,
  open again.
- **Goal:** `ball_lifted` is 1 when both pads touch the ball and its bottom is `lift_height` (5 cm) above the table.
- **Perceptions:** the ball relative to the pinch point (`ball_x`, `dist_gripper_ball`, `gripper_ball_dz`), plus
  two readings that the real robot provides: `gripper_height` (pinch point height, from the arm's kinematics) and
  `gripper_width` (aperture, as OnRobot's driver reports it). `grip_force` and `ball` are published for watching.
- **Motion:** every 2 ms (the UR5e's 500 Hz) a jerk-limited trajectory generator (Ruckig) moves a setpoint of the
  pinch point towards the current target at URScript movel's defaults (`tool_speed` 0.25 m/s, `tool_accel`
  1.2 m/s^2). IK turns it into joint targets, sent with velocity feedforward (tracking error below 1.5 mm) and
  gravity compensation. An action ends `blend_time` (0.15 s) before its motion does, while the arm still moves,
  like a UR blend radius, so the motion does not stop between actions; only closing the gripper ends at rest.
- **Clock:** by default physics runs only during actions, as fast as it can, and is frozen (velocities included)
  while e-MDB decides, so the motion is continuous in simulated time. With `realtime:=true` physics runs all the
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
  object-detected status), otherwise open again. `grasped_ball` is 1 only for a verified grasp.
- **Magnet:** the ball attaches to the flange when it is within 7 cm.

## Models and licenses

The package is Apache-2.0. The robot models keep their own licenses, in the `LICENSE` file of each folder:
`models/universal_robots_ur5e` (MuJoCo Menagerie, BSD-3-Clause, see its `README.md`), `models/robotiq_2f85`
(MuJoCo Menagerie, BSD-2-Clause, see `SOURCE.txt`) and `models/onrobot_2fg7` (OnRobot's onrobot-ros2,
BSD-3-Clause, see `SOURCE.txt`). The scene files (`ur5e_*_scene.xml`) and `ur5e_2fg7.xml`, `ur5e_2f85.xml` were
written for this package.
