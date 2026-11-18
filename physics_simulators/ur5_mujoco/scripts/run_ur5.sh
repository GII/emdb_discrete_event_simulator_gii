#!/usr/bin/env bash
# Run a UR5 e-MDB experiment (ROS 2 Jazzy) in its own results folder, with the MuJoCo viewer.
#
# Usage:  run_ur5.sh <experiment yaml> [seed] [extra launch args...]
#   run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml                 # seed 1, viewer open
#   VISUALIZE=False run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 2         # no viewer
#   VIDEO_EVERY=50 run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 1          # mp4 of 1 trial in 50, in <run>/videos
#   run_ur5.sh ur5_2fg7_lift_pl_only_experiment.yaml 1 realtime:=true sim_speed:=1.0   # watch at real speed
#
# Seed: use 1, 2, 3... (0 means "a random seed from the clock"; the one used is printed in the log).
# Output: <results>/<yaml name>/seed<N>_<date-time>/ (launch.log, trials_0.txt, goodness_0.txt, logs/policy_*/...,
#         videos/). Ctrl+C stops the run.
# Env:    VISUALIZE (True), PERCEPTION_RATE (40 Hz), VIDEO_EVERY (off).
#         EMDB_WS: the colcon workspace (default: the first folder above this script with install/setup.bash).
#         EMDB_RESULTS: the results folder (default: <workspace>/results).
#         <workspace>/emdb_env.sh, if it exists, is sourced instead of ROS + the workspace: per-machine settings
#         (e.g. a Python venv on PYTHONPATH, ROS_DOMAIN_ID). See the README.
set -eo pipefail
if [[ $# -lt 1 ]]; then sed -n '2,19p' "$0"; exit 1; fi

ws="${EMDB_WS:-}"
if [[ -z "$ws" ]]; then
    d="$(dirname "$(readlink -f "$0")")"
    while [[ "$d" != "/" ]]; do
        if [[ -f "$d/install/setup.bash" ]]; then ws="$d"; break; fi
        d="$(dirname "$d")"
    done
fi
if [[ -z "$ws" ]]; then echo "colcon workspace not found: set EMDB_WS" >&2; exit 1; fi
if [[ -f "$ws/emdb_env.sh" ]]; then
    source "$ws/emdb_env.sh"
else
    source /opt/ros/jazzy/setup.bash
    source "$ws/install/setup.bash"
fi

yaml="$1"; seed="${2:-1}"; shift; [[ $# -gt 0 ]] && shift
run_dir="${EMDB_RESULTS:-$ws/results}/${yaml%.yaml}/seed${seed}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir" && cd "$run_dir"
echo ">> $run_dir  (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0})"
video=()
if [[ -n "${VIDEO_EVERY:-}" ]]; then
    video=(video_dir:="$run_dir/videos" video_every:="$VIDEO_EVERY")
    # Offscreen rendering without a window; the viewer (VISUALIZE=True) uses its own GLFW context
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
ros2 launch experiments ur5_launch.py experiment_file:="$yaml" random_seed:="$seed" \
    visualize:="${VISUALIZE:-True}" perception_rate:="${PERCEPTION_RATE:-40.0}" "${video[@]}" "$@" 2>&1 | tee launch.log
