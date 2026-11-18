#!/usr/bin/env python3
"""Compare finished (or running) UR5 e-MDB runs: a text summary per run and one figure.

Usage:
    compare_runs.py                       # newest seed folder of each 2FG7 lift experiment
    compare_runs.py RUN_DIR [RUN_DIR ...] [--out FILE.png] [--block 300]

Reads, per run folder: trials_0.txt (one row per trial), goodness_0.txt (which process ran each iteration),
logs/policy_*/traces.csv and progress.csv (the learned policy), videos/.
The figure goes to <results>/ur5_2fg7_comparison.png unless --out says otherwise. The results folder is
$EMDB_RESULTS, else the results folder of the workspace (see find_results).
"""
import argparse
import csv
import glob
import os
import statistics
import subprocess


def find_results():
    """Results folder of the runs: $EMDB_RESULTS, else <workspace>/results. The workspace is $EMDB_WS, else the first
    folder above this script that has install/setup.bash (the colcon workspace: with --symlink-install the scripts
    run from the source), else the current folder."""
    if os.environ.get("EMDB_RESULTS"):
        return os.path.expanduser(os.environ["EMDB_RESULTS"])
    ws = os.environ.get("EMDB_WS")
    if not ws:
        ws = os.getcwd()
        d = os.path.dirname(os.path.realpath(__file__))
        while d != os.path.dirname(d):
            if os.path.exists(os.path.join(d, "install", "setup.bash")):
                ws = d
                break
            d = os.path.dirname(d)
    return os.path.join(os.path.expanduser(ws), "results")


RESULTS = find_results()
DEFAULT_EXPERIMENTS = ("ur5_2fg7_lift_experiment", "ur5_2fg7_lift_td_experiment", "ur5_2fg7_lift_pl_only_experiment")
NAMES = {  # experiment folder -> name in the figure
    "ur5_2fg7_lift_experiment": "Main run (td_learning: False)",
    "ur5_2fg7_lift_td_experiment": "RL-mode run (td_learning: True)",
    "ur5_2fg7_lift_pl_only_experiment": "Policy only (no utility model)",
}
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")  # categorical slots in fixed order
# The colour follows the experiment (the order of the runs on the command line does not repaint them)
EXPERIMENT_COLORS = {exp: RUN_COLORS[i] for i, exp in enumerate(DEFAULT_EXPERIMENTS)}
PROCESSES = (  # (name, colour) of what ran each iteration
    ("novelty exploration", "#cfcec9"),
    ("utility model (deliberation)", "#1baf7a"),
    ("learned policy", "#6250d6"),
)
TRIAL_ACTIONS = 251        # a failed trial: 5 iterations of ~50 actions, then the world is reset
VIDEO_TRIAL_OFFSET = 2     # the sim resets twice before trial 0 (constructor, yaml load): video N = trial N - 2


def process_of(policy):
    if policy == "novelty_exploration":
        return "novelty exploration"
    if policy.startswith("UtilityModel"):
        return "utility model (deliberation)"
    if policy.startswith("policy_"):
        return "learned policy"
    return None  # model_creation_policy and the like (start-up only)


def load_run(path):
    exp = os.path.basename(os.path.dirname(path.rstrip("/")))
    run = {"path": path, "exp": exp, "name": NAMES.get(exp, f"{exp} / {os.path.basename(path)}")}

    run["trials"] = []  # (iteration, trial, actions, success)
    with open(os.path.join(path, "trials_0.txt")) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) == 5 and c[0].isdigit():
                run["trials"].append((int(c[0]), int(c[1]), int(c[3]), c[4] == "True"))

    run["process"] = {}  # iteration -> process that ran it
    with open(os.path.join(path, "goodness_0.txt")) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[0].isdigit():
                p = process_of(c[3])
                if p:
                    run["process"][int(c[0])] = p
    run["iterations"] = max(run["process"], default=0)

    run["policies"] = []
    for d in sorted(glob.glob(os.path.join(path, "logs", "policy_*"))):
        info = {"name": os.path.basename(d), "traces": 0, "by_policy": 0, "updates": 0}
        tpath, ppath = os.path.join(d, "traces.csv"), os.path.join(d, "progress.csv")
        if os.path.exists(tpath):
            with open(tpath) as f:
                rows = list(csv.DictReader(f))
            info["traces"] = len(rows)
            info["by_policy"] = sum(1 for r in rows if (r.get("source_policy") or "").startswith("policy_"))
        if os.path.exists(ppath):
            with open(ppath) as f:
                rows = list(csv.DictReader(f))
            if rows:
                info["updates"] = int(float(rows[-1]["train/n_updates"]))
        turns_path = os.path.join(d, "turns.csv")  # written since 11 Nov 2026: one row per own turn
        if os.path.exists(turns_path):
            with open(turns_path) as f:
                rows = list(csv.DictReader(f))
            info["turns"] = len(rows)
            info["trial_turns"] = sum(1 for r in rows if r["trial"] == "1")
            info["turn_successes"] = sum(1 for r in rows if r["success"] == "1")
            info["trial_successes"] = sum(1 for r in rows if r["success"] == "1" and r["trial"] == "1")
            info["last_scale"] = float(rows[-1]["scale"]) if rows else None
            # scale_activation False (policy only): no trials and always 1.0
            info["scaled"] = any(r["trial"] == "1" or float(r["scale"]) != 1.0 for r in rows)
        run["policies"].append(info)

    run["videos"] = sorted(glob.glob(os.path.join(path, "videos", "trial_*.mp4")))
    return run


def blocks(run, size):
    """Per block of iterations: trials ending in it, successes, median actions per success, process counts."""
    out = []
    for start in range(0, run["iterations"], size):
        end = start + size
        t = [r for r in run["trials"] if start < r[0] <= end]
        ok = [r[2] for r in t if r[3]]
        procs = [p for i, p in run["process"].items() if start < i <= end]
        out.append({
            "start": start, "end": min(end, run["iterations"]), "trials": len(t), "successes": len(ok),
            "rate": 100.0 * len(ok) / len(t) if t else float("nan"),
            "median_actions": statistics.median(ok) if ok else float("nan"),
            "share": {name: 100.0 * procs.count(name) / len(procs) if procs else 0.0 for name, _ in PROCESSES},
            "counts": {name: procs.count(name) for name, _ in PROCESSES},
        })
    return out


def video_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return float("nan")


def summary(run, size):
    trials = run["trials"]
    ok = [r for r in trials if r[3]]
    print(f"\n=== {run['name']}\n    {run['path']}")
    print(f"Iterations {run['iterations']}, trials {len(trials)}, successful {len(ok)} "
          f"({100.0 * len(ok) / max(1, len(trials)):.0f} %)" + (f", first at iteration {ok[0][0]}" if ok else ""))

    # Who ran the iterations, and who was running when a trial succeeded
    counts = {name: list(run["process"].values()).count(name) for name, _ in PROCESSES}
    by = {name: sum(1 for r in ok if run["process"].get(r[0]) == name) for name, _ in PROCESSES}
    print("Iterations run by:  " + ",  ".join(f"{n} {c}" for n, c in counts.items()))
    print("Successes reached by:  " + ",  ".join(f"{n} {c}" for n, c in by.items()))
    for p in run["policies"]:
        print(f"{p['name']}: {p['traces']} successful traces in its buffer ({p['by_policy']} made by the policy "
              f"itself), {p['updates']} SAC updates")
        if p.get("turns") and not p["scaled"]:
            print(f"  its turns: {p['turns']}, successful {p['turn_successes']} "
                  f"({100.0 * p['turn_successes'] / p['turns']:.0f} %); activation not scaled (it acts whenever its "
                  "C-node allows)")
        elif p.get("turns"):
            print(f"  its turns: {p['turns']}, successful {p['turn_successes']}; trial turns (right after a training) "
                  f"{p['trial_turns']}, successful {p['trial_successes']}; activation factor at the end "
                  f"{p['last_scale']:.2f} (1 = as good as the utility model, 0.1 = no success)")

    print(f"\n{'iterations':>11} {'trials':>6} {'success':>8} {'median actions':>15}   "
          f"{'novelty':>7} {'util.model':>10} {'policy':>6}  (% of iterations)")
    for b in blocks(run, size):
        med = f"{b['median_actions']:.0f}" if b["successes"] else "-"
        s = b["share"]
        print(f"{b['start'] + 1:>5}-{b['end']:<5} {b['trials']:>6} {b['rate']:>7.0f}% {med:>15}   "
              f"{s['novelty exploration']:>7.0f} {s['utility model (deliberation)']:>10.0f} "
              f"{s['learned policy']:>6.1f}")

    if run["videos"]:
        by_trial = {r[1]: r for r in trials}
        print(f"\nVideos: {len(run['videos'])} in {os.path.dirname(run['videos'][0])}")
        for v in run["videos"]:
            n = int(os.path.basename(v)[6:11]) - VIDEO_TRIAL_OFFSET
            r = by_trial.get(n)
            what = "unfinished" if r is None else (f"SUCCESS in {r[2]} actions" if r[3] else "no success")
            print(f"  {os.path.basename(v)}  {video_duration(v):5.0f} s   trial {n}: {what}")


def figure(runs, size, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": ink, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    })
    n_proc_rows = len(runs)
    fig = plt.figure(figsize=(13, 9 + 2.6 * n_proc_rows), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2 + n_proc_rows, 2, height_ratios=[1, 1] + [0.8] * n_proc_rows, hspace=0.62, wspace=0.18,
                          top=0.95)
    x_max = max(r["iterations"] for r in runs)

    # (a) cumulative successes
    ax = fig.add_subplot(gs[0, 0])
    for run in runs:
        col = run["color"]
        xs, ys, n = [0], [0], 0
        for it, _, _, ok in run["trials"]:
            if ok:
                n += 1
                xs.append(it)
                ys.append(n)
        xs.append(run["iterations"])
        ys.append(n)
        ax.step(xs, ys, where="post", color=col, lw=2, label=run["name"])
        ax.annotate(f"{n}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=ink)
    ax.set_title("(a) Successful trials, cumulative")
    ax.set_xlabel("iteration")
    ax.set_ylabel("successful trials")
    ax.set_xlim(0, x_max * 1.05)
    ax.legend(loc="upper left")

    # (b) success rate per block
    ax = fig.add_subplot(gs[0, 1])
    for run in runs:
        col = run["color"]
        b = blocks(run, size)
        xs = [(x["start"] + x["end"]) / 2 for x in b]
        ax.plot(xs, [x["rate"] for x in b], color=col, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5,
                label=run["name"])
    ax.set_title(f"(b) Success rate, per block of {size} iterations")
    ax.set_xlabel("iteration")
    ax.set_ylabel("% of the block's trials that succeeded")
    ax.set_xlim(0, x_max * 1.05)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    # (c) actions per successful trial: dots, and the median per block
    ax = fig.add_subplot(gs[1, :])
    for run in runs:
        col = run["color"]
        ok = [(it, a) for it, _, a, s in run["trials"] if s]
        ax.scatter([o[0] for o in ok], [o[1] for o in ok], s=16, color=col, alpha=0.4, lw=0)
        b = [x for x in blocks(run, size) if x["successes"] >= 3]
        ax.plot([(x["start"] + x["end"]) / 2 for x in b], [x["median_actions"] for x in b], color=col, lw=2,
                marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=f"{run['name']}: median per block")
    ax.axhline(TRIAL_ACTIONS, color=ink2, lw=1, ls="--")
    ax.text(x_max * 0.01, TRIAL_ACTIONS + 5, f"trial limit: {TRIAL_ACTIONS} actions, then the world is reset",
            fontsize=8.5, color=ink2, va="bottom")
    ax.set_title(f"(c) Actions needed in each successful trial (dots); median per block of {size} iterations "
                 "(blocks with 3 or more successes)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("actions in the trial")
    ax.set_xlim(0, x_max * 1.05)
    ax.set_ylim(0, TRIAL_ACTIONS + 30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    # (d...) which process ran the iterations
    for row, run in enumerate(runs):
        ax = fig.add_subplot(gs[2 + row, :])
        b = blocks(run, size)
        xs = [(x["start"] + x["end"]) / 2 for x in b]
        widths = [(x["end"] - x["start"]) * 0.86 for x in b]  # the last block can be shorter
        bottom = [0.0] * len(b)
        for name, col in PROCESSES:
            vals = [x["share"][name] for x in b]
            if not any(vals):
                continue  # never ran in this run (e.g. no utility model): no bar and no legend entry
            ax.bar(xs, vals, width=widths, bottom=bottom, color=col, edgecolor="#fcfcfb", lw=1.5, label=name)
            bottom = [p + v for p, v in zip(bottom, vals)]
        for x, blk in zip(xs, b):
            n = blk["counts"]["learned policy"]
            if n:
                ax.text(x, 101, f"policy\n×{n}", ha="center", va="bottom", fontsize=8, color=ink)
        total = sum(1 for p in run["process"].values() if p == "learned policy")
        ax.set_title(f"({chr(ord('d') + row)}) {run['name']}: what ran each iteration "
                     f"(learned policy: {total} of {run['iterations']})")
        ax.set_xlabel("iteration")
        ax.set_ylabel("% of iterations")
        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)

    fig.suptitle("UR5e + OnRobot 2FG7, grasp and lift the ball", x=0.06, y=0.985, ha="left", fontsize=13, fontweight="bold",
                 color=ink)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (default: newest seed folder of each 2FG7 lift experiment)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "ur5_2fg7_comparison.png"))
    ap.add_argument("--block", type=int, default=300, help="iterations per block in the table and in (b), (d)")
    args = ap.parse_args()

    paths = args.runs
    if not paths:
        for exp in DEFAULT_EXPERIMENTS:
            found = sorted(glob.glob(os.path.join(RESULTS, exp, "seed*_*")), key=os.path.getmtime)
            if found:
                paths.append(found[-1])
    if not paths:
        raise SystemExit(f"No run folders found under {RESULTS}")
    runs = [load_run(p) for p in paths]
    spare = [c for c in RUN_COLORS if c not in {EXPERIMENT_COLORS.get(r["exp"]) for r in runs}]
    for run in runs:  # known experiments keep their colour; others (or a second seed) take the free slots in order
        col = EXPERIMENT_COLORS.get(run["exp"])
        if col is None or col in {r.get("color") for r in runs}:
            col = spare.pop(0) if spare else RUN_COLORS[-1]
        run["color"] = col
    for run in runs:
        summary(run, args.block)
    figure(runs, args.block, args.out)


if __name__ == "__main__":
    main()
