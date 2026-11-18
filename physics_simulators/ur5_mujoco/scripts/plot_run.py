#!/usr/bin/env python3
"""One UR5 e-MDB run in detail: a figure of how it learned (compare_runs.py compares several runs).

Usage:
    plot_run.py                     # the newest run folder (results folder: see compare_runs.find_results)
    plot_run.py RUN_DIR [--out FILE.png] [--block 25] [--video trial_02200.mp4]

Panels: (a) successful trials over the wall-clock time of the run, by who reached them (novelty exploration,
utility model, learned policy); (b) success rate per block of trials; (c) actions in each trial; (d), (e) the SAC
training of the learned policy (critic loss, entropy coefficient); (f) frames of a recorded successful trial.
Reads trials_0.txt, goodness_0.txt, launch.log (times of the iterations), logs/policy_*/progress.csv and
turns.csv, videos/. The figure goes to RUN_DIR/run_summary.png unless --out says otherwise.
"""
import argparse
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_runs import NAMES, RESULTS, TRIAL_ACTIONS, VIDEO_TRIAL_OFFSET, load_run  # noqa: E402

# Who ran an iteration: the colours of compare_runs.py, with a darker grey for small marks (novelty exploration's
# light bar grey is too faint as a dot). Checked 13 Nov 2026: grey/violet OKLab distance 23 (normal vision),
# >= 14.5 under protan/deutan/tritan simulation, contrast >= 3.3:1 on the surface.
WHO = {
    "novelty exploration": "#8c8b85",
    "utility model (deliberation)": "#1baf7a",
    "learned policy": "#6250d6",
}
INK, INK2, GRID, SURFACE, FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c2"
FRAME_CROP = (420, 0, 720, 470)  # x, y, w, h of the 1280x720 video frames: arm, ball and lift (fixed camera)


def iteration_times(path):
    """Wall-clock time of the start of each iteration (the "*** ITERATION: n/N ***" lines of launch.log)."""
    log = os.path.join(path, "launch.log")
    if not os.path.exists(log):
        return {}
    out = subprocess.run(["grep", "-a", "-o", r"\[[0-9.]*\] \[main_loop\]: \*\*\* ITERATION: [0-9]*", log],
                         capture_output=True, text=True).stdout
    times = {}
    for m in re.finditer(r"\[([0-9.]+)\] \[main_loop\]: \*\*\* ITERATION: ([0-9]+)", out):
        times.setdefault(int(m.group(2)), float(m.group(1)))
    return times


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load(path):
    run = load_run(path)
    times = iteration_times(path)
    t0 = times.get(1)
    # A trial ends with the iteration in its row: its end is the start of the next iteration (or that one's start)
    run["trial_rows"] = []
    for it, trial, actions, ok in run["trials"]:
        t = times.get(it + 1, times.get(it))
        run["trial_rows"].append({
            "iteration": it, "trial": trial, "actions": actions, "success": ok,
            "who": run["process"].get(it), "hours": (t - t0) / 3600.0 if t is not None and t0 else None,
        })
    run["hours_total"] = (max(times.values()) - t0) / 3600.0 if t0 else None
    pol = sorted(glob.glob(os.path.join(path, "logs", "policy_*")))
    run["policy_dir"] = pol[0] if pol else None
    run["progress"] = read_csv(os.path.join(pol[0], "progress.csv")) if pol else []
    run["turns"] = read_csv(os.path.join(pol[0], "turns.csv")) if pol else []
    first_policy_it = min((i for i, p in run["process"].items() if p == "learned policy"), default=None)
    run["policy_start"] = None
    if first_policy_it is not None:
        row = next((r for r in run["trial_rows"] if r["iteration"] >= first_policy_it), None)
        first_training = int(run["turns"][0]["training"]) if run["turns"] else None
        updates = None
        if first_training and len(run["progress"]) >= first_training:
            updates = float(run["progress"][first_training - 1]["train/n_updates"])
        t = times.get(first_policy_it)
        run["policy_start"] = {
            "iteration": first_policy_it, "trial": row["trial"] if row else None, "updates": updates,
            "hours": (t - t0) / 3600.0 if t is not None and t0 else None,
            "successes_before": sum(1 for r in run["trial_rows"] if r["success"] and r["iteration"] < first_policy_it),
        }
    return run


def pick_video(run, name=None):
    """The video to show: --video, else the newest one whose trial succeeded."""
    by_trial = {r["trial"]: r for r in run["trial_rows"]}
    candidates = run["videos"] if not name else [v for v in run["videos"] if os.path.basename(v) == name]
    for v in reversed(candidates):
        r = by_trial.get(int(os.path.basename(v)[6:11]) - VIDEO_TRIAL_OFFSET)
        if r and (r["success"] or name):
            return v, r
    return None, None


def video_frames(video, n=5):
    """n frames evenly spread over the video, cropped to the workspace (numpy arrays), and their times."""
    import matplotlib.image as mpimg
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                               capture_output=True, text=True).stdout.strip() or 0)
    if dur <= 0:
        return []
    x, y, w, h = FRAME_CROP
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = min(dur - 0.05, dur * i / (n - 1))
            png = os.path.join(tmp, f"f{i}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                            "-vf", f"crop={w}:{h}:{x}:{y}", png], check=False)
            if os.path.exists(png):
                frames.append((t, mpimg.imread(png)))
    return frames


def summary(run):
    rows = run["trial_rows"]
    ok = [r for r in rows if r["success"]]
    print(f"{run['name']}\n  {run['path']}")
    hours = f", {run['hours_total']:.1f} h" if run["hours_total"] else ""
    print(f"  {run['iterations']} iterations, {len(rows)} trials, {len(ok)} successful "
          f"({100.0 * len(ok) / max(1, len(rows)):.0f} %){hours}")
    for who in WHO:
        n = sum(1 for r in ok if r["who"] == who)
        if n:
            acts = statistics.median(r["actions"] for r in ok if r["who"] == who)
            print(f"  reached by {who}: {n} (median {acts:.0f} actions per success)")
    s = run["policy_start"]
    if s:
        when = f" after {s['hours']:.2f} h" if s["hours"] is not None else ""
        print(f"  learned policy first acted at iteration {s['iteration']} (trial {s['trial']}){when}, "
              f"with {s['successes_before']} successes before it")
    if run["turns"]:
        good = sum(1 for t in run["turns"] if t["success"] == "1")
        print(f"  learned policy turns: {len(run['turns'])}, successful {good}")


def figure(run, block, out, video_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    })
    rows = run["trial_rows"]
    start = run["policy_start"]
    video, vrow = pick_video(run, video_name)
    frames = video_frames(video) if video else []

    fig = plt.figure(figsize=(13, 15.5 if frames else 12.5), facecolor=SURFACE)
    gs = fig.add_gridspec(4 if frames else 3, 2, height_ratios=[1.1, 1, 0.9] + ([0.55] if frames else []),
                          hspace=0.5, wspace=0.2, top=0.93)

    def mark_start(ax, x, text=None, y=0.97, side="left"):
        """Dashed line where the learned policy first acted, with the text on one side of it."""
        if x is None:
            return
        ax.axvline(x, color=INK2, lw=1, ls="--")
        if text:
            ax.text(x, y, text + " " if side == "left" else " " + text, transform=ax.get_xaxis_transform(),
                    fontsize=8.5, color=INK2, va="top", ha="right" if side == "left" else "left")

    # (a) successful trials over time, by who reached them (log scale: 20 random successes next to 2000)
    ax = fig.add_subplot(gs[0, :])
    timed = [r for r in rows if r["hours"] is not None]
    if timed:
        handles = []
        for who, col in WHO.items():
            xs, ys, n = [], [], 0
            for r in timed:
                if r["success"] and r["who"] == who:
                    n += 1
                    xs.append(r["hours"])
                    ys.append(n)
            if not n:
                continue
            xs.append(run["hours_total"])
            ys.append(n)
            ax.step(xs, ys, where="post", color=col, lw=2)
            ax.annotate(f"{n}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=10)
            handles.append(Line2D([], [], color=col, lw=2, label=f"reached by {who}"))
        if start and start["hours"] is not None:
            h, m = divmod(round(start["hours"] * 60), 60)
            mark_start(ax, start["hours"], f"the learned policy starts acting after {h} h {m:02d} min\n"
                                           f"({start['successes_before']} successes so far)", y=0.97)
        ax.set_yscale("log")
        ax.set_ylim(0.8, None)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(0, run["hours_total"] * 1.04)
        ax.set_xlabel("hours since the start of the run")
        ax.legend(handles=handles, loc="upper left")
    ax.set_title("(a) Successful trials over the run (cumulative), by who was acting when the ball was lifted")
    ax.set_ylabel("successful trials (log scale)")

    # (b) success rate per block of trials
    ax = fig.add_subplot(gs[1, 0])
    xs, ys = [], []
    for i in range(0, len(rows), block):
        chunk = rows[i:i + block]
        xs.append(chunk[0]["trial"] + len(chunk) / 2)
        ys.append(100.0 * sum(r["success"] for r in chunk) / len(chunk))
    ax.plot(xs, ys, color=INK, lw=2)
    mark_start(ax, start["trial"] if start else None, "learned policy starts acting", y=0.55, side="right")
    ax.set_title(f"(b) Success rate, per block of {block} trials")
    ax.set_xlabel("trial")
    ax.set_ylabel("% of the trials that succeeded")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, len(rows))

    # (c) actions in each trial
    ax = fig.add_subplot(gs[1, 1])
    failed = [r for r in rows if not r["success"]]
    if failed:
        ax.scatter([r["trial"] for r in failed], [r["actions"] for r in failed], s=14, marker="|", color=FAINT,
                   lw=1.2, label=f"failed: world reset after {TRIAL_ACTIONS} actions")
    for who, col in WHO.items():
        pts = [r for r in rows if r["success"] and r["who"] == who]
        if pts:
            ax.scatter([r["trial"] for r in pts], [r["actions"] for r in pts], s=16, color=col, lw=0, alpha=0.75,
                       label=f"success, reached by {who}")
    mark_start(ax, start["trial"] if start else None)
    ax.set_yscale("log")
    ax.set_ylim(2, TRIAL_ACTIONS * 1.8)
    ax.set_xlim(0, len(rows))
    ax.set_title("(c) Actions in each trial")
    ax.set_xlabel("trial")
    ax.set_ylabel("actions (log scale)")
    ax.legend(loc="upper right", fontsize=8.5, markerscale=1.3)

    # (d), (e) SAC training of the learned policy
    prog = [p for p in run["progress"] if p.get("train/n_updates")]
    for col_i, (key, title, label) in enumerate([
            ("train/critic_loss", "(d) Learned policy: critic loss of SAC", "critic loss (log scale)"),
            ("train/ent_coef", "(e) Learned policy: entropy coefficient of SAC", "entropy coefficient (log scale)")]):
        ax = fig.add_subplot(gs[2, col_i])
        pts = [(float(p["train/n_updates"]), float(p[key])) for p in prog if p.get(key) not in (None, "")]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=WHO["learned policy"], lw=1.5)
            ax.set_yscale("log")
            mark_start(ax, start["updates"] if start else None, "policy starts acting" if col_i == 0 else None)
            ax.set_xlim(0, pts[-1][0] * 1.02)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        else:
            ax.text(0.5, 0.5, "no learned policy in this run", transform=ax.transAxes, ha="center", color=INK2)
        ax.set_title(title)
        ax.set_xlabel("SAC updates (one per action, trained in the background)")
        ax.set_ylabel(label)

    # (f) frames of a recorded successful trial
    if frames:
        sub = gs[3, :].subgridspec(1, len(frames), wspace=0.04)
        for i, (t, img) in enumerate(frames):
            ax = fig.add_subplot(sub[0, i])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(f"t = {t:.1f} s", color=INK2)
            if i == 0:
                who = vrow["who"] or "?"
                ax.set_title(f"(f) Trial {vrow['trial']} ({os.path.basename(video)}): "
                             + (f"success in {vrow['actions']} actions, reached by {who}" if vrow["success"]
                                else "no success")
                             + "; simulated time, real speed", loc="left")

    ok = [r for r in rows if r["success"]]
    turns = run["turns"]
    head = (f"{len(rows)} trials, {len(ok)} successful ({100.0 * len(ok) / max(1, len(rows)):.0f} %)"
            + (f" in {run['hours_total']:.1f} h" if run["hours_total"] else ""))
    if turns:
        good = sum(1 for t in turns if t["success"] == "1")
        acts = [r["actions"] for r in ok if r["who"] == "learned policy"]
        head += (f"  ·  learned policy: {good} of {len(turns)} turns successful"
                 + (f", median {statistics.median(acts):.0f} actions per success" if acts else ""))
    fig.suptitle(f"UR5e + OnRobot 2FG7, grasp and lift the ball: {run['name']}", x=0.06, y=0.985, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.06, 0.962, f"{os.path.basename(run['path'])}  ·  {head}", ha="left", fontsize=10, color=INK2)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run folder (default: the newest one under the results folder)")
    ap.add_argument("--out", help="figure file (default: RUN_DIR/run_summary.png)")
    ap.add_argument("--block", type=int, default=25, help="trials per block in (b)")
    ap.add_argument("--video", help="video file name to show in (f) (default: the newest successful one)")
    args = ap.parse_args()
    path = args.run
    if not path:
        found = sorted(glob.glob(os.path.join(RESULTS, "*", "seed*_*")), key=os.path.getmtime)
        if not found:
            raise SystemExit(f"No run folders found under {RESULTS}")
        path = found[-1]
    path = os.path.abspath(path.rstrip("/"))
    run = load(path)
    run["name"] = NAMES.get(run["exp"], run["exp"])
    summary(run)
    figure(run, args.block, args.out or os.path.join(path, "run_summary.png"), args.video)


if __name__ == "__main__":
    main()
