#!/usr/bin/env python3
"""Progress of a UR5 e-MDB run: pace, ETA, success rate, steps per success, policy learning.

Usage:  progress.py [run folder]     (default: the newest run folder under the results folder,
        $EMDB_RESULTS or <workspace>/results; see compare_runs.find_results)
"""
import csv
import glob
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_runs import find_results  # noqa: E402

ITER = re.compile(rb"\[(\d{10}\.\d+)\] \[main_loop\]: \*\*\* ITERATION: (\d+)/(\d+)")


def newest_run():
    runs = glob.glob(os.path.join(find_results(), "*", "seed*_*"))
    return max(runs, key=os.path.getmtime) if runs else None


def iteration_times(log_path, tail_bytes=2_000_000):
    """(first, last) (time, iteration, total) of the main loop. Reads the start and the end of the log only."""
    with open(log_path, "rb") as f:
        head = f.read(tail_bytes)
        f.seek(max(0, os.path.getsize(log_path) - tail_bytes))
        tail = f.read()
    first, last = ITER.search(head), list(ITER.finditer(tail))
    if not first or not last:
        return None, None
    conv = lambda m: (float(m.group(1)), int(m.group(2)), int(m.group(3)))
    return conv(first), conv(last[-1])


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    if not run:
        sys.exit("No run folder found")
    print(f"Run: {run}")
    first, last = iteration_times(os.path.join(run, "launch.log"))
    if last:
        elapsed = last[0] - first[0]
        pace = elapsed / max(1, last[1] - first[1])
        print(f"Iteration {last[1]}/{last[2]}  elapsed {elapsed/3600:.1f} h  {pace:.1f} s/iteration  "
              f"ETA {pace*(last[2]-last[1])/3600:.1f} h  (last log line {time.time()-last[0]:.0f} s ago)")

    rows = []
    path = os.path.join(run, "trials_0.txt")
    if os.path.exists(path):
        for line in open(path):
            c = line.strip().split("\t")
            if len(c) in (4, 5) and c[0].isdigit():
                rows.append((int(c[0]), int(c[-2]), c[-1] == "True"))
    if rows:
        ok = [r for r in rows if r[2]]
        print(f"Trials {len(rows)}, successful {len(ok)}" + (f", first at iteration {ok[0][0]}" if ok else ""))
        for n in (20, 50):
            if len(rows) >= n:
                w = rows[-n:]
                w_ok = [r[1] for r in w if r[2]]
                med = f", median {statistics.median(w_ok):.0f} steps per success" if w_ok else ""
                print(f"  last {n} trials: {100*len(w_ok)/n:.0f} % success{med}")

    for traces in glob.glob(os.path.join(run, "logs", "policy_*", "traces.csv")):
        with open(traces) as f:
            src = [r["source_policy"] or "novelty_exploration" for r in csv.DictReader(f)]  # exploration episodes have no parent policy
        counts = {s: src.count(s) for s in sorted(set(src))}
        prog = os.path.join(os.path.dirname(traces), "progress.csv")
        trainings = max(0, sum(1 for _ in open(prog)) - 1) if os.path.exists(prog) else 0
        print(f"{os.path.basename(os.path.dirname(traces))}: {len(src)} traces {counts}, {trainings} SAC trainings")


if __name__ == "__main__":
    main()
