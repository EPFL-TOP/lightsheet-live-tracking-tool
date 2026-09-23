"""Merge several acquisitions of the same positions into one movie.

When an acquisition is started, stopped and started again, the result
is several experiment folders holding the same positions, each one
numbering its timepoints from scratch:

    20260918_101500_Experiment/Position 1_Settings 1/t0000_BF.tif ...
    20260918_143000_Experiment/Position 1_Settings 1/t0000_BF.tif ...

This tool builds ONE new experiment folder where every position holds
the timepoints of all the runs, in run order, renumbered 0..N-1:

    merged/Position 1_Settings 1/t0000_BF.tif ... t0137_BF.tif

Only the timepoint number changes. The channel part of the name
(``_BF``, ``_C01``, ``_Channel 1``, ``_BF_z003`` ...) is kept as is, so
every channel of a timepoint follows it to its new number. Runs are
taken in the order given on the command line; a glob is expanded and
sorted, which for timestamped folder names is chronological. The order
is printed before anything is written — check it.

Timepoints are renumbered by rank, so a gap inside a run (t0003 missing)
is closed rather than carried over; it is reported. Everything that is
not a ``t{N}...tif`` file directly in a position folder (embryo_tracking/,
tracking_RoIs.json, logs, companions) is NOT carried over: those belong
to one run and would be wrong for the merged movie.

Sources are never modified unless --move is given. A ``merge_log.csv``
in the output root records where every merged file came from.

Usage:
    # Merge every run matching a glob (quote it so the shell leaves it alone)
    python tools/merge_movies.py "D:/data/20260918_*_Experiment" -o D:/data/merged

    # Explicit runs, in this order
    python tools/merge_movies.py D:/data/run_a D:/data/run_b -o D:/data/merged

    # See the plan first
    python tools/merge_movies.py "D:/data/2026*" -o D:/data/merged --dry-run

    # Hard links instead of copies: instant, no extra disk space
    # (output must be on the same drive as the runs)
    python tools/merge_movies.py "D:/data/2026*" -o D:/data/merged --link

Inputs can also be position folders directly (the folders holding the
TIFFs); the merged position then takes the name of the first one.

Exit codes:
    0  merged (or, with --dry-run, would be)
    1  bad input, nothing to merge, or the output would overwrite files
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sys
from collections import defaultdict

#: ``t0012_BF.tif`` -> timepoint '0012', rest '_BF'. Same shape as the
#: regex in tracking_tools/ome_companion/companion.py.
_TNAME = re.compile(r"^t(\d+)((?:_.*)?\.tiff?)$", re.IGNORECASE)

LOG_NAME = "merge_log.csv"


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _natural_key(path: str):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path)]


def expand_inputs(patterns: list[str]) -> list[str]:
    """Expand globs in argument order, each expansion sorted; drop repeats."""
    runs: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat), key=_natural_key) if glob.has_magic(pat) \
            else [pat]
        for h in hits:
            h = os.path.abspath(h)
            if os.path.isdir(h) and h not in runs:
                runs.append(h)
    return runs


def scan_folder(folder: str) -> dict[int, list[tuple[str, str]]]:
    """``{timepoint: [(filename, rest), ...]}`` for the TIFFs in *folder*."""
    tps: dict[int, list[tuple[str, str]]] = defaultdict(list)
    with os.scandir(folder) as it:
        for e in it:
            m = _TNAME.match(e.name)
            if m and e.is_file():
                tps[int(m.group(1))].append((e.name, m.group(2)))
    return dict(tps)


def is_position(folder: str) -> bool:
    return bool(scan_folder(folder))


def collect(runs: list[str]) -> dict[str, list[tuple[str, str, dict]]]:
    """``{position: [(run, position folder, scan), ...]}`` in run order.

    Raises ValueError when inputs mix experiment roots and position
    folders, since the two cannot be lined up with each other.
    """
    kinds = {r: is_position(r) for r in runs}
    if len(set(kinds.values())) > 1:
        pos = [r for r, k in kinds.items() if k]
        raise ValueError("inputs mix position folders and experiment roots "
                         f"(position folders: {pos})")

    positions: dict[str, list[tuple[str, str, dict]]] = {}
    if all(kinds.values()):
        name = os.path.basename(runs[0])
        positions[name] = [(r, r, scan_folder(r)) for r in runs]
        return positions

    for run in runs:
        with os.scandir(run) as it:
            subs = sorted((e.path for e in it if e.is_dir()),
                          key=_natural_key)
        for sub in subs:
            scan = scan_folder(sub)
            if scan:
                positions.setdefault(os.path.basename(sub), []).append(
                    (run, sub, scan))
    return positions


def plan_position(parts: list[tuple[str, str, dict]], start: int, width: int,
                  drop_incomplete: bool, warn) -> list[tuple[str, str, int]]:
    """``[(source path, new name, new timepoint), ...]`` for one position."""
    plan = []
    t_new = start
    for _, folder, scan in parts:
        tps = sorted(scan)
        if tps and tps[-1] - tps[0] + 1 != len(tps):
            missing = sorted(set(range(tps[0], tps[-1] + 1)) - set(tps))
            warn(f"{folder}: timepoints {_short(missing)} missing, gap closed")
        n_ch = max(len(v) for v in scan.values())
        for t in tps:
            files = sorted(scan[t])
            if len(files) < n_ch:
                what = "dropped" if drop_incomplete else "kept"
                warn(f"{folder}: t{t} has {len(files)}/{n_ch} files, {what}")
                if drop_incomplete:
                    continue
            for fname, rest in files:
                plan.append((os.path.join(folder, fname),
                             f"t{t_new:0{width}d}{rest}", t_new))
            t_new += 1
    return plan


def _short(nums: list[int], limit: int = 8) -> str:
    s = ", ".join(map(str, nums[:limit]))
    return s + (f" ... (+{len(nums) - limit})" if len(nums) > limit else "")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge several acquisitions of the same positions into "
                    "one movie, renumbering timepoints from 0.")
    p.add_argument("inputs", nargs="+",
                   help="Experiment roots (or position folders), in order. "
                        "Globs are expanded here, so quote them.")
    p.add_argument("-o", "--output", required=True,
                   help="New experiment folder to create.")
    p.add_argument("--start", type=int, default=0,
                   help="First timepoint number of the merged movie "
                        "(default: 0).")
    p.add_argument("--width", type=int, default=4,
                   help="Zero-padding of the timepoint (default: 4, "
                        "widened automatically if needed).")
    p.add_argument("--positions", default="*",
                   help="Glob selecting positions by name (default: all).")
    p.add_argument("--drop-incomplete", action="store_true",
                   help="Skip timepoints with fewer files than the others "
                        "of the same run (e.g. the last one of a run that "
                        "was stopped mid-timepoint).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--link", action="store_true",
                      help="Hard-link instead of copy (same drive only).")
    mode.add_argument("--move", action="store_true",
                      help="Move the files out of the runs. Irreversible.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan; write nothing.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import fnmatch

    args = parse_args(argv)
    runs = expand_inputs(args.inputs)
    out = os.path.abspath(args.output)
    runs = [r for r in runs if r != out]
    if not runs:
        _log("No input folder found.")
        return 1

    _log("Runs, in merge order:")
    for i, r in enumerate(runs):
        _log(f"  {i + 1}. {r}")

    try:
        positions = collect(runs)
    except ValueError as e:
        _log(f"Error: {e}")
        return 1
    positions = {k: v for k, v in positions.items()
                 if fnmatch.fnmatch(k, args.positions)}
    if not positions:
        _log("No position with t*.tif files found.")
        return 1

    warnings: list[str] = []
    plans: dict[str, list] = {}
    for name, parts in positions.items():
        n_tp = sum(len(scan) for _, _, scan in parts)
        width = max(args.width, len(str(args.start + n_tp - 1)))
        plans[name] = plan_position(parts, args.start, width,
                                    args.drop_incomplete,
                                    lambda m, n=name: warnings.append(f"[{n}] {m}"))
        missing = [r for r in runs if r not in {r2 for r2, _, _ in parts}]
        if missing:
            warnings.append(f"[{name}] absent from {len(missing)} run(s), its "
                            f"timepoints will not line up with the other "
                            f"positions: {missing}")

    _log()
    _log(f"Output: {out}")
    for name, plan in plans.items():
        tps = sorted({t for _, _, t in plan})
        per_run = [len(scan) for _, _, scan in positions[name]]
        _log(f"  {name}: {len(tps)} timepoints "
             f"(t{tps[0]}..t{tps[-1]}), {len(plan)} files, "
             f"per run {per_run}")
    for w in warnings:
        _log(f"  WARNING {w}")

    clashes = [os.path.join(out, n, new) for n, plan in plans.items()
               for _, new, _ in plan if os.path.exists(os.path.join(out, n, new))]
    if clashes:
        _log(f"\nRefusing: {len(clashes)} file(s) already exist in the "
             f"output, e.g. {clashes[0]}")
        return 1

    if args.dry_run:
        _log("\nDry run, nothing written.")
        return 0

    op = os.link if args.link else shutil.move if args.move else shutil.copy2
    verb = "Linking" if args.link else "Moving" if args.move else "Copying"
    total = sum(len(p) for p in plans.values())
    done = 0
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, LOG_NAME), "a", newline="",
              encoding="utf-8") as fh:
        log = csv.writer(fh)
        if fh.tell() == 0:
            log.writerow(["position", "new_file", "timepoint", "source"])
        for name, plan in plans.items():
            os.makedirs(os.path.join(out, name), exist_ok=True)
            for src, new, t in plan:
                op(src, os.path.join(out, name, new))
                log.writerow([name, new, t, src])
                done += 1
                if done % 100 == 0 or done == total:
                    _log(f"  {verb} {done}/{total}")
    _log(f"\nDone. Provenance in {os.path.join(out, LOG_NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
