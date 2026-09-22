"""Build an OME companion for movies that are ALREADY on disk.

Writes a small XML index, ``ome-tiff.companion.ome``, that tells Fiji,
Bio-Formats, OMERO and tifffile how the per-timepoint TIFFs in an
experiment compose into a 5-D (T, C, Z, Y, X) series.

NOTHING IS REWRITTEN. No pixel is copied, re-encoded or renamed; the
TIFFs are opened read-only for their headers and nothing else. That is
the whole point: one Viventis timepoint is 1.06 GB, so consolidating a
series would mean rewriting hundreds of gigabytes to gain nothing.

It expects the layout every backend in this project writes:

    <experiment root>/<position>/t{NNNN}_{channel}.tif

and copes with the variations actually found in the wild: a 3-digit
timepoint with no channel suffix (the Cell Tracking Challenge sets), a
channel name containing spaces ('Channel 1'), LZW-compressed data, 2-D
frames alongside z-stacks, and positions with no images at all.

Two companions are written per run, because two readers resolve paths
differently:

  * ``<root>/ome-tiff.companion.ome``  — every position, referenced as
    ``<position>/<file>.tif``. This is the one to open in Fiji: it
    presents the whole experiment as a multi-series dataset.
  * ``<root>/<position>/ome-tiff.companion.ome`` — one per position,
    bare filenames. tifffile resolves references against the directory
    of the member file you opened, so it needs these; it is also what
    lets a single position folder be handed to a collaborator intact.

Usage:
    # Index an experiment, writing both companion levels
    python tools/make_ome_companion.py /data/20260918_run

    # See what it would do first
    python tools/make_ome_companion.py /data/20260918_run --dry-run

    # Supply the calibration the files cannot report
    python tools/make_ome_companion.py /data/run --pixel-size 0.65 \
        --z-step 1.5 --interval 60

    # Only some positions; only the per-position files
    python tools/make_ome_companion.py /data/run --positions 'scene_0*' --no-root

Open the result in Fiji with File > Import > Bio-Formats, or in Python:

    import tifffile
    with tifffile.TiffFile('pos/t0000_BF.tif',
                           omexml=open('pos/ome-tiff.companion.ome').read(),
                           is_ome=True) as tf:
        movie = tf.series[0].asarray()      # (T, C, Z, Y, X)

Exit codes:
    0  companion(s) written (or, with --dry-run, would be)
    1  nothing indexable found, or the scan failed
    2  a dependency is missing
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _hdr(title: str) -> None:
    _log()
    _log("=" * 70)
    _log(title)
    _log("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index existing TIFFs with an OME companion file. "
                    "Rewrites nothing.",
    )
    p.add_argument("root", help="Experiment root: the folder holding "
                                "the position folders.")
    p.add_argument("--positions", default="*",
                   help="Glob selecting position folders "
                        "(default: all).")
    p.add_argument("--pixel-size", type=float, default=None,
                   metavar="UM",
                   help="XY pixel size in um. Omitted means the "
                        "companion carries no XY calibration, which is "
                        "better than a wrong one.")
    p.add_argument("--z-step", type=float, default=None, metavar="UM",
                   help="Spacing between z-planes in um.")
    p.add_argument("--interval", type=float, default=None, metavar="S",
                   help="Seconds between timepoints.")
    p.add_argument("--no-root", action="store_true",
                   help="Skip the experiment-level companion.")
    p.add_argument("--no-per-position", action="store_true",
                   help="Skip the per-position companions.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written; write nothing.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite a companion that already exists. "
                        "Without this, an existing one is left alone "
                        "so a hand-edited or vendor-written file is "
                        "never clobbered.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from tracking_tools.ome_companion import (
            COMPANION_NAME, CompanionError, write_companion,
        )
    except ImportError as e:
        _log(f"Cannot import the companion builder: {e}")
        return 2

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        _log(f"Not a directory: {root}")
        return 1

    _hdr("SCAN")
    _log(f"root: {root}")

    # Refuse to clobber silently. A folder from the lightsheet may
    # already hold a vendor-written companion.
    if not args.force and not args.dry_run:
        existing = [
            p for p in [os.path.join(root, COMPANION_NAME)] + [
                os.path.join(root, d, COMPANION_NAME)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ] if os.path.exists(p)
        ]
        if existing:
            _log("A companion already exists:")
            for p in existing:
                _log(f"  {os.path.relpath(p, root)}")
            _log()
            _log("Re-run with --force to overwrite, or --dry-run to "
                 "see what would change.")
            return 1

    metadata = {
        "pixel_size_um": args.pixel_size,
        "z_step_um": args.z_step,
        "interval_s": args.interval,
    }

    try:
        report = write_companion(
            root,
            pattern=args.positions,
            per_position=not args.no_per_position,
            root_level=not args.no_root,
            metadata=metadata,
            dry_run=args.dry_run,
        )
    except CompanionError as e:
        _log(f"FAILED: {e}")
        return 1
    except Exception as e:                       # pragma: no cover
        _log(f"FAILED: {type(e).__name__}: {e}")
        return 1

    _hdr("FOUND")
    for name, info in sorted(report["positions"].items()):
        chans = ", ".join(c or "<unnamed>" for c in info["channels"])
        _log(f"{name}")
        _log(f"    {info['files']} file(s): "
             f"T={info['timepoints']}  C={len(info['channels'])} "
             f"[{chans}]  Z={info['planes']}  "
             f"{info['shape'][0]}x{info['shape'][1]}  {info['dtype']}")
        if info["images"] > 1:
            _log(f"    split into {info['images']} OME Images - the "
                 f"files do not all share one shape")

    if report["skipped"]:
        _log()
        _log(f"No indexable TIFFs, skipped: "
             f"{', '.join(report['skipped'])}")

    if report["warnings"]:
        _hdr("WARNINGS")
        for w in report["warnings"]:
            _log(f"  - {w}")

    _hdr("WRITTEN" if not args.dry_run else "WOULD WRITE")
    for p in report["written"]:
        _log(f"  {os.path.relpath(p, root)}")

    if args.dry_run:
        _log()
        _log("--dry-run: nothing was written.")
        return 0

    _log()
    _log("Open in Fiji:  File > Import > Bio-Formats  ->  "
         f"{COMPANION_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
