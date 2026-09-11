"""Functional test of Definite Focus 2 on real hardware.

`mtb_smoke_test.py --df2` dumps the INTERFACE. This drives it, through
the same MTBDefiniteFocus wrapper the panel and the acquisition loop
use, and answers the only question that matters: after the focus has
been disturbed, does DF2 put it back?

That is exactly the experiment:

    1. learn the plane you are focused on, keeping its reference blob
    2. read Z
    3. deliberately defocus by --offset um
    4. restore the blob and stabilize
    5. read Z again and compare with step 2

A pass means DF2 returned the focus to within --tol um of where it
started. The round trip through the blob is deliberate: it is the
mechanism that lets each POSITION carry its own plane, and one global
reference cannot serve positions sitting at different heights.

SAFETY: this MOVES THE FOCUS by --offset um (default 5) and relies on
DF2 to bring it back. Make sure the objective has that much clearance
and that something is actually in focus before starting.

Usage:
    # Learn here, defocus 5 um, prove DF2 recovers
    python tools/df2_test.py

    # Bigger disturbance, tighter tolerance, most precise DF2 mode
    python tools/df2_test.py --offset 20 --tol 0.5 --mode Exact

    # Report only; nothing moves
    python tools/df2_test.py --report-only

Exit codes:
    0  DF2 learned and recovered the plane
    1  DF2 is present but did not recover it
    2  DF2 is absent, or a dependency is missing
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
        description="Prove Definite Focus 2 recovers a learned plane.",
    )
    p.add_argument("--offset", type=float, default=5.0,
                   help="Defocus in um to recover from (default: 5).")
    p.add_argument("--tol", type=float, default=1.0,
                   help="Acceptable residual error in um (default: 1).")
    p.add_argument("--mode", default="Balanced",
                   choices=("Exact", "Balanced", "Fast"),
                   help="DF2 precision (default: Balanced).")
    p.add_argument("--z-axis", default="piezo",
                   choices=("piezo", "focus"),
                   help="Which axis to read and disturb "
                        "(default: piezo).")
    p.add_argument("--report-only", action="store_true",
                   help="Print DF2's state and exit. Nothing moves.")
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat the disturb/recover cycle N times, to "
                        "see whether the error accumulates.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from tracking_tools.microscope_interface.mtb import (
            MTBError, MTBSession,
        )
    except Exception as e:
        _log(f"Cannot import the MTB layer: {e}")
        return 2

    _hdr("CONNECT")
    try:
        # shared(): MTB permits one Login per process, and logging out
        # burns the process for good.
        session = MTBSession.shared()
    except Exception as e:
        _log(f"MTB login failed: {e}")
        return 2
    _log(f"connected, clientId={session.client_id}")

    try:
        df2 = session.definite_focus()
    except MTBError as e:
        _log(f"No Definite Focus 2 on this stand: {e}")
        return 2
    _log(f"Definite Focus 2 found via {df2.interface}")

    axis = session.axis(args.z_axis)

    _hdr("STATE")
    _log(df2.describe())
    _log(f"  {args.z_axis} at {axis.position:.3f} um, "
         f"range {axis.limits[0]:.1f} .. {axis.limits[1]:.1f}")
    try:
        obj = session.objective()
        order = obj.order_number
        _log(f"  objective: {obj.name} (order {order})")
        if order:
            _log(f"  DF2 suitability: {df2.objective_suitable(order)}")
        else:
            _log("  (no order number readable, so DF2 suitability "
                 "cannot be checked — that is informational only)")
    except Exception as e:
        _log(f"  (could not check the objective: {e})")

    if args.report_only:
        _log()
        _log("--report-only: nothing moved.")
        return 0

    _hdr("SET PRECISION")
    if df2.set_operation_mode(args.mode):
        _log(f"mode set to {args.mode}")
    else:
        _log(f"could not set mode {args.mode}; using "
             f"{df2.operation_mode()}")

    _hdr("LEARN")
    _log("Learning the current plane. This takes several seconds.")
    start = time.monotonic()
    try:
        ref = df2.learn()
    except MTBError as e:
        _log(f"FAILED: {e}")
        _log("Is anything actually in focus? DF2 needs a surface.")
        return 1
    took = time.monotonic() - start
    z0 = axis.position
    if ref:
        _log(f"learned in {took:.1f}s, reference is {len(ref)} bytes")
        _log("  -> this blob is what a position stores, so each "
             "position can hold its own plane")
    else:
        _log(f"learned in {took:.1f}s, but NO reference blob came "
             f"back — this build can only hold ONE plane, so "
             f"per-position focus is not available")
    _log(f"plane is at {args.z_axis} = {z0:.3f} um")

    failures = []
    for cycle in range(1, max(1, args.repeat) + 1):
        _hdr(f"DISTURB AND RECOVER  (cycle {cycle})")
        target = z0 + args.offset
        lo, hi = axis.limits
        if not (lo <= target <= hi):
            target = z0 - args.offset
            if not (lo <= target <= hi):
                _log(f"cannot defocus by {args.offset} um without "
                     f"leaving the {lo:.1f}..{hi:.1f} um range")
                return 1
        axis.move_to(target)
        _log(f"defocused to {axis.position:.3f} um "
             f"({axis.position - z0:+.3f})")

        if ref:
            try:
                df2.restore(ref)
                _log("reference restored")
            except MTBError as e:
                _log(f"restore failed: {e}")

        start = time.monotonic()
        try:
            df2.stabilize_now()
        except MTBError as e:
            _log(f"FAILED: {e}")
            failures.append(f"cycle {cycle}: {e}")
            continue
        z1 = axis.position
        err = z1 - z0
        _log(f"stabilized in {time.monotonic() - start:.1f}s: "
             f"{z1:.3f} um, error {err:+.3f} um")
        if abs(err) > args.tol:
            failures.append(
                f"cycle {cycle}: residual {err:+.3f} um exceeds "
                f"{args.tol} um"
            )

    _hdr("RESULT")
    if failures:
        for f in failures:
            _log(f"  - {f}")
        _log()
        _log("DF2 DID NOT RECOVER THE PLANE.")
        _log("Try --mode Exact, a smaller --offset, or check that the "
             "objective is one DF2 supports.")
        return 1
    _log(f"DF2 recovered the plane within {args.tol} um, "
         f"{max(1, args.repeat)} time(s).")
    _log("DEFINITE FOCUS 2 TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
