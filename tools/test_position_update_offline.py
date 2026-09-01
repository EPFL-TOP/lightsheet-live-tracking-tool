"""
Verify that TilesService.{clear, add_positions} actually mutates the
positions of an experiment that has been Loaded via the API but has NOT
been started.

This is the precondition for the new architecture Zeiss themselves
suggested: drive the experiment one timepoint at a time from our side,
updating positions between iterations.

What the script does:

  1. Connect to ZEN.
  2. Load --experiment by name → fresh experiment_id.
  3. Read positions via ExperimentService.Export + <SingleTileRegion>.
  4. Apply a known shift (--dx --dy --dz, defaults 50, 50, 0 µm) to
     every position.
  5. Call TilesService.clear + TilesService.add_positions with the new
     values.
  6. Re-read positions via Export and confirm they match the expected
     post-shift values.
  7. PASS/FAIL summary + reminder to check the ZEN UI.

If this PASSES, we have the building block for the new cycle loop.
If it FAILS, the API call is silently no-oping on a non-running
experiment and we need to revisit (e.g., maybe a Clone is required
first, à la zenapi_positions.py).

Usage (from the repo root, on the Windows ZEN machine):

    python tools/test_position_update_offline.py --experiment test-tracking-clement

Add --no-revert to leave the positions shifted (default reverts at the
end so the .czexp stays clean).  Add --dx/--dy/--dz to control the
shift in µm.
"""

import argparse
import asyncio
import configparser
import os
import re
import ssl
import sys
import xml.etree.ElementTree as ET

import grpclib.client

from zen_api.acquisition.v1beta import (
    ExperimentServiceStub,
    ExperimentServiceLoadRequest,
    ExperimentServiceExportRequest,
)
from zen_api.lm.acquisition.v1 import (
    TilesServiceStub,
    TilesServiceIsTilesExperimentRequest,
    TilesServiceClearRequest,
    TilesServiceAddPositionsRequest,
    Position3D,
)


# ─── tiny pretty-print helpers ──────────────────────────────────────────────

def _step(n, title):
    print()
    print(f"━━━ STEP {n} — {title} {'─' * max(0, 60 - len(title))}")


def _ok(msg):
    print(f"  ✓ {msg}")


def _info(msg):
    print(f"    {msg}")


def _warn(msg):
    print(f"  ⚠ {msg}")


def _fail(msg):
    print(f"  ✗ {msg}")


# ─── helpers reused from the smoke test ─────────────────────────────────────

def _open_channel(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    host  = cfg.get('host', 'address', fallback='localhost')
    port  = cfg.getint('host', 'port', fallback=5002)
    cert  = cfg.get('cert', 'path', fallback='')
    token = cfg.get('api', 'control_token', fallback='')

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if cert and os.path.exists(cert):
        ctx.load_verify_locations(cafile=cert)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
    else:
        _warn(f"no cert at {cert!r} — TLS verification disabled")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])
    channel = grpclib.client.Channel(host=host, port=port, ssl=ctx)
    metadata = [("control-token", token)]
    _ok(f"connected to {host}:{port}")
    return channel, metadata


def _extract_positions_um(xml_str):
    """Pull (name, (x_um, y_um, z_um)) for each <SingleTileRegion>."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        _fail(f"XML parse error: {e}")
        return []
    for elem in root.iter():
        elem.tag = re.sub(r'^\{[^}]+\}', '', elem.tag)

    def _f(node):
        if node is None or node.text is None:
            return None
        try:
            return float(node.text.strip())
        except ValueError:
            return None

    def _pick(region, name):
        direct = _f(region.find(name))
        if direct is not None:
            return direct
        return _f(region.find(f'.//Center/{name}'))

    out = []
    for region in root.iter('SingleTileRegion'):
        name = region.get('Name') or '?'
        x = _pick(region, 'X')
        y = _pick(region, 'Y')
        z = _pick(region, 'Z')
        if z is None:
            z = 0.0
        if x is None or y is None:
            continue
        out.append((name, (x, y, z)))
    return out


# ─── main ───────────────────────────────────────────────────────────────────

async def _read_positions(exp_svc, exp_id):
    xml_resp = await exp_svc.export(
        ExperimentServiceExportRequest(experiment_id=exp_id)
    )
    return _extract_positions_um(xml_resp.xml or "")


async def _push_positions(tiles_svc, exp_id, positions_um):
    """positions_um: list of (name, (x, y, z)) tuples in µm."""
    await tiles_svc.clear(TilesServiceClearRequest(experiment_id=exp_id))
    pos_msgs = [
        Position3D(x=x * 1e-6, y=y * 1e-6, z=z * 1e-6)
        for _, (x, y, z) in positions_um
    ]
    await tiles_svc.add_positions(TilesServiceAddPositionsRequest(
        experiment_id=exp_id, positions=pos_msgs,
    ))


async def main(cfg_path, exp_name, dx, dy, dz, revert):
    _step(1, "Connect to ZEN gateway")
    channel, metadata = _open_channel(cfg_path)
    exp_svc   = ExperimentServiceStub(channel=channel, metadata=metadata)
    tiles_svc = TilesServiceStub(channel=channel, metadata=metadata)

    try:
        _step(2, f"Load {exp_name!r} (NOT started)")
        loaded = await exp_svc.load(
            ExperimentServiceLoadRequest(experiment_name=exp_name)
        )
        exp_id = loaded.experiment_id
        _ok(f"experiment_id = {exp_id}")

        is_tiles = await tiles_svc.is_tiles_experiment(
            TilesServiceIsTilesExperimentRequest(experiment_id=exp_id)
        )
        if not is_tiles.is_tiles_experiment:
            _fail("Not a tiles experiment — TilesService.add_positions is a no-op.")
            return False
        _ok("is_tiles_experiment: True")

        _step(3, "Read CURRENT positions (Export)")
        before = await _read_positions(exp_svc, exp_id)
        if not before:
            _fail("No <SingleTileRegion> in exported XML — nothing to modify.")
            return False
        for name, (x, y, z) in before:
            _info(f"{name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm")

        # Build expected post-shift positions
        shifted = [
            (name, (x + dx, y + dy, z + dz))
            for name, (x, y, z) in before
        ]

        _step(4, f"Apply shift ({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) µm to every position")
        for name, (x, y, z) in shifted:
            _info(f"{name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm  ← target")

        _step(5, "TilesService.clear + TilesService.add_positions")
        await _push_positions(tiles_svc, exp_id, shifted)
        _ok("TilesService calls returned OK")

        _step(6, "Read positions AGAIN (Export) to verify")
        after = await _read_positions(exp_svc, exp_id)
        for name, (x, y, z) in after:
            _info(f"{name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm")

        _step(7, "Compare")
        # Tolerance because we're round-tripping through µm → m → µm
        TOL = 1e-2  # 10 nm
        ok = True
        if len(after) != len(shifted):
            _fail(
                f"position count changed: {len(before)} → {len(after)}  "
                f"(expected {len(shifted)})"
            )
            ok = False
        else:
            for (n1, (x1, y1, z1)), (n2, (x2, y2, z2)) in zip(shifted, after):
                if (abs(x1 - x2) > TOL
                        or abs(y1 - y2) > TOL
                        or abs(z1 - z2) > TOL):
                    _fail(
                        f"{n2}: expected ({x1:+.3f},{y1:+.3f},{z1:+.3f}), "
                        f"got ({x2:+.3f},{y2:+.3f},{z2:+.3f})"
                    )
                    ok = False
        if ok:
            _ok("Positions read back match expected post-shift values.")

        if revert:
            _step(8, "Revert to original positions (cleanup)")
            await _push_positions(tiles_svc, exp_id, before)
            verify = await _read_positions(exp_svc, exp_id)
            if verify == before:
                _ok("Revert OK — experiment back to original positions.")
            else:
                _warn("Revert read-back doesn't exactly match original — "
                      "check the .czexp before re-running.")

        _step(9, "SUMMARY")
        if ok:
            print("  ✓ TilesService can MODIFY positions on a loaded "
                  "(not running) experiment.")
            print()
            print("  Next step: open ZEN's Position list / Tiles UI and "
                  "confirm that")
            print("    a) when --no-revert: P1/P2/P3 show the SHIFTED "
                  "values in the UI.")
            print("    b) when reverting: P1/P2/P3 are back to the "
                  "original values.")
            print()
            print("  If the ZEN UI matches the API read-back, we have "
                  "the building block")
            print("  for cycle-by-cycle position updates.")
            return True
        else:
            print("  ✗ TilesService did NOT modify positions as expected.")
            print("  Possibilities:")
            print("    - clone is required first (zenapi_positions.py "
                  "does this)")
            print("    - save is required after add_positions")
            print("    - experiment ID went stale between calls")
            return False
    finally:
        channel.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='zeiss_config.ini',
                    help='Path to zeiss_config.ini (default: cwd)')
    ap.add_argument('--experiment', required=True,
                    help='Experiment name (without .czexp)')
    ap.add_argument('--dx', type=float, default=50.0,
                    help='X shift in µm (default 50)')
    ap.add_argument('--dy', type=float, default=50.0,
                    help='Y shift in µm (default 50)')
    ap.add_argument('--dz', type=float, default=0.0,
                    help='Z shift in µm (default 0)')
    ap.add_argument('--no-revert', dest='revert', action='store_false',
                    help='Leave positions shifted at the end (do not '
                         'restore the originals)')
    args = ap.parse_args()
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(2)
    ok = asyncio.run(main(args.config, args.experiment,
                          args.dx, args.dy, args.dz, args.revert))
    sys.exit(0 if ok else 1)
