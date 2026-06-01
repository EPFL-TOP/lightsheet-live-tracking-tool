"""
End-to-end smoke test for the API-started multi-position pipeline.

What this script does (and prints clearly at each step):

  1. Connect to the ZEN gateway via gRPC.
  2. List available experiments.
  3. Load the named experiment → fresh experiment_id.
  4. Export the experiment XML → parse <SingleTileRegion> entries →
     print P1, P2, … positions in µm.
  5. Start the experiment via the API → registered experiment_id.
  6. (Re-)export now that the experiment is registered, in case ZEN
     reports different positions for a running experiment.
  7. Subscribe to register_on_status_changed for N seconds → log each
     status event with scenes_index / is_acquisition_running / tp.
  8. Stop the experiment cleanly.
  9. PASS/FAIL summary.

Usage (from the repo root):

    python tools/smoke_test_multipos.py \
        --experiment test-tracking-clement \
        --output-name smoke_test_001 \
        --watch-seconds 60

If --output-name is already used in ZEN's image output folder, the
script prints the conflict and exits — pick a fresh name.

Requirements: ``zen_api`` Python package installed (whichever release
your gateway is compatible with) and ``zeiss_config.ini`` in the
working directory (or pass --config).
"""

import argparse
import asyncio
import configparser
import os
import re
import ssl
import sys
import time
import xml.etree.ElementTree as ET

import grpclib.client

from zen_api.acquisition.v1beta import (
    ExperimentServiceStub,
    ExperimentServiceGetAvailableExperimentsRequest,
    ExperimentServiceLoadRequest,
    ExperimentServiceStartExperimentRequest,
    ExperimentServiceStopRequest,
    ExperimentServiceExportRequest,
    ExperimentServiceRegisterOnStatusChangedRequest,
)
from zen_api.lm.acquisition.v1 import (
    TilesServiceStub,
    TilesServiceIsTilesExperimentRequest,
)


# ─── pretty printing ────────────────────────────────────────────────────────

_START_TS = time.time()


def _t():
    return f"+{time.time() - _START_TS:7.2f}s"


def _step(n, title):
    bar = "─" * (76 - len(title))
    print()
    print(f"━━━ STEP {n} — {title} {bar}")


def _ok(msg):
    print(f"  ✓ {_t()}  {msg}")


def _info(msg):
    print(f"    {_t()}  {msg}")


def _warn(msg):
    print(f"  ⚠ {_t()}  {msg}")


def _fail(msg):
    print(f"  ✗ {_t()}  {msg}")


# ─── helpers ────────────────────────────────────────────────────────────────

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


def _extract_single_tile_regions(xml_str):
    """Same parser as the panel uses: pull positions from
    <SingleTileRegion> elements; X/Y/Z may be direct children or under
    <Center>."""
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

    out = []
    for region in root.iter('SingleTileRegion'):
        name = region.get('Name') or '?'
        x = _f(region.find('X')) or _f(region.find('.//Center/X'))
        y = _f(region.find('Y')) or _f(region.find('.//Center/Y'))
        z = _f(region.find('Z')) or _f(region.find('.//Center/Z')) or 0.0
        if x is None or y is None:
            continue
        out.append((name, (x, y, z)))
    return out


# ─── main ───────────────────────────────────────────────────────────────────

async def main(cfg_path, exp_name, output_name, watch_seconds):
    overall_ok = True

    _step(1, "Connect to ZEN gateway")
    channel, metadata = _open_channel(cfg_path)
    exp = ExperimentServiceStub(channel=channel, metadata=metadata)
    tiles = TilesServiceStub(channel=channel, metadata=metadata)

    try:
        _step(2, "List available experiments")
        avail = await exp.get_available_experiments(
            ExperimentServiceGetAvailableExperimentsRequest()
        )
        names = [e.name for e in avail.experiments]
        if not names:
            _fail("no experiments returned by ZEN — is one configured?")
            return False
        for n in names:
            marker = " ← target" if n == exp_name else ""
            _info(f"- {n}{marker}")
        if exp_name not in names:
            _fail(f"requested experiment {exp_name!r} not in the list")
            return False
        _ok(f"target experiment found: {exp_name!r}")

        _step(3, f"Load {exp_name!r}")
        loaded = await exp.load(
            ExperimentServiceLoadRequest(experiment_name=exp_name)
        )
        exp_id = loaded.experiment_id
        _ok(f"experiment_id = {exp_id}")

        _step(4, "Export XML, parse <SingleTileRegion>")
        xml_resp = await exp.export(
            ExperimentServiceExportRequest(experiment_id=exp_id)
        )
        xml_str = xml_resp.xml or ""
        _info(f"XML length: {len(xml_str)} chars")

        is_tiles = await tiles.is_tiles_experiment(
            TilesServiceIsTilesExperimentRequest(experiment_id=exp_id)
        )
        _info(f"is_tiles_experiment: {is_tiles.is_tiles_experiment}")

        positions = _extract_single_tile_regions(xml_str)
        if not positions:
            _fail("no <SingleTileRegion> entries found in the XML")
            overall_ok = False
        else:
            _ok(f"{len(positions)} position(s) parsed from XML:")
            for name, (x, y, z) in positions:
                _info(f"    {name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm")

        _step(5, f"Start experiment as {output_name!r}.czi via API")
        try:
            await exp.start_experiment(
                ExperimentServiceStartExperimentRequest(
                    experiment_id=exp_id,
                    output_name=output_name,
                )
            )
            _ok(f"start_experiment OK, experiment_id={exp_id}")
        except Exception as e:
            code = getattr(getattr(e, 'status', None), 'value', None)
            msg  = getattr(e, 'message', None) or str(e)
            if code == 6 or 'already exists' in str(e).lower():
                _fail(f"output name {output_name!r}.czi already exists — "
                      f"pick a fresh one via --output-name")
            else:
                _fail(f"start_experiment failed: {msg}")
            return False

        _step(6, "Re-export XML (now experiment is registered)")
        try:
            xml_resp2 = await exp.export(
                ExperimentServiceExportRequest(experiment_id=exp_id)
            )
            positions2 = _extract_single_tile_regions(xml_resp2.xml or "")
            if positions2 != positions:
                _warn("positions differ between pre- and post-start export!")
                for name, (x, y, z) in positions2:
                    _info(f"  [post-start] {name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm")
            else:
                _ok("positions identical pre/post-start (good)")
        except Exception as e:
            _warn(f"post-start export failed: {e}")

        _step(7, f"Stream status for {watch_seconds}s")
        event_count = 0
        scene_idx_seen = set()
        deadline = time.time() + watch_seconds

        async def _stream():
            nonlocal event_count
            async for resp in exp.register_on_status_changed(
                ExperimentServiceRegisterOnStatusChangedRequest(exp_id)
            ):
                s = resp.status
                event_count += 1
                idx = int(getattr(s, 'scenes_index', -1))
                if idx >= 0:
                    scene_idx_seen.add(idx)
                acq = bool(s.is_acquisition_running)
                running = bool(s.is_experiment_running)
                tp = int(getattr(s, 'time_points_index', 0))
                imgs = int(getattr(s, 'images_acquired_index', 0))
                _info(
                    f"[evt #{event_count:03d}] exp_run={running} acq={acq} "
                    f"scene={idx} tp={tp} imgs={imgs}"
                )
                if time.time() >= deadline:
                    return
                if not running:
                    _info("experiment ended on its own — exit stream")
                    return

        try:
            await asyncio.wait_for(_stream(), timeout=watch_seconds + 5)
        except asyncio.TimeoutError:
            _info(f"stream timeout reached after {watch_seconds + 5}s")
        except Exception as e:
            _warn(f"status stream stopped: {e}")

        _ok(f"received {event_count} status event(s), "
            f"distinct scene indices seen: {sorted(scene_idx_seen)}")
        if positions and scene_idx_seen and \
                set(range(len(positions))) - scene_idx_seen:
            _warn(
                "Some scenes never reported is_acquisition_running. "
                "ZEN may not advance scenes_index inside a cycle — "
                "manual baselines remain the safer path."
            )

        _step(8, "Stop the experiment")
        try:
            await exp.stop(ExperimentServiceStopRequest(experiment_id=exp_id))
            _ok("stop_experiment OK")
        except Exception as e:
            _warn(f"stop_experiment failed (often harmless): {e}")

        _step(9, "SUMMARY")
        if overall_ok and positions:
            print("  ✓ Smoke test PASSED.")
            print(f"      Positions seen by the API:")
            for name, (x, y, z) in positions:
                print(f"        {name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm")
            print()
            print("  Paste those values into the panel's"
                  " 'Initial scene positions' text area, e.g.:")
            print()
            for _, (x, y, z) in positions:
                print(f"      {x}, {y}, {z}")
            return True
        else:
            print("  ✗ Smoke test FAILED — see warnings above.")
            return False
    finally:
        channel.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="API-started multi-position smoke test."
    )
    ap.add_argument('--config', default='zeiss_config.ini',
                    help='Path to zeiss_config.ini (default: cwd)')
    ap.add_argument('--experiment', required=True,
                    help='Experiment name to load (e.g. test-tracking-clement)')
    ap.add_argument('--output-name', required=True,
                    help='Output filename (no .czi) — must not already exist')
    ap.add_argument('--watch-seconds', type=int, default=60,
                    help='How many seconds to listen for status events')
    args = ap.parse_args()
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    ok = asyncio.run(main(args.config, args.experiment,
                          args.output_name, args.watch_seconds))
    sys.exit(0 if ok else 1)
