"""
Probe ZEN API for the currently-running experiment.

Run this while an experiment is acquiring in ZEN.  It prints:
  - the list of experiments ZEN can load from disk,
  - for each loadable experiment, its experiment_id and acquisition status,
  - whether the experiment is a tile/positions experiment, and if so,
    enough info to call TilesService.clear + TilesService.add_positions
    to update stored positions during a run.

Usage from the lightsheet-live-tracking-tool repo root:

    python tools/probe_running_experiment.py

Requires the same gRPC connection settings used by the panel app (host,
port, cert, control token).  By default it reads zeiss_config.ini from
the repo root; pass --config <path> to override.
"""

import argparse
import asyncio
import configparser
import os
import ssl
import sys

import grpclib.client

from zen_api.acquisition.v1beta import (
    ExperimentServiceStub,
    ExperimentServiceGetAvailableExperimentsRequest,
    ExperimentServiceGetStatusRequest,
    ExperimentServiceLoadRequest,
    ExperimentServiceExportRequest,
)
from zen_api.lm.acquisition.v1 import (
    TilesServiceStub,
    TilesServiceIsTilesExperimentRequest,
)


import re
import xml.etree.ElementTree as ET


def _extract_single_tile_regions(xml_str):
    """Pull positions out of ZEN's ``<SingleTileRegions>`` block.

    Verified against ``test-tracking-clement.czexp`` in 2026-05.  Schema::

        <SingleTileRegions>
          <SingleTileRegion Name="P1" Id="…">
            <X>0</X>
            <Y>0</Y>
            <Z>0</Z>
            …
          </SingleTileRegion>
          …
        </SingleTileRegions>

    Returns a list of ``(name, (x, y, z))`` tuples in whatever unit ZEN
    stored the values in (µm in observed samples).  X/Y/Z may live
    directly under ``<SingleTileRegion>`` or one level deeper inside a
    ``<Center>`` element — we search both.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return []

    # Strip namespaces so tag lookups are simple
    for elem in root.iter():
        elem.tag = re.sub(r'^\{[^}]+\}', '', elem.tag)

    def _f(node):
        if node is None or node.text is None:
            return None
        try:
            return float(node.text.strip())
        except ValueError:
            return None

    def _pick(region, child_name):
        # ``a or b`` would return b when a == 0.0; use ``is None``
        direct = _f(region.find(child_name))
        if direct is not None:
            return direct
        return _f(region.find(f'.//Center/{child_name}'))

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


def _extract_positions_from_xml(xml_str):
    """Find any (x, y, z) tuples that look like stored stage positions.

    ZEN's .czexp XML layout isn't fully documented, but in observed
    samples positions show up as elements containing X, Y (and optional Z)
    child tags under a region like ``SingleTileRegions`` or
    ``MultiTrackSetup/PositionList``.  Rather than hard-code one path we
    walk the tree and accept any element whose children include an ``X``
    and ``Y`` tag with a parseable float value (Z is optional).

    Returns a list of tuples ``(tag_path, (x, y, z))`` so the caller can
    see *where* in the XML each candidate came from — useful while
    figuring out the right XPath to use programmatically.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        return [], f"XML parse error: {e}"

    out = []
    # Strip namespaces for easier matching
    for elem in root.iter():
        elem.tag = re.sub(r'^\{[^}]+\}', '', elem.tag)

    def _to_float(s):
        if s is None:
            return None
        try:
            return float(s.strip())
        except (ValueError, AttributeError):
            return None

    for elem in root.iter():
        children = {c.tag: c for c in elem}
        x_node = children.get('X') or children.get('x')
        y_node = children.get('Y') or children.get('y')
        if x_node is None or y_node is None:
            continue
        x = _to_float(x_node.text)
        y = _to_float(y_node.text)
        if x is None or y is None:
            continue
        z_node = children.get('Z') or children.get('z')
        z = _to_float(z_node.text) if z_node is not None else 0.0
        out.append((elem.tag, (x, y, z)))
    return out, None


def _open_channel(cfg_path):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    host = cfg.get('host', 'address', fallback='localhost')
    port = cfg.getint('host', 'port', fallback=5002)
    cert = cfg.get('cert', 'path', fallback='')
    token = cfg.get('api', 'control_token', fallback='')

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if cert and os.path.exists(cert):
        ctx.load_verify_locations(cafile=cert)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
    else:
        print(f"  ! no cert at {cert!r} — verification disabled")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])

    channel = grpclib.client.Channel(host=host, port=port, ssl=ctx)
    metadata = [("control-token", token)]
    print(f"  → connected to {host}:{port}")
    return channel, metadata


async def main(cfg_path):
    channel, metadata = _open_channel(cfg_path)
    exp_svc   = ExperimentServiceStub(channel=channel, metadata=metadata)
    tile_svc  = TilesServiceStub(channel=channel, metadata=metadata)

    print("\n— Available experiments on disk —")
    avail = await exp_svc.get_available_experiments(
        ExperimentServiceGetAvailableExperimentsRequest()
    )
    if not avail.experiments:
        print("  (none)")
    for e in avail.experiments:
        # The field name on the response is `name`; print everything for
        # discoverability.
        print(f"  - {e}")

    print("\n— Per-experiment load + status + tiles probe —")
    for e in avail.experiments:
        name = getattr(e, 'name', None) or str(e)
        print(f"\n  • {name}")
        try:
            loaded = await exp_svc.load(
                ExperimentServiceLoadRequest(experiment_name=name)
            )
            exp_id = loaded.experiment_id
            print(f"      load → experiment_id = {exp_id!r}")
        except Exception as ex:
            print(f"      load FAILED: {ex}")
            continue

        try:
            st = await exp_svc.get_status(
                ExperimentServiceGetStatusRequest(experiment_id=exp_id)
            )
            print(f"      status: {st.status}")
        except Exception as ex:
            print(f"      get_status FAILED: {ex}")

        try:
            is_tiles = await tile_svc.is_tiles_experiment(
                TilesServiceIsTilesExperimentRequest(experiment_id=exp_id)
            )
            print(f"      is_tiles_experiment: "
                  f"{is_tiles.is_tiles_experiment}")
        except Exception as ex:
            print(f"      is_tiles_experiment FAILED: {ex}")

        # Try to recover stored positions from the experiment XML.
        try:
            xml_resp = await exp_svc.export(
                ExperimentServiceExportRequest(experiment_id=exp_id)
            )
            xml_str = xml_resp.xml
            if not xml_str:
                print("      export: <empty XML>")
            else:
                # Always save the full XML so we can grep / inspect it
                # offline.  This is the most useful artifact for figuring
                # out the position-tag schema when our heuristic walker
                # comes up empty.
                safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
                out_path = os.path.abspath(f'export_{safe_name}.xml')
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(xml_str)
                print(f"      export XML saved to: {out_path}  ({len(xml_str)} chars)")

                # Targeted parse for the ZEN <SingleTileRegions> schema
                stile = _extract_single_tile_regions(xml_str)
                if stile:
                    print(f"      SingleTileRegions: {len(stile)} entry/ies")
                    for nm, (x, y, z) in stile:
                        print(f"        {nm}: x={x}  y={y}  z={z}")

                positions, err = _extract_positions_from_xml(xml_str)
                if err:
                    print(f"      export XML: {err}")
                elif not positions:
                    print("      X/Y/Z heuristic found no positions.")
                    # Helpful hints for where to look next.
                    for needle in ('Position', 'Tile', 'SinglePosition',
                                   'TileRegion', 'CarrierMap', 'X='):
                        if needle.lower() in xml_str.lower():
                            print(f"        hint: substring {needle!r} "
                                  f"appears in the XML — open the file and "
                                  f"search for it to find the position schema.")
                else:
                    print(f"      export XML: {len(positions)} position-like element(s):")
                    for tag, (x, y, z) in positions:
                        print(f"        <{tag}>  x={x}  y={y}  z={z}")
        except Exception as ex:
            print(f"      export FAILED: {ex}")

    channel.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--config', default='zeiss_config.ini',
        help='Path to zeiss_config.ini (default: zeiss_config.ini in cwd)'
    )
    args = ap.parse_args()
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(args.config))
