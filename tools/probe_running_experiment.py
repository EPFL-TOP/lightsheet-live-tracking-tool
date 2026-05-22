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
)
from zen_api.lm.acquisition.v1 import (
    TilesServiceStub,
    TilesServiceIsTilesExperimentRequest,
)


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
