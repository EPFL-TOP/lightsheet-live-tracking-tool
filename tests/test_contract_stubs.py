"""Unit tests for the LS1/ZEN contract-compatibility no-op stubs.

The tracking runner has two entry points (``run_zeiss`` and ``run_LS1``)
that call a common set of methods on the microscope interface —
``wait_for_pause``, ``pause_after_position``, ``no_pause_after_position``,
``continue_from_pause``, and ``refresh_filename``.  Not every backend
has a physical concept for all of them; where the concept is absent the
method must still exist as a no-op so any backend can be dispatched
through any runner path without ``AttributeError``.

These tests guard those stubs against accidental removal.

Run with:

    pytest tests/test_contract_stubs.py -v
"""

from __future__ import annotations

import sys
import time
import types
import unittest.mock as mock
from pathlib import Path

import pytest


# Ensure the repo root is on sys.path so ``tracking_tools`` resolves
# regardless of pytest's rootdir / invocation cwd.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Fake heavy dependencies before importing MicroscopeInterface.
#
# The module unconditionally imports numpy / tifffile at module load; those
# are lightweight and already required elsewhere.  gRPC / zen_api / pymcs /
# pymmcore_plus are imported LATE (inside methods), so simply *not*
# instantiating those code paths is enough — however MM's __init__ probes
# ``import pymmcore_plus`` so we register the same fakes test_mm_interface
# uses, keeping this module safe to import alongside it.
# ---------------------------------------------------------------------------


def _install_fake_modules():
    """Register minimal fakes for optional heavy deps."""
    if 'pymmcore_plus' not in sys.modules:
        fake_pmp = types.ModuleType('pymmcore_plus')
        fake_pmp.CMMCorePlus = mock.MagicMock(name='CMMCorePlus_stub')
        sys.modules['pymmcore_plus'] = fake_pmp
    if 'useq' not in sys.modules:
        fake_useq = types.ModuleType('useq')

        class _FakeMDAEvent:  # minimal placeholder
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
        fake_useq.MDAEvent = _FakeMDAEvent
        sys.modules['useq'] = fake_useq


_install_fake_modules()

from tracking_tools.microscope_interface.MicroscopeInterface import (  # noqa: E402
    MicroscopeInterface_Files,
    MicroscopeInterface_Zeiss,
)


# ---------------------------------------------------------------------------
# MicroscopeInterface_Files pause-cycle stubs
# ---------------------------------------------------------------------------


@pytest.fixture
def files_iface(tmp_path):
    """A minimally-constructed MicroscopeInterface_Files.

    ``positions_config`` is empty so ``__init__`` doesn't try to parse any
    filename patterns; that keeps the fixture hardware-free.
    """
    return MicroscopeInterface_Files(
        positions_config={},
        dirpath=str(tmp_path),
        file_params={},
    )


def test_files_pause_stubs_are_noop_and_return_none(files_iface):
    """All three pause-cycle stubs must exist, be callable, return None,
    and have no visible side effect on the object's state."""
    iface = files_iface

    # Snapshot public/private state before the stub calls
    snapshot_keys = set(vars(iface).keys())

    assert iface.pause_after_position() is None
    assert iface.no_pause_after_position() is None
    assert iface.continue_from_pause() is None

    # No attributes added, none removed.
    assert set(vars(iface).keys()) == snapshot_keys


def test_files_wait_for_pause_delegates_to_wait_for_image(files_iface):
    """wait_for_pause is an alias for wait_for_image on this backend.

    Both must return the same triple-None on an empty queue within a
    small timeout budget.
    """
    iface = files_iface

    # wait_for_image path (already tested elsewhere for behaviour) — we
    # exercise it here purely so wait_for_pause has an oracle to match.
    t0 = time.monotonic()
    from_image = iface.wait_for_image(timeout_ms=50)
    elapsed_image = time.monotonic() - t0

    t0 = time.monotonic()
    from_pause = iface.wait_for_pause(timeout_ms=50)
    elapsed_pause = time.monotonic() - t0

    # Same shape, same values (both should be triple-None on an empty
    # queue).  Comparing identity is over-strict; equality is enough.
    assert from_image == from_pause == (None, None, None)

    # Both must actually wait ~50 ms (proves they hit the queue-empty
    # timeout branch rather than short-circuiting).
    assert 0.02 <= elapsed_image < 0.5
    assert 0.02 <= elapsed_pause < 0.5


def test_files_wait_for_pause_signature_matches_wait_for_image(files_iface):
    """wait_for_pause must accept the same ``timeout_ms`` keyword."""
    iface = files_iface
    # Passing as keyword AND positional both work (contract with runner).
    assert iface.wait_for_pause(10) == (None, None, None)
    assert iface.wait_for_pause(timeout_ms=10) == (None, None, None)


# ---------------------------------------------------------------------------
# MicroscopeInterface_Zeiss refresh_filename stub
# ---------------------------------------------------------------------------


@pytest.fixture
def zeiss_iface(tmp_path):
    """A MicroscopeInterface_Zeiss instantiated without any gRPC I/O.

    __init__ only sets attributes and creates a queue/thread event — it
    does not touch the network, so no additional mocking is needed here.
    """
    zeiss_params = {
        'address':          'localhost',
        'port':             5002,
        'cert_path':        '',
        'control_token':    '',
        'experiment_name':  'unit-test-exp',
        'z_projection':     'max',
        'tracking_channel': 0,
        'max_xy_um':        500.0,
        'max_z_um':         100.0,
    }
    return MicroscopeInterface_Zeiss(
        positions_config={'scene_0': {'filename': 't0000_C00.tif'}},
        dirpath=str(tmp_path),
        zeiss_params=zeiss_params,
    )


def test_zeiss_refresh_filename_is_noop(zeiss_iface):
    """refresh_filename must exist, be callable with a pos_name, and
    return None without raising or mutating state."""
    iface = zeiss_iface

    snapshot_keys = set(vars(iface).keys())

    assert iface.refresh_filename('scene_0') is None
    assert iface.refresh_filename('nonexistent_scene') is None  # no lookup

    # No attribute drift.
    assert set(vars(iface).keys()) == snapshot_keys


def test_zeiss_refresh_filename_has_docstring():
    """A docstring is load-bearing here — it documents WHY the method is a
    no-op (users who need mid-run channel switches must switch backend).
    Guard against accidental removal.
    """
    doc = MicroscopeInterface_Zeiss.refresh_filename.__doc__
    assert doc is not None and doc.strip(), (
        "refresh_filename must retain its explanatory docstring"
    )
