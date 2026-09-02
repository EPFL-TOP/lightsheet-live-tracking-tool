# Micro-Manager backend — the flagship path forward

## 1. What

`MicroscopeInterface_Micromanager` is the **flagship hardware backend** for
closed-loop live tracking. It talks to any Micro-Manager-supported
microscope through [pymmcore-plus](https://pymmcore-plus.github.io/) +
[useq-schema](https://pymmcore-plus.github.io/useq-schema/), which makes
it vendor-agnostic — the same tracker code drives a Zeiss Axio Observer,
a Nikon Ti, or a Leica DMi8 without touching the runner. It ships with a
**DemoCamera** path so you can validate the whole closed loop end-to-end
on a laptop with no hardware attached.

The implementation lives at
[`tracking_tools/microscope_interface/MicroscopeInterface.py`](../tracking_tools/microscope_interface/MicroscopeInterface.py)
(`MicroscopeInterface_Micromanager`), with synthetic image sources in
[`tracking_tools/microscope_interface/synthetic_source.py`](../tracking_tools/microscope_interface/synthetic_source.py).

---

## 2. Why we pivoted from the ZEN backend

We spent Q1–Q2 shipping the ZEN backend
([`MicroscopeInterface_Zeiss`](../tracking_tools/microscope_interface/MicroscopeInterface.py),
[`MicroscopeInterface_Files`](../tracking_tools/microscope_interface/MicroscopeInterface.py)
with `zen_feedback=True`, and the whole
[Zeiss panel walkthrough](zeiss_panel_walkthrough.md) workflow) and hit
a fundamental wall: **ZEN does not allow the API to modify scene
positions while an experiment is running**. Zeiss support confirmed
this in writing. `TilesService.add_positions` succeeds silently but the
running experiment keeps using the position list it loaded at
`StartExperiment` time — subsequent updates are only picked up on the
next start, which defeats the entire point of closed-loop tracking.
Micro-Manager solves this cleanly because **we own the acquisition
loop**: `pymmcore-plus` gives us an event-queue-backed MDA where each
timepoint's per-scene stage targets are computed by us, right before
the frame is snapped. The tracker's drift correction lands on the
*very next* MDAEvent — no vendor RPC contract to renegotiate, no
"restart the experiment" workaround.

---

## 3. Quickstart — closed-loop tracking with zero hardware in five minutes

```bash
pip install pymmcore-plus useq-schema scipy
mmcore install     # first-time only, downloads MMCore + DemoCamera adapters (~2 min)
python tools/mm_democam_smoke_test.py
```

The smoke test wires a
[`DriftingGaussianEmbryo`](../tracking_tools/microscope_interface/synthetic_source.py)
(known ground-truth drift of `(0.0, 5.0, 0.0)` µm/tp) into the
Micro-Manager DemoCamera path, runs the full tracker → `relative_move`
→ `_enqueue_timepoint` loop for a handful of cycles, and asserts the
cumulative drift the tracker reported matches the ground truth to
within one pixel.

Expected success tail:

```text
[MicroscopeInterface_Micromanager] MDA started — 3 scene(s), interval=0.00s, z_stack=OFF, synthetic_source=ON
[MicroscopeInterface_Micromanager] Enqueued tp=0 for 3 scene(s); drift snapshot: scene_0=(+0.00,+0.00,+0.00), scene_1=(+0.00,+0.00,+0.00), scene_2=(+0.00,+0.00,+0.00)
[MicroscopeInterface_Micromanager] [scene_0] tp=0 frame queued -> tracker (saved t0000_Brightfield.tif)
[MicroscopeInterface_Micromanager] [scene_1] tp=0 frame queued -> tracker (saved t0000_Brightfield.tif)
[MicroscopeInterface_Micromanager] [scene_2] tp=0 frame queued -> tracker (saved t0000_Brightfield.tif)
[MicroscopeInterface_Micromanager] [scene_0] cum_drift updated: shift=(+0.00,+5.02,+0.00) µm  cumulative=(+0.00,+5.02,+0.00) µm
[MicroscopeInterface_Micromanager] [scene_1] cum_drift updated: shift=(+0.00,+4.98,+0.00) µm  cumulative=(+0.00,+4.98,+0.00) µm
[MicroscopeInterface_Micromanager] [scene_2] cum_drift updated: shift=(+0.00,+5.01,+0.00) µm  cumulative=(+0.00,+5.01,+0.00) µm
[MicroscopeInterface_Micromanager] Enqueued tp=1 for 3 scene(s); drift snapshot: scene_0=(+0.00,+5.02,+0.00), scene_1=(+0.00,+4.98,+0.00), scene_2=(+0.00,+5.01,+0.00)
...
[mm_democam_smoke_test] PASS  scene_0 cumulative=(+0.00,+45.02,+0.00) µm  expected=(+0.00,+45.00,+0.00) µm  err<0.35 µm
[mm_democam_smoke_test] PASS  scene_1 cumulative=(+0.00,+44.98,+0.00) µm  expected=(+0.00,+45.00,+0.00) µm  err<0.35 µm
[mm_democam_smoke_test] PASS  scene_2 cumulative=(+0.00,+45.01,+0.00) µm  expected=(+0.00,+45.00,+0.00) µm  err<0.35 µm
[mm_democam_smoke_test] SMOKE TEST PASSED
```

If the last line reads `SMOKE TEST PASSED`, your dev machine can drive
the full closed loop. You are ready to work on the tracker, the panel,
or the ROI plumbing without ever touching a real microscope.

---

## 4. How it works

### 4.1 CMMCorePlus + useq-schema, queue-driven MDA

`connect()` loads the MMCore system config (defaulting to the built-in
demo config if none is supplied), wires the
`mmc.mda.events.frameReady` signal to our handler, and starts
`mmc.run_mda(iterator)` where the iterator drains an internal
`_mda_queue`. We seed the queue with one `useq.MDAEvent` per scene for
`tp=0` and then, inside `frameReady`, **when the last scene of a cycle
emits its last z-slice we unconditionally enqueue the next timepoint's
events** — computed from `baseline_um + cum_drift`. That is why
tracker-computed corrections always land on the next cycle: the events
are built at enqueue time, not at MDA-start time.

Three invariants make this robust (see the block comment above
`class MicroscopeInterface_Micromanager` in the source):

- **Bug 1 prevention**: cycle advance is unconditional (never gated on
  drift), so a zero-drift run still completes.
- **Bug 2 prevention**: `relative_move` only mutates `_cum_drift` under
  a lock; the stage is never poked directly. The MDA engine drives the
  stage from `x_pos`/`y_pos`/`z_pos` on each event.
- **Bug 3 prevention**: baselines come from
  `positions_config[pos_name]["xyz_um"]`, never from
  `mmc.getXYPosition` (which would silently pick up whatever the last
  move landed on).

### 4.2 The `synthetic_source` hook

`mm_params["synthetic_source"]` is any callable with signature
`(x_um, y_um, z_um, pos_name) -> np.ndarray | None`. When set, the
`frameReady` handler **replaces the DemoCamera image with the returned
array** before saving + enqueueing. This is what lets Phase A ship
without hardware: the MDA engine still executes, the stage
coordinates still update event-to-event, the frame files are still
written to disk under the same naming convention as LS1 — only the
pixels come from `DriftingGaussianEmbryo` or `ReplayFromFolder`
instead of a camera.

### 4.3 Frame-file convention (shared with LS1)

Frames are written as `t{tp:04d}_{channel}.tif` under
`{dirpath}/{pos_name}/`, matching the convention already used by
`MicroscopeInterface_LS1` and `MicroscopeInterface_Files`. This means
existing dashboards, offline replay scripts, and ROI-selection tools
work unchanged.

### 4.4 Runner contract compatibility

The class implements the same public surface as every other backend:

```
wait_for_image(timeout_ms) -> (image, tp, pos_name) or (None, None, None)
relative_move(pos_name, dx, dy, dz)
connect(); disconnect(); stop()
refresh_filename(pos_name)
pause_after_position(); no_pause_after_position(); continue_from_pause()
```

Drop-in replacement — `TrackingRunner` needs no changes.

---

## 5. Supported microscopes

| Microscope | Camera | Stage | Focus | Illumination | Status | Notes |
|---|---|---|---|---|---|---|
| **Micro-Manager DemoCamera** | DemoCamera | DemoStage | DemoFocus | DemoShutter | **Validated (Phase A)** | Ships with pymmcore-plus. Used by `tools/mm_democam_smoke_test.py`. No hardware required. |
| **Zeiss Axio Observer 7** | Photometrics Prime 95B | Marzhauser Tango | Zeiss Definite Focus 2 | Zeiss Colibri 5 / 7 | **Expected (Phase B)** | On-site rig at EPFL LiveScope. Bring-up scheduled once the DemoCamera path is signed off on. Uses the standard MM device adapters — no custom code expected. |
| _Add yours_ | _model_ | _model_ | _model_ | _model_ | _pending_ | Open a PR against this table with your `mm_params` snippet and any device-specific gotchas. |

---

## 6. Configuration on real hardware

### 6.1 Build the MM system config with the Hardware Configuration Wizard

Micro-Manager itself ships the wizard that produces the `.cfg` file we
pass to `cfg_path`. Follow the Micro-Manager docs at
<https://micro-manager.org/Micro-Manager_Configuration_Guide> to click
through camera, stage, focus, shutter, and light-path device adapters
against your specific hardware. Save the result somewhere stable —
e.g. `C:\ProgramData\lightsheet-live-tracking\MMConfig_AxioObserver7.cfg`.

Sanity-check the config outside Python by opening it in the
Micro-Manager GUI once — if the GUI can snap an image and move the
stage, our backend can too.

### 6.2 `mm_params` dict — Axio Observer 7 example

```python
mm_params = {
    # Path to the config produced by the MM Hardware Configuration Wizard.
    # Leave empty ('') to load the built-in demo config — DemoCamera path.
    'cfg_path': r'C:\ProgramData\lightsheet-live-tracking\MMConfig_AxioObserver7.cfg',

    # MM channel-group and preset names must match those defined in the .cfg.
    # For a Colibri 5/7 you typically have presets like 'DAPI', 'GFP', 'RFP',
    # 'Brightfield' grouped under 'Channel'.
    'channel_group':  'Channel',
    'channel_preset': 'GFP',

    'exposure_ms': 100.0,

    # Optional per-scene z-stack. If present, we shoot a symmetric stack around
    # the tracker-computed z at each timepoint. The tracker still sees a single
    # 2-D slice (the middle one); all slices are saved as t{NNNN}_{ch}_z{NNN}.tif.
    'z_stack': {
        'range_um': 20.0,
        'step_um':   1.0,
    },

    # Wall-clock spacing between cycles. 0.0 = as-fast-as-possible.
    'interval_s': 30.0,

    # Soft clamps applied inside relative_move before the shift lands in
    # _cum_drift. Prevents a runaway tracker from crashing the stage.
    'max_xy_um': 500.0,
    'max_z_um':  100.0,

    # Real-hardware runs: leave synthetic_source unset. Set only for offline
    # smoke tests / regression harnesses.
    'synthetic_source': None,

    # Optional guardrail — stop the MDA after N timepoints (used by smoke tests
    # to bound runtime).  Leave None for open-ended live acquisitions.
    'stop_after_tp': None,
}
```

The per-scene baselines come from `positions_config[pos_name]["xyz_um"]`,
not from `mm_params` — mirroring how the ZEN panel's *Initial scene
positions* text area was already wired up. See
[Zeiss panel walkthrough §2](zeiss_panel_walkthrough.md) for the same
concept.

---

## 7. Troubleshooting

### `.cfg` fails to load with a device-interface-version error

MM device adapters and MMCore have to be built against the same
device-interface version; a mismatch (usually after upgrading
pymmcore-plus or installing MMCore separately) makes
`loadSystemConfiguration` throw. Check the version at runtime:

```python
from pymmcore_plus import CMMCorePlus
mmc = CMMCorePlus.instance()
print(mmc.getAPIVersionInfo())   # device-interface version
print(mmc.getVersionInfo())      # MMCore build
```

`connect()` already logs both lines on startup so grep the runtime log
first. Fix by re-running `mmcore install` (aligns the two) or
downgrading pymmcore-plus to a build compatible with the device
adapters your Wizard-produced `.cfg` references.

### Smoke test hangs

`tools/mm_democam_smoke_test.py` should **never** hang — that would
regress Bug 1 (cycle advance must be unconditional). If it does, treat
it as a bug: capture the log and file an issue citing the zero-drift
regression test in the smoke test (the `expected_drift=(0, 0, 0)` case
must complete cleanly).

### `waitForDevice` hangs on real hardware

We wrap all device waits with `_wait_for_device` (bounded polling of
`mmc.deviceBusy` with a 5 s default timeout), so the *symptom* on our
side is a `TimeoutError` from that helper rather than a permanent
hang. The *cause* is almost always a stuck stage: an obstruction, a
disabled axis in the Zeiss stage controller, or a Definite Focus that
lost its reference. Diagnose from the MM GUI — if the GUI's stage
panel can't move either, the issue is upstream of us.

---

## 8. Roadmap

- **Phase A — DONE**: DemoCamera path, `MicroscopeInterface_Micromanager`
  class, `synthetic_source` hook, smoke test with zero-drift and
  known-drift regression cases.
- **Phase B — in progress**: Axio Observer 7 hardware bring-up at the
  EPFL LiveScope rig. Deliverables: a validated
  `MMConfig_AxioObserver7.cfg`, an updated row in the supported-scope
  table, and a walkthrough analogous to
  [`zeiss_panel_walkthrough.md`](zeiss_panel_walkthrough.md).
- **Phase C — deferred**: unified backend selector in the panel
  (LS1 / Files / Zeiss / Micro-Manager) so a user picks the backend
  from a dropdown instead of editing `zeiss_config.ini`. Tracked
  alongside items **F**/**G**/**H** in
  [`work_queue.md`](work_queue.md).

---

## What lives in which file

| File | Role |
|---|---|
| [`tracking_tools/microscope_interface/MicroscopeInterface.py`](../tracking_tools/microscope_interface/MicroscopeInterface.py) — `MicroscopeInterface_Micromanager` | The backend: MDA queue, `frameReady` handler, `relative_move` → `_cum_drift`. |
| [`tracking_tools/microscope_interface/synthetic_source.py`](../tracking_tools/microscope_interface/synthetic_source.py) | `DriftingGaussianEmbryo` + `ReplayFromFolder` — the hardware-free image sources. |
| `docs/mm_demo_config.cfg` | Minimal MM system config used by the smoke test (DemoCamera + DemoStage + DemoShutter). Also serves as a template for real-hardware `.cfg` files. |
| `tools/mm_democam_smoke_test.py` | End-to-end regression harness. Zero-drift + known-drift cases; PASS/FAIL summary at the bottom. |
| [`docs/work_queue.md`](work_queue.md) | Deferred items — Phase C panel unification tracked here. |
