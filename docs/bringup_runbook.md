# Bring-up runbook — Axio Observer 7 on the Zeiss machine

Step-by-step deployment of the lightsheet-live-tracking-tool with the
Micro-Manager backend on the Windows PC attached to your Axio Observer
7. Each phase has a **goal**, **steps**, **expected output**, and a
hard **STOP** gate so a failure at one phase doesn't cascade into
confusion at the next.

Read the whole file once before running the first command.

---

## Prerequisites (verify before starting)

- Windows 10/11 with Python 3.11+ installed (miniconda or system Python).
- `git` available on the command line.
- Local Administrator rights (needed for `mmcore install` to write DLL
  paths under `C:\Program Files\Micro-Manager-*`).
- Micro-Manager Java GUI 2.0 downloaded from
  <https://micro-manager.org/Micro-Manager_Nightly_Builds>. **Match
  device-interface version 75** (the pymmcore-plus 0.18.x default) —
  wrong nightly = silent config failure. Check the wiki table.
- Photometrics **PVCAM SDK** installed (needed by MM's PVCAM device
  adapter to talk to the Prime 95B). Download from Photometrics.
- ZEN can still be installed on the same machine but should be
  **closed** during Phase 3+ (the Prime 95B framegrabber cannot be
  shared with ZEN).
- Hardware powered on and initialised via ZEN at least once (ensures
  the stage controller is homed, Definite Focus sensor is aligned).

Not needed: the ZEN gateway. The MM backend does not talk to ZEN.

Also not needed for bring-up: **no Qt / napari GUI**. Do NOT
`pip install pyqt6 pyside6 napari pymmcore-widgets pymmcore-gui` at
this stage. The Java MMStudio + Python REPL flow below covers the
whole hardware checkout, and adding Qt now imports pain (PyQt5-vs-6
collisions, PySide6 wheels breaking behind corporate SSL, etc.) for
zero bring-up benefit. Once Phase 5 is stable we can decide whether
to compose a `pymmcore-widgets` tab into the Panel app; see
`docs/work_queue.md`.

---

## Phase 0 — Deploy the code (~5 min)

### Goal
Get the repo on the Windows PC and confirm the pure-Python parts run.

### Steps

```bat
cd C:\Users\helsens\software
git pull                           :: assuming the repo is already cloned there
:: if not:
:: git clone <your-fork-or-remote>

conda activate microscope-tracking  :: or whatever env you use
pip install -r requirements.txt     :: pytest lives here — needed for Phase 0
pip install -r requirements-mm.txt  :: pymmcore-plus[cli], useq-schema, scipy

python -m pytest tests/ -v
```

### Expected

```
18 passed in 0.5s
```

(5 in `tests/test_contract_stubs.py` + 13 in `tests/test_mm_interface.py`.)

### STOP if
- Tests fail. Do not proceed — the code doesn't even run in isolation
  from any hardware. Paste the pytest output.

---

## Phase 1 — DemoCamera closed-loop (~5 min, no hardware)

### Goal
Prove `pymmcore-plus` + our `MicroscopeInterface_Micromanager` +
synthetic drifting embryo work end-to-end on this Windows machine.
This is the same test that passes on my Mac; if it fails on Windows we
know it's a per-machine problem, not our code.

### Steps

```bat
mmcore install                            :: first time only, ~2 min
                                          :: downloads DemoCamera etc.

python tools\mm_democam_smoke_test.py
```

### Expected

```
[mm_democam_smoke_test] SMOKE TEST PASSED
```

Plus per-scene PASS lines showing the tracker converged on the
synthetic drifting embryo in scene_0/1/2.

### STOP if
- `mmcore install` fails: usually a missing MSVC runtime or firewall.
- Smoke test hangs (>60 s without output): a regression. File a bug.
- Smoke test fails with `RuntimeError: No device with label ""`: the
  demo cfg is broken — check `docs/mm_demo_config.cfg` has a Shutter
  device declared.

---

## Phase 2 — Micro-Manager GUI hardware configuration (~30-60 min)

### Goal
Get the MM Java GUI talking to every real device on the microscope,
saved as a `.cfg` file we'll later hand to `pymmcore-plus`.

**Every device must be independently controllable from the MM GUI
before we go anywhere near Python.** MM's GUI is easier to debug than
pymmcore-plus for first-time bring-up.

### Prerequisite check

- MM Java GUI 2.0 installed at `C:\Program Files\Micro-Manager-2.0`.
- PVCAM SDK installed (verify `pvcam32.dll` or `pvcam64.dll` exists
  under `C:\Program Files\Photometrics\PVCAM\`).
- Close ZEN (it holds an exclusive lock on the Prime 95B).

### Steps

1. Launch **Micro-Manager 2.0 GUI**.
2. **Tools → Hardware Configuration Wizard → Create new configuration**.
3. Add devices one at a time:

   | Adapter library | Device name | Label (recommendation) |
   |---|---|---|
   | `ZeissCAN29` | `ZeissScope` | `Scope` |
   | `ZeissCAN29` | `Objectives` | `Objective` |
   | `ZeissCAN29` | `ReflectorTurret` | `Reflector` |
   | `ZeissCAN29` | `Colibri` (or `Colibri7`) | `LED` |
   | `ZeissCAN29` | `DefiniteFocus` | `DF2` |
   | `ZeissCAN29` | `ZStage` (or the DF2-managed one) | `Z` |
   | `PVCAM` | (Prime 95B will auto-detect) | `Camera` |
   | `Marzhauser` (or `MarzhauserLStep` — try both) | XY | `XY` |

4. **Set default devices** (in the wizard):
   - Core Camera = `Camera`
   - Core XY stage = `XY`
   - Core Focus = `Z`
   - Core AutoFocus = `DF2` (if the DF2 adapter exposes one)
   - Core Shutter = whichever the Colibri adapter provides, or add
     `DemoCamera`/`DShutter` as a fallback

5. **Configure a Channel group** with at least one preset (e.g.
   `Channel` group → `Brightfield` preset that sets the LED
   intensity + reflector position + exposure to reasonable values).

6. **Save the config** to
   `C:\Users\helsens\software\lightsheet-live-tracking-tool\docs\axio_observer_7.cfg`.

7. **Test each device from the MM GUI** in a single session:

   | Device | Test | Expected |
   |---|---|---|
   | Camera | *Snap* button | Image appears in the MM viewer |
   | XY stage | Type `100` in X, hit *Move* | Stage moves ~100 µm (visible in stand readout) |
   | Focus | *Move up 10 µm* | Z drive moves, sample refocuses |
   | Colibri | Turn LED on at 20 % | Actual light on the sample |
   | DF2 | *Full Focus* button | Definite Focus locks a surface |
   | Channel `Brightfield` preset | *Apply* | Correct filter cube + LED intensity |

### Expected

Every row above works from the MM GUI. The `.cfg` file exists at the
target path.

### STOP if
- Any device doesn't enumerate: the adapter/driver is missing.
  ZeissCAN29 needs the ZEISS CAN29 serial cable connected; PVCAM
  needs the SDK; Marzhauser needs the vendor's virtual COM port
  driver.
- Wizard fails on `ZeissCAN29`: the CAN29 serial port might be held by
  ZEN. Close ZEN completely (check Task Manager for `Zen.exe`).
- Camera snaps but returns a black image: the light path might not be
  set to the right port. Try setting the reflector / sideport
  manually.

Do not proceed to Phase 3 until every row above passes.

---

## Phase 3 — Same hardware, but from Python (~15 min)

### Goal
Confirm `pymmcore-plus` can drive the same `.cfg` you just saved.

### Steps

Open a Python REPL in the repo root:

```bat
cd C:\Users\helsens\software\lightsheet-live-tracking-tool
python
```

```python
from pymmcore_plus import CMMCorePlus
mmc = CMMCorePlus()
mmc.loadSystemConfiguration('docs/axio_observer_7.cfg')

# 1. Enumerate
print('Camera:', mmc.getCameraDevice())
print('XY:',     mmc.getXYStageDevice())
print('Focus:',  mmc.getFocusDevice())
print('AF:',     mmc.getAutoFocusDevice())
print('Image size:', mmc.getImageWidth(), 'x', mmc.getImageHeight())

# 2. Snap
img = mmc.snap()
print('img shape:', img.shape, 'dtype:', img.dtype, 'mean:', img.mean())

# 3. Stage read/write
x, y = mmc.getXYPosition()
print(f'XY: ({x}, {y})')
mmc.setXYPosition(x + 100.0, y)      # move 100 µm in X
mmc.waitForDevice(mmc.getXYStageDevice())
print(f'XY after move: {mmc.getXYPosition()}')

# 4. Focus read/write
z = mmc.getZPosition()
print(f'Z: {z}')
mmc.setZPosition(z + 5.0)
mmc.waitForDevice(mmc.getFocusDevice())
print(f'Z after move: {mmc.getZPosition()}')

# 5. Colibri via channel preset — same call the backend uses at
#    tracking_tools/microscope_interface/MicroscopeInterface.py:2171
mmc.setConfig('Channel', 'Brightfield')
mmc.waitForConfig('Channel', 'Brightfield')
img = mmc.snap()
print(f'Brightfield frame mean: {float(img.mean()):.1f}')
# If a preset is misbehaving, poke a specific property directly (property
# names come from MMStudio's Device/Property Browser — do not guess):
#   mmc.setProperty('LED', 'Intensity', 20)

# 6. Return to start
mmc.setXYPosition(x, y)
mmc.setZPosition(z)
```

For a repeatable scripted version of the whole checkout — snap, XY ±100
µm, Z ±5 µm, each channel preset — run:

```bat
python tools\hw_smoke_test.py --cfg docs\axio_observer_7.cfg
```

It exits with a non-zero code on any hardware failure, so you can wire
it into a smoke-test habit.

### Expected
- Each `print` returns real numbers, not zeros.
- `img.shape` matches your Prime 95B ROI (2048×2048 for full frame, or
  whatever binning you set in the .cfg).
- `img.mean()` is a plausible intensity for your sample / illumination.
- Stage/focus visibly move on the microscope stand and reported
  coordinates match the commanded values within a few µm.
- Brightfield preset apply lights up the sample; frame mean rises above
  the dark-frame baseline.

### STOP if
- `loadSystemConfiguration` raises: device-interface-version mismatch.
  Run `python -c "from pymmcore_plus import CMMCorePlus; print(CMMCorePlus().getAPIVersionInfo())"` and compare to your MM
  nightly's version. Reinstall a matching nightly.
- `snap()` returns all-zero image: same light-path issue as Phase 2.
- Stage doesn't move but reports success: wrong Marzhauser adapter.
  Rebuild `.cfg` with a different one from the drop-down.

---

## Phase 4 — Capture positions (~10 min)

### Goal
Record the (x, y, z) of every position you want to track, in the same
coordinate system pymmcore-plus reads.

### Steps

For each embryo / scene of interest:

1. In the MM GUI (or via the Python REPL from Phase 3), **move the
   stage to the position and refocus**.
2. Read out and save:

   ```python
   x, y = mmc.getXYPosition()
   z = mmc.getZPosition()
   print(f'{x}, {y}, {z}')
   ```

3. Copy the printed line into a scratch file. You'll paste all the
   lines into the panel's *Initial scene positions* text area in
   Phase 5.

Example after 3 positions:

```
125.4, 340.2, -8.1
612.9, 340.4, -8.5
1052.1, 341.0, -7.9
```

### Expected
- 1 line per position, `x, y, z` in µm.
- All Z values reasonable relative to each other (small differences —
  DF2 will handle fine tuning at runtime).

### STOP if
- All positions read the same numbers: `getXYPosition()` isn't updating.
  Try reconnecting the stage in the MM GUI.

---

## Phase 5 — First panel run on real hardware, **no drift** (~30 min)

### Goal
Full software → hardware integration. Snap images, save to disk,
compute (noise-level) shifts, apply them to the stage. Do this on a
**still sample** — a fixed slide, or an embryo you're okay stressing
briefly — so any real shift you see is instrument drift, not
biological motion.

### Steps

1. **Close ZEN.**
2. Launch the panel:
   ```bat
   cd C:\Users\helsens\software\lightsheet-live-tracking-tool
   panel serve interactive_tools\zeiss_panel_app.py --show --port 5022
   ```
   Browser opens at `http://localhost:5022/zeiss_panel_app`.

3. **In the Tracking tab**:
   - *Microscope backend* → **Micro-Manager** (default).
   - *Experiment root* → a fresh folder, e.g.
     `C:\Users\helsens\Pictures\mm_test_01`.
   - *Number of scenes* → 1 (start simple).
   - *Cfg path* → change from `docs/mm_demo_config.cfg` to
     `docs/axio_observer_7.cfg`.
   - *Channel group / preset* → `Channel` / `Brightfield` (or whatever
     you defined in Phase 2).
   - *Exposure* → 100 ms.
   - *Z-stack* → **OFF**.
   - *Interval* → 30 s.
   - *Synthetic source* → **OFF** (real camera).
   - *Initial scene positions* → paste one line from Phase 4:
     ```
     125.4, 340.2, -8.1
     ```

4. **Save an ROI in the ROI Selection tab**:
   - Wait for the first frame to appear in
     `mm_test_01\scene_000\t0000_Brightfield.tif` — this happens after
     you press Run below. So:

5. Back in Tracking, click **Run Tracking**.
6. Status pane should turn to `▶ Tracking started (backend=micro-manager)`.
7. First frame lands on disk within a few seconds.
8. Switch to ROI Selection, load `t0000_Brightfield.tif`, draw an ROI,
   save. Tracking will pick it up on the next cycle.

### Expected

Watching the log for the first 3–4 cycles:
- `MDA started` line from the MM backend.
- One `frame queued -> tracker` per timepoint per scene.
- `cum_drift updated` with **small** values (a few pixels of noise-level
  jitter, not tens of µm).
- Stage moves by those small amounts between cycles (visible on the
  stand's XY position readout).

Files on disk:
- `mm_test_01\scene_000\t0000_Brightfield.tif`, `t0001_…`, `t0002_…`
- `mm_test_01\scene_000\embryo_tracking\tracking_RoIs.json` (from your
  ROI save)
- `mm_test_01\scene_000\embryo_tracking\logs.json` (per-frame shifts)

Click **Stop** cleanly. Status pane returns to `_Ready._`.

### STOP if
- Panel refuses to start with an ⚠️ message: read it and fix the flagged
  input (missing dirpath, missing baseline, cfg path invalid, etc.).
- MDA runs but no frames appear on disk: filesystem permissions on
  `mm_test_01`.
- Stage moves erratically (large XY jumps, or Z falling far): the
  tracker is being fed a saturated / all-black image. Check camera
  exposure + LED intensity.

---

## Phase 6 — First real biological run

Once Phase 5 works cleanly:

- Multiple scenes: paste all Phase-4 lines into *Initial scene
  positions*, set *Number of scenes* to match, save an ROI in each
  `scene_NNN\` folder.
- Turn Z-stack on if your imaging needs it.
- Run for the target duration.
- Inspect the Visualisation tab periodically — it reads the same
  `t*_Brightfield.tif` and `logs.json` files that were fine in
  Phase 5, so it should just work.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `mmcore install` fails with SSL error | Corporate firewall inspection | Set `HTTPS_PROXY` env var or run once from a network with direct outbound |
| `RuntimeError: MMCore version mismatch` | MM Java nightly and pymmcore-plus disagree on device-interface version | Reinstall a nightly whose device interface version matches `mmc.getAPIVersionInfo()` (75 for 0.18.x) |
| `Property 'Config' does not exist for device 'Camera'` | You named the Channel group something the code doesn't expect | Either rename it to `Channel` in your .cfg or override `channel_group` in the panel widget |
| Stage moves in MM GUI but not from Python | The .cfg file open in the GUI is different from the one Python loaded | Save-then-reopen; then close the GUI before running Python |
| Camera returns all-zero images | Light path wrong OR ZEN still has the framegrabber | Close ZEN, check reflector turret is on your imaging port |
| `waitForDevice` hangs | Stuck stage or misconfigured DF2 | Homing + re-initialise in MM GUI; if this fixes it, add a longer `_wait_for_device` timeout in `mm_params` |
| Panel shows ⚠️ "MM cfg path does not exist" | You started `panel serve` from a directory other than the repo root, so the relative path can't resolve | Either `cd` to the repo root before `panel serve`, or put an absolute Windows path in the cfg field |
| `Applied TilesService position update` line missing | You're on the MM backend — that log line is ZEN-only. MM applies drift via `MDAEvent` x/y/z inline. Look for `cum_drift updated:` instead. |

---

## Sanity checklist

Before every session:
- [ ] ZEN is **closed** (or not installed).
- [ ] Repo is up to date (`git pull`).
- [ ] MM cfg path in the panel points at the file you actually built.
- [ ] Baselines in *Initial scene positions* were captured today
      (Definite Focus and thermal drift can shift things overnight).
- [ ] Experiment root is a **new** folder (no filename collisions with
      previous runs).

After every session:
- [ ] Click **Stop** in the panel before closing the browser tab.
- [ ] Check `logs.json` for the last few timepoints — clean run has
      `shift_um` values within a few µm.

---

## Where things live

| Path | What |
|---|---|
| `interactive_tools/zeiss_panel_app.py` | The panel (backend dropdown, widgets, routing) |
| `tracking_tools/microscope_interface/MicroscopeInterface.py` | All four `MicroscopeInterface_*` classes |
| `tracking_tools/microscope_interface/synthetic_source.py` | `DriftingGaussianEmbryo`, `ReplayFromFolder` |
| `docs/mm_demo_config.cfg` | DemoCamera cfg (Phase 1) |
| `docs/axio_observer_7.cfg` | Real hardware cfg (you build in Phase 2) |
| `tools/mm_democam_smoke_test.py` | Phase 1 smoke test |
| `tests/` | 18 unit tests (pytest) |
| `docs/micromanager_backend.md` | User-facing MM backend overview |
| `docs/bringup_runbook.md` | This document |
| `docs/work_queue.md` | Deferred improvements |

Ping me at any STOP gate with the actual error message and we'll unstick it before it compounds.
