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

## Phase 2 — Micro-Manager GUI hardware configuration (~30–60 min)

### Goal
Get the MM Java GUI talking to every real device on the microscope,
saved as a `.cfg` file we'll later hand to `pymmcore-plus`.

**Every device must be independently controllable from the MM GUI
before we go anywhere near Python.** MM's GUI is easier to debug than
pymmcore-plus for first-time bring-up, AND it is the reference
implementation that Zeiss + Photometrics QA their adapters against.

### Phase 2A — Prerequisite check (5 min)

Do these five checks in order — do not launch the wizard until every
one passes.

**1. Establish the DIV (device interface version) target.**

The single most common Windows failure is a DIV mismatch between the
adapter tree `mmcore install` populated
(`%LOCALAPPDATA%\pymmcore-plus\mm\`) and the MMStudio Java install
(`C:\Program Files\Micro-Manager-2.0\`). Both trees must have the same
DIV, otherwise a `.cfg` saved by MMStudio silently fails when
pymmcore-plus loads it.

```bat
python -c "import pymmcore; print(pymmcore.__version__)"
python -c "from pymmcore_plus import CMMCorePlus; c=CMMCorePlus(); print(c.getAPIVersionInfo())"
```

Write down the **4th number** of the `pymmcore.__version__` output
(e.g. `12.5.0.75.0` → DIV = **75**). Cross-check with the
`getAPIVersionInfo()` line — it prints `Device API version N, Module
API version M`; the N must match.

**2. Confirm PVCAM runtime is installed and the camera enumerates.**

```bat
where pvcam64.dll
```

Must return `C:\Windows\System32\pvcam64.dll`. If not, install **PVCAM
for Windows** (currently 3.10.2.7) from
<https://www.teledynevisionsolutions.com/support/software-and-drivers>
— the "PVCAM for Windows" runtime installer, NOT the "SDK-only"
package. Reboot. Then in Device Manager confirm the Prime 95B appears
without a yellow bang.

Note: on PVCAM 3.9+ the ini file lives at
`C:\ProgramData\Photometrics\PVCAM\pvcam.ini`, not `C:\Windows\`.

**3. Confirm Marzhäuser controller enumerates as a COM port.**

Device Manager → Ports (COM & LPT). Power the Marzhäuser controller
on with its cable connected. Expect a COM port labeled *Marzhauser*,
*USB Serial Port*, or *FTDI*. Note the number (e.g. `COM4`).

**4. Confirm the CAN29 path to the Axio Observer 7.**

Don't trace cables — probe from software. ZeissCAN29 never
auto-detects, but a successful `initializeDevice` on the right port
*is* the detection (the adapter then reads real labels back off the
stand). `tools/probe_serial_devices.py` brute-forces the (port, baud)
grid for you:

```bat
:: 1. What ports and device libraries does MM actually see?
python tools\probe_serial_devices.py --list

:: 2. Probe every visible port for the Zeiss stand
python tools\probe_serial_devices.py

:: 3. Or target one port/baud directly
python tools\probe_serial_devices.py --ports COM1 --bauds 57600 -v
```

A hit prints the winning port + baud and dumps every property the hub
reports — objective labels, reflector positions, firmware strings.
That readback is proof you're talking to the real stand.

The same script probes other serial devices:

```bat
python tools\probe_serial_devices.py --library Marzhauser       --device XYStage
python tools\probe_serial_devices.py --library MarzhauserLStep  --device XYStage
```

If `--list` shows **no** `ZeissCAN29` library, your `mmcore install`
tree is minimal — install the MMStudio Java nightly (Phase 2B) and
re-run with
`--adapter-path "C:\Program Files\Micro-Manager-2.0"`.

Physical fallback, only if the probe finds nothing: the Observer 7
usually has a **DB9 RS-232** connector labeled `CAN` on the back
panel. If your stand exposes CAN29 over USB instead, Zeiss's CAN29-USB
driver (bundled with ZEN) provides the virtual COM port.

**5. Close ZEN completely.** This is more than just closing the
window:

- Task Manager → End task on: `zen.exe`, `zenblue.exe`,
  `ZenApiGateway.exe`, `MTB2.exe`, `ZeissLightManager.exe` if present.
- `services.msc` → find any service starting with "Zeiss" or "Carl
  Zeiss" (e.g. *Carl Zeiss MTB Service*, *ZEN API Gateway*). Set to
  **Manual**, then **Stop**.
- On the stand's touchscreen: turn **OFF** *Light Manager* and
  *Dazzle Protect*. Both fight the CAN29 adapter (documented on the
  ZeissCAN29 wiki).
- Do NOT uninstall ZEN — its CAN29-USB driver may be your only bus if
  your stand uses USB.

### Phase 2B — Install matching MMStudio Java nightly (10 min)

**1. Consult the DIV history table** at
<https://micro-manager.org/Device_change_log>. Find the date range for
your target DIV from Phase 2A step 1.

**2. Download** from
<https://download.micro-manager.org/nightly/2.0/Windows/>. Pick a
`MMSetup_64bit_2.0.3_YYYYMMDD.exe` whose date falls inside that DIV
window. Prefer the newest such date — PVCAM adapter fixes accumulate.

**3. Install to the default** `C:\Program Files\Micro-Manager-2.0\`.
Intentionally SEPARATE from the `%LOCALAPPDATA%\pymmcore-plus\mm\`
tree that `mmcore install` already populated.

**4. Confirm the two adapter trees agree on DIV.** Launch
`C:\Program Files\Micro-Manager-2.0\ImageJ.exe` → Help → About
Micro-Manager. The "Device Interface version" must equal your DIV
from Phase 2A step 1. If not: uninstall this nightly, pick a
different date within the correct DIV window, reinstall.

Escape hatch if the DIVs are close but not equal (e.g. testing across
a boundary): in your Phase 3 REPL, before `loadSystemConfiguration`,
call
`mmc.setDeviceAdapterSearchPaths([r"C:\Program Files\Micro-Manager-2.0"])`
— this points pymmcore-plus at the Java tree instead of its own.
Prefer matching DIVs when you have time.

### Phase 2C — Ensure Colibri + DF2 are POWERED UP (30 s)

The ZeissCAN29 adapter does NOT auto-detect. It queries the stand's
controller for what is currently talking. If Colibri or DF2 is
powered off, or was powered on AFTER MM opened the port, the wizard
silently omits sub-devices — Colibri 7 shows 2 LEDs instead of 7, DF2
does not appear at all.

Power the stand + Colibri + DF2, wait 30 s for the touchscreen to
finish boot, THEN start the wizard.

### Phase 2D — Configuration Wizard (15 min)

1. `ImageJ.exe` → **View → Micro-Manager Log Panel** (keep it open
   throughout — watch for red errors).
2. **Tools → Hardware Configuration Wizard → Create new
   configuration**.
3. Add devices **IN THIS ORDER** (order matters for CAN29 — the hub
   must be initialized first):

   | # | Library | Sub-device | Label | Notes |
   |---|---|---|---|---|
   | 1 | `ZeissCAN29` | `ZeissScope` | `Scope` | The hub. Port = your CAN29 COM. Baud 57600, timeout 500 ms, 8N1, no handshake, no auto-detect. Initialize BEFORE adding sub-devices. |
   | 2 | `ZeissCAN29` | `ObjectiveTurret` | `Objective` | |
   | 3 | `ZeissCAN29` | `ReflectorTurret` | `Reflector` | |
   | 4 | `ZeissCAN29` | `ZDrive` | `Z` | Main Z focus. |
   | 5 | `ZeissCAN29` | `DefiniteFocus` | `DF2` | Autofocus device. |
   | 6 | `ZeissCAN29` | `ZeissDefiniteFocusOffset` | `DFOffset` | Stage device — enables per-position DF2 offsets in MDA. |
   | 7 | `ZeissCAN29` | `Colibri` | `LED` | Same ZeissCAN29 library — there is NO separate Colibri adapter binary since v1.3.40. Colibri 5 and Colibri 7 both come out here; count of LED sub-channels matches your physical octagon. |
   | 8 | `Marzhauser` **or** `MarzhauserLStep` | XY Stage | `XY` | Try `Marzhauser` (TANGO command set) first — most common ship on Axio Observer 7. If XY device appears but does NOT move in the smoke test, come back and switch to `MarzhauserLStep`. Port = the Marzhäuser COM from Phase 2A step 3. 57600/8/N/1. Also: press the joystick button so its LED goes OFF (joystick has priority; MM commands ignored while it's on). |
   | 9 | `PVCAM` | `Camera-1` | `Camera` | Auto-detects. Just Initialize; no ROI/exposure setup here. |

4. **Set default devices** in the wizard:
   - Core **Camera** = `Camera` (PVCAM)
   - Core **XYStage** = `XY` (Marzhäuser)
   - Core **Focus** = `Z` (Zeiss ZDrive)
   - Core **AutoFocus** = `DF2`
   - Core **Shutter** = the Colibri-provided shutter device

5. **Configure a Channel group.** Tools → Hardware Configuration
   Wizard → *Group / Preset Editor*. Create group `Channel` with at
   least one preset `Brightfield` that sets LED intensity + reflector
   position + exposure.

6. **Set Core-Timeout = 25000 ms.** Tools → Options → Core-Timeout.
   Default 5000 ms is too short for DF2 *Measure*; ignoring this
   causes `waitForDevice` to return before DF2 is finished, and the
   next MDAEvent snaps out of focus.

7. **Save the .cfg** to
   `C:\Users\helsens\software\lightsheet-live-tracking-tool\docs\axio_observer_7.cfg`.

### Phase 2E — Test every device from MMStudio (10 min)

Do all 9 in a single session with the Log Panel open. If any step
throws a red error or the physical hardware doesn't respond, **STOP**
— fix it here, not in Python.

| # | Test | Menu | Expected |
|---|---|---|---|
| 1 | **Snap** camera | Main window Snap button | 16-bit image, histogram not flat/saturated |
| 2 | **Live** mode | Main window Live | ≥5 fps, stops on second click |
| 3 | Change **exposure** to 100 ms, Snap | Main window Exposure box | ~10× brighter frame |
| 4 | **XY nudge** 10 µm | Tools → Stage Control | Physical motion + XY readout updates |
| 5 | **Z nudge** 1 µm | Tools → Stage Control (Z) | Focus knob turns; use a dry 10× objective so you can't crash |
| 6 | **Objective turret** to another position | Tools → Device Property Browser → `Objective-Label` | Turret rotates; label matches |
| 7 | **Colibri LED on** at 20 % + open Shutter | Tools → Device Property Browser → `Colibri-Intensity1` | LED physically lights |
| 8 | **DF2 train + apply** | Property Browser → `DefiniteFocus-FocusMethod` = *Measure*, press stand autofocus button, jog Z ±20 µm, then set *Apply* | DF2 restores focus to trained plane |
| 9 | Apply the **`Brightfield`** channel preset | Main window Channel dropdown | Reflector rotates, correct LED on, exposure sets |

### Expected

All 9 pass with zero red errors in the Log Panel.

### STOP if
- **ZeissScope Initialize** hangs or fails: the CAN29 port is held by
  ZEN/MTB2 or the cable/baud/driver is wrong. Re-run Phase 2A step 5.
  Don't add sub-devices until the hub initializes.
- **PVCAM library lists no camera** in Add Device: `pvcam64.dll` can't
  see the camera. Re-check Phase 2A step 2 — PVCAM install + reboot.
- **Marzhäuser XY appears but doesn't move** (and joystick LED is
  OFF): controller is in LSTEP mode but you loaded the `Marzhauser`
  adapter. Switch to `MarzhauserLStep`. See also: motors disabled
  ("Emergency Off" latched), joystick LED still on, or wrong axis
  mapping.
- **Colibri shows fewer LEDs than physically installed**: Colibri
  wasn't fully booted when the wizard opened the port. Close the
  wizard, power-cycle Colibri, wait 30 s, retry — cited failure mode
  on image.sc.
- **DF2 not in ZeissCAN29 sub-device list**: same power-on ordering
  issue. Reset and start over with everything already booted.
- **Any red ERROR in the Log Panel** during Add Device or a smoke
  test: fix it before continuing. pymmcore-plus will reproduce the
  failure with worse diagnostics.

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
