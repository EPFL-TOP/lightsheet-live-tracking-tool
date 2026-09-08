# Running the tracker on the Axio Observer 7

Operating instructions for the Zeiss Axio Observer 7 at EPFL, plus a
reference for every function the panel exposes.

```bat
cd D:\Users\zeiss\Software\lightsheet-live-tracking-tool
git pull
panel serve interactive_tools\mtb_setup.py --show --port 5023
```

**Close ZEN first.** It holds the camera exclusively, and its MTB
client competes with ours.

---

## How the pieces fit together

| Subsystem | Owner | Why |
|---|---|---|
| Camera (Prime 95B) | Micro-Manager / PVCAM | works, and MM's camera layer is solid |
| Stage, focus, piezo, objective, reflector, sideport, condenser, LEDs | **Zeiss MTB 2011** via `pythonnet` | the Observer 7 carries CAN29 over USB into `CZCanSrv`, which MM's serial `ZeissCAN29` adapter cannot reach |
| Time loop | our own Python | so the stage can move *while* acquiring — the thing ZEN could not do |

MTB is Zeiss's hardware abstraction *service*, not ZEN. ZEN is merely
another MTB client, and it does not need to run.

### Two hard constraints

- **One MTB login per process.** `Logout` destroys the realtime
  subsystem and no later login succeeds. The panel owns the session and
  hands it to the tracker.
- **One PVCAM handle per camera.** The panel opens it and hands it
  over; nothing else may open it.

---

## Session walkthrough

### 1. Power up in order

**Restart the PC, then the camera, then the microscope.** This is the
documented lab procedure and it matters: PVCAM must be enumerated
before its client starts, or ZEN and the panel both fail to find a
camera.

### 2. Connect

Press **Connect hardware**. The status line reports what came up, and
the hardware panel prints every axis with its range, deviation and
arrival tolerance, plus the LED table.

### 3. Optics — get an image before anything else

Order matters here; light travels **shutter → condenser → objective →
reflector → sideport**.

1. **Read light path.** Reading never moves anything.
2. **Sideport → position 2** (`Sideport 100% Left`). The camera is on
   the **left** port. Position 1 sends everything to the eyepieces and
   the camera sees nothing.
3. **Condenser** off a phase ring (`PH1`/`PH2`/`PH3`) for brightfield.
4. **Read objective** → pick the objective → **Rotate to it.** The
   pixel pitch is then derived from the hardware, not typed.
5. **TL light ON**, then **Live**.

If the image is flat at `mean≈99, std≈2.2` regardless of exposure,
**no light is reaching the sensor** — that is the sensor's dark offset
and read noise. A real signal scales with exposure. Check the sideport
first.

### 4. Positions

Jog with the arrow buttons, then **Capture current position** per
embryo. Names are editable; **Go to selected** returns there. Save the
list to JSON so a later session can reload it.

Baselines are stored from these captures and are **never** re-read from
the stage during a run — doing so would make accumulated drift the new
reference.

### 5. Channels

A channel is a **saved light-path recipe**, applied at every position.

| Column | Meaning |
|---|---|
| `name` | goes in the filename: `t0000_<name>.tif` |
| `reflector_pos` | filter cube **position number**; `0` = leave alone |
| `led_nm` | Colibri LED wavelength; `0` = none |
| `led_percent` | LED intensity |
| `tl` | transmitted light on/off |
| `invert` | flip contrast before tracking (see below) |
| `exposure_ms` | per channel |
| `z_stack` | per channel |

The reliable way to build one:

1. **Live** on
2. Optics tab → set cube, **Read LEDs**, pick an LED, slide intensity
3. **TL light OFF** for fluorescence
4. When it looks right → **Capture light path → channel**

Typical two-channel setup:

| name | reflector_pos | led_nm | led_percent | tl | invert | exposure_ms | z_stack |
|---|---|---|---|---|---|---|---|
| BF | 0 | 0 | 0 | ☑ | ☑ | 5 | ☐ |
| GFP | 3 | 475 | 20 | ☐ | ☐ | 200 | ☑ |

Available LEDs: **385 / 430 / 475 / 511 / 567 / 630 nm**.

**Fluorescence needs the transmitted light OFF.** Left on, brightfield
swamps the emission and the "fluo" frame comes back with a bright
background. Selecting a channel's LED also switches every other LED
off, so channels cannot bleed into each other.

Channels are acquired **sequentially** — one camera, one cube.

### 6. Acquisition settings

- **Interval / timepoints** — `0` timepoints means run until stopped
- **Z-stack** — range and step. Keep the step **above ~0.06 µm** for
  the piezo: finer steps do not move and silently duplicate planes.
  Warnings appear in the log at start.
- **Pixel size XY** — set by *Read objective*. Wrong here means every
  correction is mis-scaled.
- **2-D tracking** — on ignores Z drift; leave on until XY is trusted
- **Max XY / Z correction** — clamps per correction, a safety net
  against a wild measurement driving the stage away

### 7. Run

1. Set **Experiment root** to a fresh folder
2. **Start** — acquisition begins immediately, open loop
3. In another terminal, draw ROIs while it runs:
   `panel serve interactive_tools\panel_app.py`
4. Save ROIs for **every** position
5. Within ~5 s the log reads *ROIs found … attaching tracker* and
   correction begins

No restart is needed — the tracker attaches to the running loop, and
re-saving ROIs mid-run re-initialises that position.

```
<experiment root>/
  scene_000/
    t0000_BF.tif
    t0000_GFP.tif
    embryo_tracking/
      tracking_RoIs.json
      logs.json
```

---

## Brightfield tracking and the `invert` flag

The ROI tracker segments by thresholding for a **bright object on a
dark background**. That is how fluorescence looks. Transmitted light is
the opposite: the background is the brightest thing in the frame, so it
survives every threshold and the embryo is never isolated.

`invert` flips the contrast of the frame **handed to the tracker**,
within the data's own range so a frame spanning 11000–14800 keeps its
full contrast rather than collapsing into the bottom of a 0–65535
scale. **The file on disk stays raw**, so the ROI dashboard and later
review see the real image, and ROI geometry is unaffected.

Set `invert` ☑ for brightfield, ☐ for fluorescence.

If brightfield masking is still poor, the remaining knobs live in
`tracking_tools/tracking_config.yaml` under `roi_tracker`:
`blur_factor`, `scaling_factor`, `mask_kernel_size`,
`containment_threshold`, `score_threshold`. Transmitted light gives
lower contrast than fluorescence, so more blur and a gentler threshold
usually help.

---

## Function reference

### `tracking_tools/microscope_interface/mtb.py`

| Object | Purpose |
|---|---|
| `MTBSession.shared()` | **the only** way to get a session — one login per process |
| `MTBSession.close_shared()` | log out, once, at process exit |
| `.axis(role)` | `MTBAxis` for `axis_x`, `axis_y`, `focus`, `piezo` |
| `.changer(role)` | `MTBChangerDevice` for `tl_shutter`, `sideport`, `condenser`, `reflector` |
| `.objective()` | `MTBObjective` — nosepiece, magnification |
| `.colibri()` | `MTBColibri` — the fluorescence LEDs |
| `.light_path()` | every reachable light-path changer, in optical order |
| `MTBAxis.position` / `.move_to()` / `.move_by()` | µm; clamps to limits, skips no-op moves |
| `MTBAxis.arrival_tolerance` | `max(step, typical_deviation)` — a servoing piezo dithers well above its step |
| `MTBChangerDevice.elements()` | all positions with the element in each |
| `MTBChangerDevice.set_position(p)` | 1-based position |
| `MTBChangerDevice.set_open(bool)` | for two-state shutters |
| `MTBObjective.slots()` | every position with objective, magnification, NA |
| `MTBObjective.set_position(p)` | rotate the nosepiece |
| `MTBColibri.select(nm, pct)` | switch one LED on exclusively |
| `MTBColibri.all_off()` | every LED off |
| `sample_pixel_size_um(cam, mag, adapter)` | `11 / (mag × adapter)` for the Prime 95B |

**Indexing:** `Position` is **1-based**; `GetElement(n)` is
**0-based**. The element at position `p` is `GetElement(p-1)`. Verified
against the stand's own display — the sideport reports position 2 as
"Sideport 100% Left" while `GetElement(1)` returns `'100% L'`.

### `tracking_tools/microscope_interface/mtb_backend.py`

`MicroscopeInterface_MTB(positions_config, dirpath, mtb_params)` —
implements the same contract as the LS1 / ZEN / Files / Micro-Manager
backends.

Contract: `connect()` (idempotent), `wait_for_image(timeout_ms)` →
**always a 3-tuple**, `(None, None, None)` on timeout, paired with
`stop_requested`; `relative_move(pos, dx, dy, dz)` records a
correction applied on the next visit; `refresh_filename(pos)`;
`disconnect()`; `stop()`.

Key `mtb_params`: `session`, `mmc`, `camera_label` (shared resources),
`channels`, `apply_channel`, `z_axis`, `interval_s`, `settle_s`,
`max_xy_um`, `max_z_um`, `z_stack`, `stop_after_tp`,
`synthetic_source`.

### Command-line tools

```bat
:: hardware checks
python tools\mtb_smoke_test.py                 :: read-only
python tools\mtb_smoke_test.py --move          :: small test moves
python tools\mtb_smoke_test.py --lightpath     :: shutters, turrets, lamps
python tools\mtb_smoke_test.py --objective     :: nosepiece detail
python tools\mtb_smoke_test.py --colibri       :: LED controller detail

:: camera
python tools\probe_camera.py --all             :: real camera or phantom slot?
python tools\probe_camera.py --sweep           :: exposure / readout sweep

:: end to end
python tools\mtb_imaging_test.py --exposure-scan
python tools\mtb_imaging_test.py --exposure 5 --n-timepoints 3
python tools\mtb_imaging_test.py --test-drift 5   :: proves the closed loop

:: MTB API exploration
python tools\probe_mtb.py --dump-api --members --filter Changer
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Flat frame, `mean≈99 std≈2.2`, unchanged by exposure | No light reaching the sensor. Sideport on the wrong port, condenser on a phase ring, TL shutter closed, or lamp off |
| Frame all `65535` | Saturated. Reduce exposure — 5–20 ms is the working range |
| "Fluo" image with a bright background | Transmitted light still on. **TL light OFF** |
| Nothing looks fluorescent | LED not on, or cube and LED mismatched. A cube change alone gives transmitted light through a filter |
| Brightfield masking fails | `invert` not ticked for that channel |
| `pl_cam_open failed … ALREADY_OPEN` | Something else holds the camera — ZEN, or a second panel process |
| `MTB could not be initialized … OpenRtNet` | A logout already happened in this process. Restart it |
| `rotation to slot 0 was refused` | Stale code — positions are 1-based. `git pull` |
| Tracker says `No module named torch` | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Serverkit `404` on `//services` | Trailing slash in `server_addresses`. Fixed in the config |
| Z-stack planes look identical | Step below the actuator's resolution, or the range exceeds its limits — both are warned about at start |

### Hardware reference

| | |
|---|---|
| Camera | Prime 95B, GS144BSI, 1200×1200, 16-bit, 11 µm pixels |
| Working exposure | 5–20 ms brightfield; fluorescence needs far more |
| Stage | Zeiss SMC2009, ±350 mm, 0.25 µm step, **1.875 µm** repeatability |
| Focus | ±14 mm, 0.01 µm step |
| Piezo | Wienecke & Sinske WSB, 0–500 µm, 0.01 µm step, ~0.055 µm deviation |
| Objectives | position 2 `Fluar 10x/0.50`, position 4 `Plan-Apochromat 40x/0.95` |
| Pixel pitch | 10× → 1.1 µm · 40× → 0.275 µm |
| LEDs | 385 / 430 / 475 / 511 / 567 / 630 nm |
| Sideport | position 2 = **Left = camera** |

---

## Testing

```bat
python -m pytest tests\ -q
```

231 tests, none requiring hardware — the MTB and camera layers are
faked. Regression tests are named for the failure they prevent, and
several encode behaviour the hardware taught us: the 0-based
`GetElement`, the 3-tuple `wait_for_image`, the shared session and
camera, no-op moves, servoing tolerance, duplicate z-planes.
