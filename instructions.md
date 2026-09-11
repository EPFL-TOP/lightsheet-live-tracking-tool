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

The dashboard **connects by itself** once the page loads, and reads
back the objective, light path and LEDs at the same time — each of
those used to need its own button press, and a forgotten objective
leaves the pixel size at its default, which mis-scales every drift
correction. *Connect hardware* is still there if the stand was not
ready. Set `MTB_AUTOCONNECT=0` to disable it, e.g. when someone else
has the microscope.


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
- **Z-stack** — **planes × step**. Keep the step **above ~0.06 µm** for
  the piezo: finer steps do not move and silently duplicate planes.
  Warnings appear in the log at start. A stack belongs to a
  **position**: it centres on that position's stored Z, not on wherever
  the focus happens to be, so touching the focus never moves it. Use
  *Stack bottom / centre / top* to check the whole range is in focus,
  and *Set position Z = current focus* to re-anchor.
- **Settle per plane** — sized for the **piezo**, not the stage.
  *Settle after move* waits for a heavy XY stage to stop ringing
  between positions; a 1 µm piezo step settles in milliseconds. These
  were once the same number, which made a 100-plane stack sleep for
  30 s. Default 0.02 s.
- **Definite Focus 2** — see below

> **A z-stack needs BOTH switches.** *Acquire Z-stack* here sets the
> plane count and step; the **`z_stack` column in the Channels tab**
> decides which channels actually get stacked. That is deliberate —
> brightfield single-plane with a fluorescence stack is a normal
> combination — but with every channel unticked you get single planes
> and the setting looks ignored. The run is now refused with an
> explanation rather than quietly producing flat frames.
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

One file per **position, timepoint and channel**. A z-stack is written
into that same file as a real ImageJ stack — `axes=ZYX`, one page per
plane, with the z-step as `spacing` and the XY pixel size in the
resolution tags, so ImageJ and Fiji open it as a calibrated stack.

This was previously written as a bare array, which tifffile turned
into a single RGB page whenever the stack had **exactly 3 planes** —
so a 3-plane stack became a colour snapshot. At other plane counts the
axes came out as `Q` (unknown) with no z-step.

---

## How long a z-stack takes

A 100-plane stack took over a minute, which was ~75 % pure
`time.sleep`. Rough per-plane budget:

| Cost | Per plane | Notes |
|---|---|---|
| Settle | **0.02 s** | was 0.3 s — the XY stage figure |
| MTB move + readback | ~0.05 s | synchronous `SetPosition` |
| Snap | ~0.02 s | `snapImage` + `getImage` |
| Exposure | your setting | |
| LED gating | ~0.02 s | only if *Switch light per z-plane* is on |

The Acquisition tab now **predicts the total** from your actual
settings and warns when a stack would outlast the timepoint interval.

If a stack is still too slow:

1. **Fewer planes, or a coarser step.** 100 planes is a lot; the piezo
   resolves ~0.06 µm but your objective's depth of field is the real
   limit.
2. **Untick *Switch light per z-plane*** — holds the shutter open for
   the whole stack. Fewer calls, but more light dose, so this is a
   phototoxicity trade, not a free win.
3. **Drop *Settle per plane* to 0** if the piezo can keep up; watch
   for plane-to-plane blur.

The camera exposure is now set **once per stack** rather than once per
plane — `setExposure` can reconfigure the camera, so 100 planes meant
100 needless round trips.

---

## Definite Focus 2

DF2 holds a focus plane **optically**. Over a multi-hour run that beats
correcting Z from images: it costs no light, no exposure time, and it
works on a specimen with nothing trackable in it.

### One reference per position

DF2 identifies a learned plane by an opaque byte blob it hands back
from `InitOnCurrentFocusPosition`. That blob is what makes
multi-position work possible — **positions sit at different heights, so
one global reference would focus every position on the first one's
plane.** Each position therefore carries its own, saved with it in
`positions.json` as `df2_ref` (base64).

On the *Acquisition* tab:

| Control | What it does |
|---|---|
| **Learn focus → selected position** | Learns the plane you are focused on now and attaches it to the selected position. Also sets that position's `z_um` to the current focus, so "go to position" and "stabilize" agree. |
| **Learn at every position** | Walks the whole list, moving to each and learning there. The normal way to prepare a run. |
| **Stabilize now** | Restores the selected position's plane and corrects focus once. Use it to check DF2 actually works before committing. |
| **Clear reference** | Drops the selected position's plane. |
| **Precision** | `Exact` (slowest, most precise) / `Balanced` / `Fast`. |

The `df2` column in the position table shows which positions are
taught. `0 of N` in the status line means a run would stabilize every
position on whatever plane DF2 last held — which is another position's.

### Who owns Z

This is the one real decision, and it is scientific rather than
technical.

- **DF2 owns Z** (ticked, the default) — the tracker's Z corrections
  are **dropped**. Right when the drift you are fighting is *thermal*.
  Two controllers aiming at the same axis from different measurements
  fight each other every timepoint.
- **Unticked** — DF2 removes the hardware drift and the tracker's
  accumulated dz is applied *on top of* the stabilized plane. Right
  when the **specimen itself** moves in Z. This composes correctly
  because DF2 re-establishes the plane before every frame, so the
  residual Z error the images show is specimen motion.

During a run DF2 fires once per position visit, after the move and
before the first exposure, so every channel and every plane of a stack
references the same plane. A z-stack centres on the Z **DF2 actually
reached**, read back rather than assumed.

### Searching for embryos with DF2

**DF2 is not an autofocus.** It reflects an IR beam off the
coverslip–medium interface and holds a fixed distance from *that
surface*. It has never seen your embryo and cannot find one.
`FindFocus()` searches for the **coverslip**, not the sample.

The searching win is indirect, and it is large: every embryo on that
coverslip sits at roughly the same height above it. So:

1. Find and focus **one** embryo by hand.
2. *Acquisition* tab → **Learn focus** (DF2 now knows "this far above
   the glass").
3. *Stage* tab → **🔒 Hold focus while searching**.
4. Drive XY freely. Every new field arrives already in focus.

**Find coverslip** is for when you are lost in Z entirely — it locates
the glass, and you focus up from there to the specimen.

Hold is switched **off automatically when a run starts**. Continuous
stabilisation holds one plane, so it would drag the focus back while
the stage visits positions at other heights; a run uses the per-visit
`stabilize_now()` with each position's own reference instead.

| | Searching | Running |
|---|---|---|
| Mode | `StabilizePeriodically(On)` | `stabilize_now()` per visit |
| Reference | one plane, held everywhere | one **per position** |
| Why | fields arrive in focus as you drive | positions sit at different heights |

### Which actuator does DF2 drive?

**The trap.** DF2 corrects through *either* the motorized focus drive
*or* the piezo — `GetPiezoFocusUsage()` says which. Disturb one and
measure the other, and a perfectly working DF2 looks completely inert.
The first hardware run of `df2_test.py` reported a failure for exactly
this reason. Everything now measures the **total** focus, focus +
piezo, and `describe()` reports which actuator is in use.

**Confirmed working on this stand, 2026-09-11.** DF2 corrects through
the **focus drive**, not the piezo:

```
defocused via piezo: focus 6618.702 + piezo 255.001 = 6873.703 um (+5.009)
stabilized:          focus 6613.304 + piezo 255.020 = 6868.324 um
    focus moved -5.398 um,  piezo moved +0.019 um
residual total error -0.370 um
```

**Ignore `SignalQuality`.** It reads 0 on this stand even in the run
above, which recovered the plane to within 0.4 µm — it appears not to
be populated outside periodic stabilization. Trust the recovery
measurement, not the quality byte.

### If it is absent or fails

Nothing is fatal. A stand without DF2, a DF2 that will not cast, a
stale reference blob and a failed stabilization all downgrade to
image-based Z correction with a warning — losing focus stabilisation
is a far smaller loss than losing the run. A rejected reference is
reported **once** per position, not on every visit.

Verify on hardware before trusting it:

```bat
python tools\df2_test.py                       :: learn, defocus 5 um, recover
python tools\df2_test.py --offset 20 --mode Exact --tol 0.5
python tools\df2_test.py --repeat 5            :: does the error accumulate?
python tools\df2_test.py --use-piezo on        :: force the actuator
python tools\df2_test.py --offset-test         :: derive one plane from another
python tools\df2_test.py --report-only         :: nothing moves
```

---

## Brightfield tracking and preprocessing

The ROI tracker segments by thresholding for a **bright object on a
dark background**. That is how fluorescence looks. Transmitted light is
the opposite: the background is the brightest thing in the frame, so it
survives every threshold and the embryo is never isolated.

### Where preprocessing is configured

**In the ROI selection dashboard**, alongside the ROIs themselves —
because that is where you decide what to track and can see the effect
on the very image the tracker will receive.

```bat
panel serve interactive_tools\panel_app.py
```

The **Selection** tab's preprocessing controls all apply to the preview
*and* are saved into `tracking_RoIs.json`:

| Control | Effect |
|---|---|
| **Invert contrast (brightfield)** | flips dark-on-bright to bright-on-dark |
| Downscale | speed vs precision |
| Gaussian blur factor | suppresses texture; transmitted light usually needs more than fluorescence |
| Mask kernel size | morphological cleanup |
| Grid size | density of tracked points |
| Tracking mode | single vs multi-ROI |

### How it reaches the tracker

```
ROI dashboard  →  tracking_RoIs.json
               →  TrackingRunner merges every key into positions_config
               →  MicroscopeInterface_MTB.refresh_filename(pos)
               →  frames handed to the tracker are transformed
```

So the dashboard is **authoritative and per position** — different
embryos can need different preprocessing. The channel's `invert` column
is only a default for before any ROIs exist. Re-saving ROIs mid-run
re-applies immediately, no restart.

Two properties worth knowing:

- Inversion works **within the data's own range**, not the dtype's, so
  a frame spanning 11000–14800 keeps full contrast instead of
  collapsing into the bottom of a 0–65535 scale.
- **The file on disk stays raw.** Only what the tracker sees is
  transformed, so later review and the dashboard show the real image,
  and ROI geometry is unaffected.

### If brightfield masking is still poor

Raise the **blur factor** first — transmitted light is inherently
lower-contrast than fluorescence and carries more texture. Then try
downscaling, which both speeds things up and smooths. Global
thresholding knobs (`containment_threshold`, `score_threshold`, `k`,
`c0`) live in `tracking_tools/tracking_config.yaml` under
`roi_tracker`.

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
| `.definite_focus()` | `MTBDefiniteFocus` — DF2; raises if absent |
| `.has_definite_focus()` | never raises, for deciding whether to offer the controls |
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
| `MTBDefiniteFocus.learn()` | learn the current plane; returns its reference blob |
| `.restore(blob)` | re-arm DF2 on a previously learned plane |
| `.stabilize_now()` | one correction back to that plane |
| `.set_operation_mode(name)` | `Exact` / `Balanced` / `Fast` |
| `.piezo_focus_usage()` | True if DF2 corrects via the piezo, False via the focus drive |
| `.offset_reference(blob, um)` | derive a new plane `um` from an existing one |
| `.stabilize_periodically(on)` | continuous hold — the searching mode |
| `.find_focus()` | search for the coverslip when lost in Z |
| `.objective_suitable(order_no)` | whether DF2 can work with an objective |
| `MTBObjective.order_number` | the Zeiss order number DF2 wants |
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
`synthetic_source`, `use_definite_focus`, `df2_owns_z`.

Per position, `positions_config[name]` carries `xyz_um` (required),
`log_dir`, `invert` and `df2_ref` — the base64 DF2 reference blob.
`set_df2_reference(pos, blob)` updates one mid-run, and
`encode_df2_ref` / `decode_df2_ref` convert to and from JSON.

### Command-line tools

```bat
:: hardware checks
python tools\mtb_smoke_test.py                 :: read-only
python tools\mtb_smoke_test.py --move          :: small test moves
python tools\mtb_smoke_test.py --lightpath     :: shutters, turrets, lamps
python tools\mtb_smoke_test.py --objective     :: nosepiece detail
python tools\mtb_smoke_test.py --colibri       :: LED controller detail
python tools\mtb_smoke_test.py --df2           :: Definite Focus 2 interface dump

:: definite focus 2  (MOVES THE FOCUS)
python tools\df2_test.py --report-only         :: state only, nothing moves
python tools\df2_test.py                       :: learn, defocus, prove recovery

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
