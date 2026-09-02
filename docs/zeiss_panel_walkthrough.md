# Zeiss Panel — end-to-end walkthrough

Multi-position live tracking with stage feedback to ZEN, using the
**ZEN-API–started experiment + file-watching ingest + TilesService position
updates** workflow.

The panel app lives at
[`interactive_tools/zeiss_panel_app.py`](../interactive_tools/zeiss_panel_app.py).
Launch with:

```bat
panel serve interactive_tools/zeiss_panel_app.py --show --port 5022
```

It exposes three tabs: **Tracking**, **ROI Selection**, **Visualisation**.
This walkthrough focuses on the *Tracking* tab top-to-bottom.

---

## 0. Prerequisites — verify positions are saved in ZEN

ZEN persists position edits only when the experiment is *saved* (Ctrl+S
in the ZEN application after editing positions). Edits left in memory
are lost on the next load.

To confirm the positions in your `.czexp` match what you intend, run
the smoke test before you touch the panel:

```bat
:: Replace --experiment with your .czexp name (without extension)
:: Replace --output-name with a fresh value (no .czi conflict).
python tools/smoke_test_multipos.py ^
    --experiment test-tracking-clement ^
    --output-name smoke_test_001 ^
    --watch-seconds 60
```

The script walks the full API-started lifecycle in isolation:

1. Connects to the gateway.
2. Lists experiments, picks `--experiment`.
3. Loads it → gets a fresh `experiment_id`.
4. **Exports + parses `<SingleTileRegion>` positions** — prints `P1, P2, …`
   with their (x, y, z) µm values. *This is the ground truth.*
5. Starts the experiment via `start_experiment`.
6. Re-exports to verify positions are stable post-start.
7. Streams `register_on_status_changed` events for `--watch-seconds`,
   printing each `(scenes_index, is_acquisition_running, time_points_index, ...)`
   tuple as it arrives. Useful for confirming ZEN really does emit
   per-scene events.
8. Stops the experiment.
9. PASS/FAIL summary, with the parsed positions formatted exactly as
   you'd paste them into the panel's *Initial scene positions* text area.

Expected output on a healthy `test-tracking-clement` after you've saved
positions in ZEN:

```
━━━ STEP 4 — Export XML, parse <SingleTileRegion> ──────────────
  ✓ +0.07s  2 position(s) parsed from XML:
    +0.07s    P1: (+0.000, +0.000, +0.000) µm
    +0.07s    P2: (+1000.000, +1000.000, +0.000) µm

━━━ STEP 9 — SUMMARY ───────────────────────────────────────────
  ✓ Smoke test PASSED.
      Positions seen by the API:
        P1: (+0.000, +0.000, +0.000) µm
        P2: (+1000.000, +1000.000, +0.000) µm

  Paste those values into the panel's 'Initial scene positions' text area, e.g.:

      0.0, 0.0, 0.0
      1000.0, 1000.0, 0.0
```

If the printed values **don't match what you see in ZEN's Position list
UI**, you didn't save — go back to ZEN, hit Ctrl+S on the experiment,
and re-run the smoke test until they agree.

Only proceed to step 1 below once the smoke test PASSES with positions
matching ZEN's UI.

---

## 1. Acquisition / experiment

| Field | What to enter | Notes |
|---|---|---|
| *Experiment root* | A fresh folder, e.g. `H:\PROJECTS-03\clement\zeiss_tests\multi_pos_test` | The ingest auto-creates `scene_000`, `scene_001`, … underneath. |
| *Number of scenes* | `2` (or however many ZEN positions you defined) | Matches *P1, P2, …* in ZEN. |
| *Number of channels* | The channel count from your ZEN experiment | Used by the ingest only for completeness diagnostics. |
| *Pixel size x,y* / *Z step* | From your ZEN experiment | Converts pixel-shifts → µm. |
| *Number of Z-slices* | n_z per stack | Ingest uses this to know when a stack is complete. |
| *2-D tracking only* | Leave unchecked | Tick only if you want Z held constant. |
| *Use serverkit* | Leave checked | Routes CoTracker to the EPFL serverkit (`upoates-tethys.epfl.ch:8000`). |

---

## 2. ZEN experiment control (API-started)

This is the new section that owns the running `experiment_id`. Because
ZEN's API can only manipulate experiments **it itself started** (via
`StartExperiment`), you press *Start* from here — **not** from the ZEN UI
button — so the panel app has the id to call `TilesService.add_positions`
with later.

### Steps

1. **Refresh experiment list.** The dropdown gets populated from
   `ExperimentService.GetAvailableExperiments`. Pick the experiment you
   configured in ZEN, e.g. `test-tracking-clement`.

2. **Output filename**: type a **fresh** name every time, e.g.
   `track_run_001`, then `track_run_002`, then `_003` … ZEN refuses to
   overwrite an existing `.czi`. If you reuse a name, the dashboard now
   shows a clear ⚠️ message including the output folder so you can find
   and delete the file.

3. **Initial scene positions in µm** *(critical for multi-position
   feedback)*. Paste one line per scene, in scene-index order:

   ```text
   0, 0, 0
   1000, 1000, 0
   ```

   * Comma-, semicolon- or whitespace-separated.
   * 2 numbers → `(x, y, 0)`. 3 numbers → `(x, y, z)`.
   * Blank lines and `#`-comments are ignored.
   * As you type, the *Parsed N baseline(s)* preview underneath updates
     live so you can confirm the parser saw what you typed.
   * **Why this matters**: the multi-scene TilesService update writes
     `add_positions(baseline + cum_drift)` between cycles. If the
     baselines are wrong, ZEN moves the stage to the wrong place.
     Auto-discovery via `register_on_status_changed` is too racy (the
     stage hasn't settled when ZEN reports `is_acquisition_running=True`)
     — so manual entry is the reliable path.
   * Leave blank only if you want to test auto-discovery; you'll see the
     captured values tagged `[auto, racy]` in the log.

4. **Start experiment via ZEN API.** On success the status pane shows:

   ```
   ✅ Running: test-tracking-clement → track_run_001.czi
   experiment_id: …
   📁 Files at: C:\Users\helsens\Documents\Carl Zeiss\ZEN\Documents\Images
   ```

   The 📁 folder is where the CZI lands (via
   `ExperimentService.GetImageOutputPath`). The per-Z TIFs from the
   *Automated Image Export* tab still go to whichever folder you set
   there — the API doesn't expose that one.

---

## 3. ZEN ingest (per-Z TIF → 3-D stacks)

* *ZEN source folder*: the folder where ZEN drops its per-Z
  `<exp>_S0000(P4)_T000000_Z0000_C00_M0000_ORG.tif` files (the
  *Automated Image Export* path). Manual paste.
* *Ingest poll interval*: leave at 2 s.
* Click **Start Ingest**. You should see:

  ```
  Ingest started — source: <source> → out: <experiment_root>
    positions=['scene_000', 'scene_001'], n_z=2, n_channels=1, …
  Wrote …\scene_000\t0000_C00.tif  (Z=2, shape=(2, 300, 300), …)
  Wrote …\scene_001\t0000_C00.tif  (Z=2, shape=(2, 300, 300), …)
  ```

  for each cycle that completes.

---

## 4. ROI Selection tab

For every scene you want to track:

1. Browse to `<experiment_root>\scene_NNN`.
2. Load `t0000_C00.tif` (or whichever channel/timepoint you prefer).
3. Draw the ROI on the embryo.
4. Click *Save*. This creates `scene_NNN\embryo_tracking\tracking_RoIs.json`.

Scenes without a saved ROI are skipped by the tracker. Scenes get
matched to ZEN's `scenes_index` by the trailing integer in the folder
name (`scene_000` ↔ `scenes_index = 0`), which is also what the ingest
assigns.

---

## 5. Advanced expander (back in the Tracking tab)

Open the expander and toggle:

| Widget | Setting | Why |
|---|---|---|
| **Use ZEN gRPC streaming** | **Unchecked** | Streaming requires Gateway ≥ Autumn 2025; your file-watching path is what works today. |
| **Send relative_move shifts back to ZEN** | ✅ **Checked** | Enables `zen_feedback`; the file watcher opens a gRPC channel to ZEN for stage / TilesService updates. |
| ZEN Gateway address / port / cert / token | preloaded from `zeiss_config.ini` | Sanity-check before clicking *Run Tracking*. |
| Max XY shift / Max Z shift (µm) | defaults (500 / 100) | Soft clamps applied to each shift before it's sent. |

---

## 6. Run Tracking

Click **Run Tracking**. The runtime log should walk through:

```
Initializing trackers
Initialized a new position tracker for position: scene_000
Initialized a new position tracker for position: scene_001
ROI file watcher active for 2 director(y/ies)
ZEN feedback channel opened to localhost:5002 — mode: multi-scene (TilesService)
File-watch mode active — polling every 2.0s for 2 position(s) — stage feedback: ON
  [scene_000] pattern=t{N:04d}_C00.tif start_T=0 cumulative=(0.00, 0.00, 0.00) µm
  [scene_001] pattern=t{N:04d}_C00.tif start_T=0 cumulative=(0.00, 0.00, 0.00) µm
  scene 0: baseline = (+0.0, +0.0, +0.0) µm  [manual]
  scene 1: baseline = (+1000.0, +1000.0, +0.0) µm  [manual]
Status monitor subscribing to experiment …
```

Then, as ZEN produces new TIFs and the tracker computes shifts:

```
[scene_000] queued t0001_C00.tif
[scene_001] queued t0001_C00.tif
[scene_000] Real shift [um] shift (z, y, x): (…, …, …)
[scene_000] scene_idx=0 queued drift (-2.5, +12.4, -0.5) µm  cumulative=(-2.5, +12.4, -0.5) µm
[scene_001] scene_idx=1 queued drift (-3.0, +18.6, -0.5) µm  cumulative=(-3.0, +18.6, -0.5) µm
[status] tp=N scene_idx=0 acq_running=True
[status] tp=N scene_idx=1 acq_running=True
[status] tp=N scene_idx=1 acq_running=False
Applied TilesService position update — scene 0: (-2.5, +12.4, -0.5) µm, scene 1: (+997.0, +1018.6, -0.5) µm
```

The `Applied TilesService position update …` line is the proof of life —
ZEN's stored scene positions have been rewritten with `baseline +
cumulative_drift`.

### Sanity check in ZEN UI

After the first `Applied …` fires, open *Tile/Position list* in ZEN. The
displayed positions should match the values printed in the log. If they
still read `0,0,0` and `1000,1000,0`, the RPC succeeded but ZEN's running
experiment isn't picking up the updated positions for the next cycle —
which would be a deeper constraint we'd need to investigate.

---

## 7. Stopping cleanly

Click the buttons in this order:

1. **Stop Tracking** — stops the runner loop, prints a final
   per-position cumulative-drift summary.
2. **Stop Ingest** — stops the ingest thread.
3. **Stop Experiment** — calls `ExperimentService.Stop(experiment_id)`
   so ZEN exits the experiment cleanly.

---

## What lives in which file

| File | Role |
|---|---|
| [`interactive_tools/zeiss_panel_app.py`](../interactive_tools/zeiss_panel_app.py) | Panel app — widgets, experiment lifecycle, wiring |
| [`tracking_tools/microscope_interface/MicroscopeInterface.py`](../tracking_tools/microscope_interface/MicroscopeInterface.py) — `MicroscopeInterface_Files` | File watcher + ZEN feedback (StageService single-scene, TilesService multi-scene) |
| [`tracking_tools/zen_ingest/ZenIngest.py`](../tracking_tools/zen_ingest/ZenIngest.py) | Per-Z TIF → 3-D stack assembly |
| [`tracking_tools/tracking_runner/TrackingRunner.py`](../tracking_tools/tracking_runner/TrackingRunner.py) | Main tracking loop, ROI re-init watcher |
| [`tools/probe_running_experiment.py`](../tools/probe_running_experiment.py) | Standalone script to inspect what the ZEN gateway exposes |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `An output with the same name already exists.` on Start | Pick a fresh *Output filename*. The dashboard now shows the folder so you can find the conflicting file. |
| `Multi-scene feedback requires an API-started experiment.` on Run Tracking | You ticked feedback ON but started the experiment from ZEN's UI button. Stop, click *Start experiment via ZEN API* in the panel, then Run Tracking. |
| No `Applied TilesService position update` lines | Either (a) feedback is off, (b) some `_initial_pos_m[i]` is still `None` because the manual text area is blank for that scene and auto-discovery missed it, (c) no tracker drift has accumulated yet. Check the startup log lines `scene N: baseline = … [manual]` to confirm all slots are filled. |
| `Stage is not calibrated.` | ZEN-side — initialise the stage from ZEN's hardware menu. |
| `Requested position was not reached.` | Single-scene mode racing with ZEN's own stage moves. Switch to multi-scene mode by configuring two or more scenes — the TilesService path avoids the race. |
