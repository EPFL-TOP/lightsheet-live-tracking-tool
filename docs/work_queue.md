# Work queue — deferred improvements

Things we've identified as worth doing but haven't shipped yet. When you
have time, pick from here. Newest at the top.

---

## 1. Simplify the Zeiss panel workflow

**Why**: the current multi-position closed-loop tracking takes ~19 manual
steps spread across the *Tracking* tab, the *ROI Selection* tab, and the
*Advanced* expander. The most important switch (feedback ON/OFF) is the
most hidden. Position values have to be retyped each run. ZEN-detectable
parameters (n_scenes, n_channels) are typed by hand. The result is many
opportunities for "I forgot one thing and the closed loop runs open".

### Quick wins (A–E, do these first)

| Tag | Change | Effort | Impact |
|---|---|---|---|
| **A** | **Auto-pull positions from ZEN** — new button *Pull positions from ZEN* next to the *Initial scene positions* text area. Active after *Start experiment via ZEN API* sets `_state['experiment_id']`. Calls `ExperimentService.Export(experiment_id)`, parses `<SingleTileRegion>` (logic already in [`tools/smoke_test_multipos.py`](../tools/smoke_test_multipos.py)) and fills the text area. | S | High — eliminates manual typing & ordering errors |
| **B** | **Auto-detect n_scenes (and n_channels where possible) from the same XML** — set the *Number of scenes* widget from `len(<SingleTileRegion>)`. Channels can come from a `<Channels>` block in the same export if we want. Side-effect of A or attach to *Start experiment via ZEN API*. | S | High |
| **C** | **Promote *Send relative_move shifts back to ZEN* out of Advanced** — move it to a prominent row right under *Start experiment via ZEN API*, relabel to *"Apply tracker corrections to ZEN stored positions (closed-loop)"*. Default ON. The current placement caused the multi-position run to silently go open-loop because the checkbox stayed unticked. | XS | High |
| **D** | **Auto-suggest output filename** — pre-fill with `<experiment>_<YYYYMMDD_HHMMSS>` (still editable). Eliminates the routine `An output with the same name already exists` error path. | XS | Medium |
| **E** | **Remove (or deeply hide) *Use ZEN gRPC streaming*** until the gateway is upgraded. It currently sits next to the feedback checkbox as an attractive nuisance — easy to tick by accident, breaks things silently. | XS | Low (defensive) |

### Bigger changes (F–H, evaluate after A–E land)

| Tag | Change | Effort | Impact |
|---|---|---|---|
| **F** | **"Readiness" status panel above *Run Tracking*** showing green/red ticks for: ZEN connection · experiment started · positions configured · ROIs saved (n / n_scenes) · feedback enabled. *Run Tracking* button stays greyed out until all green. Single place to see what's missing. | M | High |
| **G** | **Merge Start Experiment + Start Ingest into a single button.** Already done in sequence; this just removes the click. Needs careful error handling so one failure doesn't strand the other. | S | Medium |
| **H** | **Top-to-bottom wizard layout** for the *Tracking* tab: section 1 *Connect*, section 2 *Configure ZEN*, section 3 *Wait for ROIs*, section 4 *Run*. Each unlocks the next. Removes ambiguity about what to click next. | M–L | High |

### Recommended ordering

1. Land **A + B + C + D** in one pass — small, low-risk, eliminates roughly 80 % of the typing/forgetting risk.
2. Re-assess after a successful production run whether **F** (readiness panel) is still worth doing on top.
3. **G** and **H** are polish; defer until the basic flow is rock solid.

---

## Other deferred items

(Add new entries above this divider with date prefixes so the list keeps
its order. Use one short paragraph per item — link to the relevant file
and explain the *why* in one line so we don't have to re-derive it.)

### 2026-09-11 · Definite Focus 2 — decide who owns Z, with data

**Why**: DF2 is implemented and both policies are available
(`df2_owns_z`), but which is right is an empirical question about *this*
specimen, not a design one. DF2 owning Z is the default because the
drift we have actually observed looks thermal.

**To settle it**: run `tools/df2_test.py --repeat 5` to confirm DF2
recovers the plane and that the error does not accumulate. Then run one
tracked experiment each way and compare how far the embryo leaves the
focal plane over a few hours. If the specimen genuinely sinks or rises,
untick *DF2 owns Z* and the tracked offset rides on top of the
stabilized plane.

**Answered 2026-09-11**: `InitOnCurrentFocusPosition` DOES return the
blob on this build — 9 bytes, learned in 0.4 s, with the objective
(Fluar 10x/0.50, order 420140-9901-000) reported *Suitable*. So
per-position references are available.

**Answered — DF2 holds.** It corrects through the **focus drive**
(`GetPiezoFocusUsage` = Off): disturbing the piezo by +5.009 µm, DF2
moved the focus drive −5.398 µm for a residual of −0.370 µm. The first
run only looked like a failure because the test watched the piezo,
which DF2 never touches. `SignalQuality` reads 0 even on this
successful run, so it is not a health signal on this stand.

**Still to measure**: whether the residual accumulates over many
cycles (`--repeat 10`) and whether `Exact` mode tightens it below the
~0.4 µm seen at `Balanced`. Worth knowing before choosing a z-step.

**Worth adopting if it works**:
`SetFocusPositionOffset(Byte[] data, Double offset, String unit,
Byte[]& newData)` derives a new reference a given distance from an
existing one — Zeiss's own answer to multi-position focus. The
coverslip is COMMON to every position, so one good measurement plus a
per-position offset beats N independent measurements, each of which
can fail on its own. Wrapped as `MTBDefiniteFocus.offset_reference()`
and exercised by `df2_test.py --offset-test`; if that passes, change
*Learn at every position* to learn once and offset.

### 2026-09-02 · Interactive checkout GUI — build vs reuse decision

**Why**: bring-up currently uses Java MMStudio + `tools/hw_smoke_test.py`
(scripted). Once Phase 5 stabilises on real Zeiss hardware, we may want
an interactive Qt panel for tweaking device properties, moving stage
mid-experiment, and quick snap-live between tracked runs.

**Options surveyed 2026-09-02 (see workflow `wf_da9f2a52-035`)**:

- **`pymmcore-widgets` 0.12.1** — Qt widget library
  (`SnapButton`, `LiveButton`, `StageWidget`, `ChannelGroupWidget`,
  `PropertyBrowser`, `MDAWidget`). Actively maintained. **Preferred
  path** for a project-specific composed tab.
- `napari-micromanager` 0.3.1 — napari plugin, in maintenance mode.
  Don't adopt.
- `pymmcore-gui` 0.0.1rc0 — standalone MMStudio replacement, still
  pre-release. Wait.

**Deferred decision**: don't add Qt to `requirements.txt` yet. If we
adopt `pymmcore-widgets`, either (a) create a separate
`requirements-mm-gui.txt` extra so Panel-only users don't need Qt, or
(b) build a small standalone `tools/mm_checkout_gui.py`.

**Trigger to revisit**: after the first successful production tracking
run on the Axio Observer, decide based on what actually annoyed the
user during that session.
