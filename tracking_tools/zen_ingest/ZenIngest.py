"""
ZEN ingest service.

Watches the folder where ZEN saves per-Z-slice TIF files and assembles
them into 3-D stacks under our own folder structure.

Input layout (whatever ZEN writes):
    <source>/.../<exp>_H(0)_S0000(P4)_T000000_Z0000_C00_M0000_ORG.tif

Output layout (consumed by bokeh_selection / TrackingRunner):
    <out_root>/<position_name>/t{T:04d}_C{C:02d}.tif    (3-D stack, Z×Y×X)

Notes
-----
* All channels are saved (not only the tracking channel) so the user can
  switch tracking channel mid-experiment without losing data.
* When ``n_z == 1`` the ``stack`` is a 1-Z TIF — same convention so the
  downstream code doesn't branch on dimensionality.
* Source TIFs are scanned recursively so ZEN can save them flat or in
  per-scene subfolders.
* On startup we catch up every (S, T, C) currently on disk; thereafter we
  only re-scan for new combinations every ``poll_interval_s`` seconds.
"""

import glob
import os
import re
import threading

import numpy as np
import tifffile

from ..logger.logger import init_logger


# Each ZEN dimension token (S, T, Z, C, M) appears as `_<X><digits>` and any
# of them can be omitted when the corresponding dimension has size 1
# (no scenes, single Z, single channel, no tiling).  The only field we
# require is T — without a timepoint there is nothing to track.
_TOKEN_RES = {
    'S': re.compile(r'_S(\d+)',  re.IGNORECASE),
    'T': re.compile(r'_T(\d+)',  re.IGNORECASE),
    'Z': re.compile(r'_Z(\d+)',  re.IGNORECASE),
    'C': re.compile(r'_C(\d+)',  re.IGNORECASE),
}


def _parse_zen_filename(name):
    """Return ``(S, T, Z, C)`` from a ZEN-style TIF filename.

    Missing axes default to 0 so single-scene / single-Z / single-channel
    acquisitions still parse.  ``None`` is returned when no T token is
    present or the file extension is not ``.tif`` (case-insensitive).
    """
    if not name.lower().endswith('.tif'):
        return None
    t_m = _TOKEN_RES['T'].search(name)
    if t_m is None:
        return None
    s_m = _TOKEN_RES['S'].search(name)
    z_m = _TOKEN_RES['Z'].search(name)
    c_m = _TOKEN_RES['C'].search(name)
    return (
        int(s_m.group(1)) if s_m else 0,
        int(t_m.group(1)),
        int(z_m.group(1)) if z_m else 0,
        int(c_m.group(1)) if c_m else 0,
    )


class ZenIngest:
    """Background service that converts ZEN per-Z TIFs into our 3-D stacks."""

    # A z-slice file must be at least this many seconds old before we
    # trust it to be fully written by ZEN.  Tweakable per instance via
    # ``min_file_age_s``.
    DEFAULT_MIN_FILE_AGE_S = 1.5

    def __init__(self, source_dir, out_root, position_names, n_z,
                 n_channels=None, poll_interval_s=2.0,
                 min_file_age_s=None):
        """
        Parameters
        ----------
        source_dir : str
            Folder where ZEN writes per-Z TIFs (scanned recursively).
        out_root : str
            Folder where the 3-D stacks are written.  One subfolder per
            position is created automatically.
        position_names : list[str]
            Position folder names indexed by ZEN scene number.
            ``position_names[0]`` receives all S0000 frames, etc.
        n_z : int
            Expected number of Z-slices per stack.  A stack is only flushed
            once exactly this many slices are present on disk.
        n_channels : int | None
            Expected number of channels.  Used only for periodic logging:
            the ingest reports if a scene/timepoint is missing channels so
            you can spot ZEN-side issues early.  ``None`` disables the check.
        poll_interval_s : float
            How often the ingest loop re-scans the source directory.
        """
        self.source_dir = source_dir
        self.out_root = out_root
        self.position_names = list(position_names)
        self.n_z = int(n_z)
        self.n_channels = int(n_channels) if n_channels else None
        self.poll_interval_s = float(poll_interval_s)
        self.min_file_age_s = (
            float(min_file_age_s)
            if min_file_age_s is not None
            else self.DEFAULT_MIN_FILE_AGE_S
        )

        self._stop_event = threading.Event()
        self._thread = None
        # (S, T, C) → True once we have written the stack
        self._written = set()
        # (S, T, C) → number of write attempts so far (for diagnostics)
        self._attempts = {}
        self.logger = init_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name='ZenIngestThread'
        )
        self._thread.start()
        self.logger.info(
            f"Ingest started — source: {self.source_dir} → out: {self.out_root}"
        )
        self.logger.info(
            f"  positions={self.position_names}, n_z={self.n_z}, "
            f"n_channels={self.n_channels}, poll_interval={self.poll_interval_s}s"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.logger.info("Ingest stopped")

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._scan_and_assemble()
            except Exception as e:
                self.logger.warning(f"Ingest scan error: {e}")
            self._stop_event.wait(timeout=self.poll_interval_s)

    def _scan_and_assemble(self):
        if not os.path.isdir(self.source_dir):
            self.logger.info(
                f"Source dir not present yet: {self.source_dir} — waiting"
            )
            return

        # Group files by (S, T, C) → {Z: src_path}.  Filenames missing any
        # of S/Z/C are treated as that axis having index 0 (e.g. a single-
        # channel acquisition produces files without `_C00`).
        stacks = {}
        for path in glob.iglob(
            os.path.join(self.source_dir, '**', '*.tif'),
            recursive=True,
        ):
            parsed = _parse_zen_filename(os.path.basename(path))
            if parsed is None:
                continue
            s, t, z, c = parsed
            if (s, t, c) in self._written:
                continue
            stacks.setdefault((s, t, c), {})[z] = path

        # Flush any (S, T, C) that has all expected Z-slices AND has been
        # quiet long enough for ZEN to finish writing every slice.
        import time as _time
        now = _time.time()
        for (s, t, c), z_files in sorted(stacks.items()):
            if len(z_files) < self.n_z:
                continue
            if s >= len(self.position_names):
                self.logger.warning(
                    f"Scene {s} has no mapped position name "
                    f"(have {len(self.position_names)} names) — skipping"
                )
                self._written.add((s, t, c))
                continue
            # File-age guard: skip if any source slice was modified in the
            # last `min_file_age_s` seconds — likely still being written.
            try:
                newest = max(os.path.getmtime(p) for p in z_files.values())
            except OSError:
                continue
            if (now - newest) < self.min_file_age_s:
                continue
            self._write_stack(s, t, c, z_files)

        # Diagnostic: if n_channels is configured, report any (S, T) that has
        # fewer distinct channels than expected after a flush — points at a
        # ZEN-side acquisition gap.
        if self.n_channels:
            seen_channels = {}
            for (s, t, c) in stacks.keys():
                seen_channels.setdefault((s, t), set()).add(c)
            for (s, t), chans in seen_channels.items():
                if len(chans) < self.n_channels:
                    self.logger.debug(
                        f"S{s} T{t}: only {len(chans)} channel(s) on disk "
                        f"(expected {self.n_channels})"
                    )

    def _write_stack(self, s, t, c, z_files):
        pos_name = self.position_names[s]
        pos_dir = os.path.join(self.out_root, pos_name)
        os.makedirs(pos_dir, exist_ok=True)
        out_name = f't{t:04d}_C{c:02d}.tif'
        out_path = os.path.join(pos_dir, out_name)
        tmp_path = out_path + '.tmp'

        if os.path.isfile(out_path):
            # Existing output: assume previous run already produced it.
            self._written.add((s, t, c))
            return

        # Track retries so a stuck stack is visible in the log instead of
        # silently looping forever.
        self._attempts[(s, t, c)] = self._attempts.get((s, t, c), 0) + 1
        if self._attempts[(s, t, c)] > 1:
            self.logger.warning(
                f"Retrying stack S{s} T{t} C{c} "
                f"(attempt {self._attempts[(s, t, c)]}) — previous attempt "
                "did not complete successfully"
            )

        try:
            z_arrays = [tifffile.imread(z_files[z]) for z in sorted(z_files)]
            stack = np.stack(z_arrays)
            # Atomic write: write to .tmp then rename.  Prevents a half-
            # written stack from being mistaken for a finished one if the
            # process is interrupted.
            tifffile.imwrite(tmp_path, stack)
            os.replace(tmp_path, out_path)
            self.logger.info(
                f"Wrote {out_path}  (Z={len(z_arrays)}, shape={stack.shape}, "
                f"dtype={stack.dtype})"
            )
            self._written.add((s, t, c))
        except Exception as e:
            self.logger.warning(
                f"Failed to assemble stack S{s} T{t} C{c}: {e}"
            )
            # Clean up any partial .tmp so it doesn't accumulate.
            try:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
