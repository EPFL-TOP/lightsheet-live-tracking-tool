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


_FILENAME_RE = re.compile(
    r'_S(?P<S>\d+)\(P\d+\)_T(?P<T>\d+)_Z(?P<Z>\d+)_C(?P<C>\d+)_'
    r'M\d+_ORG\.tif$',
    re.IGNORECASE,
)


class ZenIngest:
    """Background service that converts ZEN per-Z TIFs into our 3-D stacks."""

    def __init__(self, source_dir, out_root, position_names, n_z,
                 n_channels=None, poll_interval_s=2.0):
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

        self._stop_event = threading.Event()
        self._thread = None
        # (S, T, C) → True once we have written the stack
        self._written = set()
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

        # Group files by (S, T, C) → {Z: src_path}
        stacks = {}
        for path in glob.iglob(
            os.path.join(self.source_dir, '**', '*.tif'),
            recursive=True,
        ):
            m = _FILENAME_RE.search(os.path.basename(path))
            if not m:
                continue
            s = int(m.group('S')); t = int(m.group('T'))
            z = int(m.group('Z')); c = int(m.group('C'))
            if (s, t, c) in self._written:
                continue
            stacks.setdefault((s, t, c), {})[z] = path

        # Flush any (S, T, C) that has all expected Z-slices
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

        if os.path.isfile(out_path):
            self._written.add((s, t, c))
            return

        try:
            z_arrays = [tifffile.imread(z_files[z]) for z in sorted(z_files)]
            stack = np.stack(z_arrays)
            tifffile.imwrite(out_path, stack)
            self.logger.info(
                f"Wrote {out_path}  (Z={len(z_arrays)}, shape={stack.shape}, "
                f"dtype={stack.dtype})"
            )
            self._written.add((s, t, c))
        except Exception as e:
            self.logger.warning(
                f"Failed to assemble stack S{s} T{t} C{c}: {e}"
            )
