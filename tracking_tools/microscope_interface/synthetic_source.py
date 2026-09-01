"""Synthetic image sources for the Micro-Manager backend.

These callables can be plugged into ``MicroscopeInterface_Micromanager`` via
the ``synthetic_source`` mm_param.  When provided, the frameReady handler
REPLACES the DemoCamera image with the array returned here.  This lets the
smoke test drive the real MM stage/event/scheduler machinery with known
ground-truth images.

Two sources are provided:

  ``DriftingGaussianEmbryo``
      An analytical Gaussian blob whose ground-truth center advances by a
      constant per-timepoint drift in stage micrometres.  The blob is
      rendered relative to the current stage position, so when the tracker
      counter-moves the stage the blob stays centered in the field of view.

  ``ReplayFromFolder``
      Replays previously-recorded TIF stacks from disk in filename order.
      Useful for regression tests against archived acquisitions.
"""

import os
import threading
import numpy as np
import tifffile

from ..logger.logger import init_logger


class DriftingGaussianEmbryo:
    """Analytical synthetic embryo used by the Phase-A smoke test.

    Semantics
    ---------
    For each call for a given ``pos_name`` we advance an internal timepoint
    counter (independent per position) and compute the "true" embryo center
    in world (stage) micrometres:

        true_embryo_um = initial_pos + tp * drift_um_per_tp

    We then render a 2-D Gaussian into an ``uint16`` image such that the
    blob sits at::

        image_center + (true_embryo_um - stage_pos_um) / pixel_size_um

    i.e. if the stage does not move, the blob visibly drifts across the
    frame at ``drift_um_per_tp`` µm/tp; if the tracker perfectly
    compensates by counter-moving the stage, the blob stays glued to
    ``image_center``.  Gaussian noise (std = ``noise_std``) is added.

    Parameters
    ----------
    shape : tuple[int, int]
        (H, W) of the rendered image.
    sigma : float
        Standard deviation of the Gaussian in pixels.
    drift_um_per_tp : tuple[float, float, float]
        Ground-truth (dx, dy, dz) drift applied every timepoint, in µm.
    pixel_size_um : float
        Physical pixel size — converts world µm to image pixels.
    noise_std : float
        Standard deviation of additive Gaussian noise (in counts).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, shape=(512, 512), sigma=15,
                 drift_um_per_tp=(1.5, -1.0, 0.0),
                 pixel_size_um=0.347, noise_std=5.0, seed=42):
        self.shape = tuple(shape)
        self.sigma = float(sigma)
        self.drift_um_per_tp = tuple(float(v) for v in drift_um_per_tp)
        self.pixel_size_um = float(pixel_size_um)
        self.noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

        # Per-position bookkeeping (thread-safe: this callable is invoked
        # from the MDA thread, but a runner script may also inspect state).
        self._lock = threading.Lock()
        self._tp = {}                # pos_name -> next tp counter
        self._initial_stage_um = {}  # pos_name -> (x, y, z) baseline

        # Precompute the pixel-coordinate grid once
        H, W = self.shape
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        self._grid_y = ys
        self._grid_x = xs
        self._center_px = np.array([W / 2.0, H / 2.0], dtype=np.float32)

        self.logger = init_logger(self.__class__.__name__)

    def __call__(self, x_um, y_um, z_um, pos_name):
        with self._lock:
            if pos_name not in self._initial_stage_um:
                # First observation defines the anchor — the embryo starts
                # exactly centered in the field on tp 0.
                self._initial_stage_um[pos_name] = (float(x_um),
                                                    float(y_um),
                                                    float(z_um))
                self._tp[pos_name] = 0
            tp = self._tp[pos_name]
            self._tp[pos_name] = tp + 1
            ix, iy, iz = self._initial_stage_um[pos_name]

        dx_tp, dy_tp, dz_tp = self.drift_um_per_tp
        true_x_um = ix + tp * dx_tp
        true_y_um = iy + tp * dy_tp
        true_z_um = iz + tp * dz_tp

        # Blob offset in pixels = (world - stage) / pixel_size
        blob_px_x = self._center_px[0] + (true_x_um - float(x_um)) / self.pixel_size_um
        blob_px_y = self._center_px[1] + (true_y_um - float(y_um)) / self.pixel_size_um

        # Z-defocus: attenuate amplitude with |z error|
        dz_err = true_z_um - float(z_um)
        z_atten = float(np.exp(-0.5 * (dz_err / max(self.sigma, 1e-3)) ** 2))
        amp = 20000.0 * z_atten

        gy = self._grid_y - blob_px_y
        gx = self._grid_x - blob_px_x
        img = amp * np.exp(-0.5 * (gx * gx + gy * gy) / (self.sigma * self.sigma))
        if self.noise_std > 0:
            img = img + self._rng.normal(0.0, self.noise_std, size=img.shape)
        # Bias so noise is not clipped to zero
        img = img + 100.0

        return np.clip(img, 0, 65535).astype(np.uint16)


class ReplayFromFolder:
    """Replay previously-saved TIFs from ``folder/pos_name/``.

    Filename pattern is Python str-format with ``tp`` (int) and
    ``channel`` (str).  Each call for a given ``pos_name`` advances an
    internal counter and reads the next matching TIF.  When exhausted,
    returns ``None`` (callers must handle that — the MM interface treats
    ``None`` as "keep the DemoCamera image").

    Parameters
    ----------
    folder : str or Path
        Root folder; per-position TIFs live under ``folder/pos_name/``.
    filename_pattern : str
        e.g. ``"t{tp:04d}_{channel}.tif"``.  ``channel`` defaults to the
        empty string; supply via ``set_channel`` if your dataset needs it.
    """

    def __init__(self, folder, filename_pattern="t{tp:04d}_{channel}.tif"):
        self.folder = str(folder)
        self.filename_pattern = filename_pattern
        self._lock = threading.Lock()
        self._tp = {}          # pos_name -> next tp
        self._channel = ""     # can be set by caller
        self.logger = init_logger(self.__class__.__name__)

    def set_channel(self, channel):
        self._channel = channel

    def __call__(self, x_um, y_um, z_um, pos_name):
        with self._lock:
            tp = self._tp.get(pos_name, 0)
            self._tp[pos_name] = tp + 1

        name = self.filename_pattern.format(tp=tp, channel=self._channel)
        path = os.path.join(self.folder, pos_name, name)
        if not os.path.isfile(path):
            self.logger.info(f"[{pos_name}] replay exhausted at tp={tp} ({path})")
            return None
        try:
            arr = tifffile.imread(path)
        except Exception as e:
            self.logger.warning(f"[{pos_name}] could not read {path}: {e}")
            return None
        if arr.ndim == 3:
            # Project multi-plane stacks to max — DemoCamera returns 2-D
            arr = arr.max(axis=0)
        return arr.astype(np.uint16, copy=False)
