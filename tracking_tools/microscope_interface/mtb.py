"""Zeiss MTB 2011 motion layer, via pythonnet.

Shared foundation for both the tracking backend
(MicroscopeInterface_MTB) and the setup GUI. Everything here concerns
MOTION only — the camera is handled separately through
Micro-Manager's PVCAM adapter, which is what makes "move the stage
while acquiring" straightforward: two independent subsystems, no
shared lock.

Why MTB rather than Micro-Manager's ZeissCAN29 adapter: the Axio
Observer 7 carries its CAN29 bus over USB into Zeiss's CZCanSrv, so a
serial adapter cannot reach it. MTB is Zeiss's hardware abstraction
service and the supported route. Using it does NOT mean running ZEN —
ZEN is merely another MTB client.

Everything below was verified against the real microscope on
2026-09-03 (see tools/mtb_smoke_test.py output):

  - MTBConnection().Login(locale, "")  -> out-param client id
  - GetRoot(client_id)                 -> ZEISS.MTB.MicControl.MTBCtrlRoot
  - root.GetComponent(mtb_id)          -> concrete MTBCtrl* class
  - components must be CAST to IMTBContinual before their position
    members are reachable (they are explicit interface
    implementations in a different assembly, so pythonnet does not
    surface them on the raw object)
  - unit string is 'µm'  (focus also offers 'nm')
  - MTBCmdSetModes is [Flags]: Default=0 Synchronous=1 Relative=2 ...
  - SetPosition(pos, unit, mode, timeout) — the overload WITHOUT a
    clientID is accepted on every axis
  - measured move error: 0.000 µm on X/Y/focus, 0.082 µm on the piezo

Ranges as reported by the hardware:

  axis        position unit  min        max       step
  ----------  -------------  ---------  --------  ------
  x / y       µm             -350000    350000    0.25
  focus       µm / nm        -14000     14000     0.010
  piezo       µm             0          500       0.010
"""
from __future__ import annotations

import logging
import threading

DEFAULT_DLL = (
    r"C:\Program Files\Carl Zeiss\MTB 2011 - 2.12.0.7\MTBApi\MTBApi.dll"
)

# MTBIds confirmed present on the EPFL Axio Observer 7.
MTB_IDS = {
    "axis_x":    "MTBStageAxisX",
    "axis_y":    "MTBStageAxisY",
    "focus":     "MTBFocus",
    "piezo":     "MTBPiezoFocusCan",
    "stage":     "MTBStage",
    "df2":       "MTBFocusStabilizer2",
    "leds":      "MTBFLLEDController",
    "objective": "MTBObjectiveChanger",
    "reflector": "MTBReflectorChanger",
}

# MTBCmdSetModes numeric values (the enum is [Flags]).
MODE_DEFAULT = 0
MODE_SYNCHRONOUS = 1
MODE_RELATIVE = 2

_UM_ALIASES = ("um", "\u00b5m", "micron", "micrometer", "micrometre")

logger = logging.getLogger(__name__)


class MTBError(RuntimeError):
    """Raised when MTB refuses or cannot complete an operation."""


# MTB tears down its realtime subsystem on Logout and CANNOT bring it
# back up in the same process — a second Login fails with
#   MTBException: MTB could not be initialized
#   ConnectToRtSystem(): OpenRtNet("localhost", 1966, ...)
#   'No such interface supported' (0x80000004 = E_NOINTERFACE)
# Observed 2026-09-04. So a process gets exactly ONE session, and every
# consumer (tracker, GUI, tools) must share it.
_shared_session: "MTBSession | None" = None
_shared_lock = threading.Lock()
_logout_happened = False


def load_mtb_api(dll_path: str = DEFAULT_DLL):
    """Load MTBApi.dll. Returns the loaded Assembly.

    Imported lazily so this module can be imported (and unit-tested)
    on machines without pythonnet or without Zeiss software.
    """
    try:
        import clr
    except ImportError as e:
        raise MTBError(
            "pythonnet is not installed — run `pip install pythonnet`"
        ) from e

    import os
    if not os.path.exists(dll_path):
        raise MTBError(f"MTBApi.dll not found at {dll_path}")

    stem = dll_path[:-4] if dll_path.lower().endswith(".dll") else dll_path
    try:
        return clr.AddReference(stem)
    except Exception:
        from System.Reflection import Assembly
        return Assembly.LoadFrom(dll_path)


def _cast_to_continual(comp):
    """Cast a raw MTB component to the interface exposing positions.

    GetComponent() returns concrete classes from ZEISS.MTB.MicControl
    whose IMTBContinual members are explicit implementations. pythonnet
    only surfaces those after an interface cast, so calling
    GetPosition() on the raw object raises AttributeError.
    """
    ifaces = []
    for name in ("IMTBContinual", "IMTBStageAxis", "IMTBAxis",
                 "IMTBFocus"):
        try:
            mod = __import__("ZEISS.MTB.Api", fromlist=[name])
            ifaces.append(getattr(mod, name))
        except Exception:
            continue

    for iface in ifaces:
        try:
            cast = iface(comp)
            cast.GetPositionUnitCount()   # prove the cast works
            return cast
        except Exception:
            continue

    # Some builds may surface the members directly.
    try:
        comp.GetPositionUnitCount()
        return comp
    except Exception as e:
        raise MTBError(
            "component does not expose a position interface "
            "(tried IMTBContinual, IMTBStageAxis, IMTBAxis, IMTBFocus)"
        ) from e


class MTBAxis:
    """One continuously-positionable MTB axis, in micrometres.

    Wraps the IMTBContinual cast and pins the unit string once at
    construction so callers never pass units around.
    """

    def __init__(self, comp, label: str, timeout_ms: int = 10000):
        self.label = label
        self.timeout_ms = timeout_ms
        self._c = _cast_to_continual(comp)
        self.unit = self._discover_unit()
        self._lock = threading.Lock()

    def _discover_unit(self) -> str:
        """Pick the micrometre unit from what the axis advertises."""
        units = []
        try:
            for i in range(self._c.GetPositionUnitCount()):
                try:
                    units.append(self._c.GetPositionUnit(i))
                except Exception:
                    pass
        except Exception as e:
            raise MTBError(
                f"{self.label}: cannot enumerate position units"
            ) from e
        for u in units:
            if u and u.strip().lower() in _UM_ALIASES:
                return u
        if units:
            logger.warning(
                "%s: no micrometre unit among %s — using %r",
                self.label, units, units[0],
            )
            return units[0]
        raise MTBError(f"{self.label}: no position units advertised")

    # --- reads ---

    @property
    def position(self) -> float:
        try:
            return float(self._c.GetPosition(self.unit))
        except Exception as e:
            raise MTBError(f"{self.label}: GetPosition failed") from e

    @property
    def limits(self) -> tuple[float, float]:
        try:
            return (float(self._c.GetMinPosition(self.unit)),
                    float(self._c.GetMaxPosition(self.unit)))
        except Exception as e:
            raise MTBError(f"{self.label}: limit query failed") from e

    @property
    def step(self) -> float:
        try:
            return float(self._c.StepWidth(self.unit))
        except Exception:
            return 0.0

    # --- writes ---

    def move_to(self, position: float,
                mode: int = MODE_SYNCHRONOUS,
                clamp: bool = True) -> float:
        """Move to an absolute position (µm). Returns where it landed.

        Absolute rather than MODE_RELATIVE by design: a tracking loop
        computes targets as baseline + cumulative_drift, so an absolute
        target is self-correcting — a dropped move does not permanently
        offset the series the way a dropped relative delta would.
        """
        lo, hi = self.limits
        target = position
        if clamp:
            target = max(lo, min(hi, position))
            if target != position:
                logger.warning(
                    "%s: target %.3f %s clamped to %.3f (limits "
                    "%.1f..%.1f)",
                    self.label, position, self.unit, target, lo, hi,
                )

        with self._lock:
            try:
                ok = self._c.SetPosition(
                    target, self.unit, mode, self.timeout_ms
                )
            except Exception as e:
                raise MTBError(
                    f"{self.label}: SetPosition({target:.3f} "
                    f"{self.unit}) raised"
                ) from e
            if ok is False:
                raise MTBError(
                    f"{self.label}: SetPosition({target:.3f} "
                    f"{self.unit}) was refused. If this is the "
                    f"motorized focus, Definite Focus 2 may be holding "
                    f"the axis."
                )
        return self.position

    def move_by(self, delta: float, **kw) -> float:
        """Move by a relative delta (µm) via an absolute target."""
        return self.move_to(self.position + delta, **kw)

    def __repr__(self) -> str:
        return f"<MTBAxis {self.label} @ {self.position:.3f} {self.unit}>"


class MTBSession:
    """An MTB login session with component lookup.

    Use as a context manager so Logout always runs — MTB tracks client
    sessions server-side and leaking them is impolite at best.

        with MTBSession() as s:
            print(s.axis("axis_x").position
    """

    def __init__(self, dll_path: str = DEFAULT_DLL, locale: str = "en",
                 timeout_ms: int = 10000):
        self.dll_path = dll_path
        self.locale = locale
        self.timeout_ms = timeout_ms
        self._conn = None
        self._root = None
        self.client_id = None
        self._axes: dict[str, MTBAxis] = {}

    # --- lifecycle ---

    @classmethod
    def shared(cls, **kwargs) -> "MTBSession":
        """The process-wide session. Create it here, never elsewhere.

        MTB allows exactly one Login per process (see the note at the
        top of this module), so the tracker, the GUI and the CLI tools
        must all go through this. Do NOT call disconnect() on the
        result — use close_shared() once, at process exit.
        """
        global _shared_session
        with _shared_lock:
            if _shared_session is None:
                _shared_session = cls(**kwargs).connect()
            return _shared_session

    @classmethod
    def close_shared(cls) -> None:
        """Log out the shared session. Irreversible for this process."""
        global _shared_session
        with _shared_lock:
            if _shared_session is not None:
                _shared_session.disconnect()
                _shared_session = None

    def connect(self) -> "MTBSession":
        if _logout_happened:
            raise MTBError(
                "MTB was already logged out in this process and cannot "
                "be re-initialized — its realtime subsystem does not "
                "come back (OpenRtNet -> E_NOINTERFACE). Use "
                "MTBSession.shared() so one session serves the whole "
                "process, or restart the process."
            )

        load_mtb_api(self.dll_path)
        from ZEISS.MTB.Api import MTBConnection

        self._conn = MTBConnection()
        # Void Login(String culture, out String& ID) — pythonnet
        # surfaces the out-param in the return, whose exact shape
        # varies by version.
        res = self._conn.Login(self.locale, "")
        if isinstance(res, tuple):
            self.client_id = next(
                (v for v in reversed(res) if isinstance(v, str) and v),
                None,
            )
        elif isinstance(res, str):
            self.client_id = res
        if not self.client_id:
            raise MTBError(f"MTB Login returned no client id ({res!r})")

        self._root = self._conn.GetRoot(self.client_id)
        if self._root is None:
            raise MTBError("MTB GetRoot returned None")
        logger.info("MTB connected, clientId=%s", self.client_id)
        return self

    def disconnect(self) -> None:
        global _logout_happened
        if self._conn is not None and self.client_id:
            try:
                self._conn.Logout(self.client_id)
                logger.info("MTB logged out")
            except Exception as e:
                logger.warning("MTB logout failed: %s", e)
            # Mark the process as burnt: no further Login can succeed.
            _logout_happened = True
        self._conn = None
        self._root = None
        self.client_id = None
        self._axes.clear()

    @property
    def is_connected(self) -> bool:
        return self._root is not None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        # Deliberately does NOT disconnect: logging out would prevent
        # any later Login in this process. Call close_shared()
        # explicitly at process exit if you really want to log out.
        return False

    # --- components ---

    def component(self, role_or_id: str):
        """Fetch a raw component by role name or raw MTBId."""
        if self._root is None:
            raise MTBError("not connected — call connect() first")
        mtb_id = MTB_IDS.get(role_or_id, role_or_id)
        comp = self._root.GetComponent(mtb_id)
        if comp is None:
            raise MTBError(f"MTB component {mtb_id!r} is absent")
        return comp

    def axis(self, role: str) -> MTBAxis:
        """Get a cached MTBAxis wrapper for a positionable role."""
        if role not in self._axes:
            self._axes[role] = MTBAxis(
                self.component(role), role, self.timeout_ms
            )
        return self._axes[role]

    def available(self) -> dict[str, str]:
        """Map role -> reported hardware name, for present components."""
        out = {}
        for role, mtb_id in MTB_IDS.items():
            try:
                comp = self._root.GetComponent(mtb_id)
            except Exception:
                continue
            if comp is None:
                continue
            try:
                out[role] = str(comp.Name)
            except Exception:
                out[role] = mtb_id
        return out


class MTBMotion:
    """XYZ motion facade — the surface the tracker and GUI both use.

    Z is served by either the motorized focus or the 500 µm piezo. The
    piezo is the better drift-correction actuator (10 nm step, and it
    sits mid-travel with ~250 µm of headroom either way), so it is the
    default; the motorized drive stays available for coarse work.
    """

    def __init__(self, session: MTBSession, z_axis: str = "piezo"):
        if z_axis not in ("piezo", "focus"):
            raise ValueError("z_axis must be 'piezo' or 'focus'")
        self.session = session
        self.z_role = z_axis
        self.x = session.axis("axis_x")
        self.y = session.axis("axis_y")
        self.z = session.axis(z_axis)

    def get_xyz(self) -> tuple[float, float, float]:
        return (self.x.position, self.y.position, self.z.position)

    def move_to(self, x=None, y=None, z=None) -> tuple[float, float, float]:
        """Move to absolute µm coordinates; None leaves an axis alone."""
        if x is not None:
            self.x.move_to(x)
        if y is not None:
            self.y.move_to(y)
        if z is not None:
            self.z.move_to(z)
        return self.get_xyz()

    def move_by(self, dx=0.0, dy=0.0, dz=0.0) -> tuple[float, float, float]:
        """Relative move in µm. Zero deltas are skipped, not commanded."""
        if dx:
            self.x.move_by(dx)
        if dy:
            self.y.move_by(dy)
        if dz:
            self.z.move_by(dz)
        return self.get_xyz()

    def describe(self) -> str:
        lines = [f"Z axis in use: {self.z_role}"]
        for ax in (self.x, self.y, self.z):
            lo, hi = ax.limits
            lines.append(
                f"  {ax.label:8s} {ax.position:12.3f} {ax.unit}  "
                f"[{lo:.1f}, {hi:.1f}]  step {ax.step:g}"
            )
        return "\n".join(lines)
