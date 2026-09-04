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
import re
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

# MTBCmdSetModes numeric values (the enum is [Flags]). These are the
# documented values, but do NOT pass them to .NET directly: pythonnet
# >= 3.0 refuses implicit int -> Enum conversion and fails with
#   "since Python.NET 3.0 int can not be converted to Enum implicitly"
# Use resolve_mode() to obtain a real enum member.
MODE_DEFAULT = 0
MODE_SYNCHRONOUS = 1
MODE_RELATIVE = 2

_MODE_INTS = {
    "Default": MODE_DEFAULT,
    "Synchronous": MODE_SYNCHRONOUS,
    "Relative": MODE_RELATIVE,
    "UnidirectionalBacklash": 4,
    "BidirectionalBacklashSmart": 8,
    "BidirectionalBacklash": 16,
    "Smooth": 32,
    "VariableProfile": 64,
    "Fast": 128,
}
_mode_cache: dict[str, object] = {}


def resolve_mode(name: str = "Synchronous"):
    """Return a real MTBCmdSetModes member for `name`.

    Falls back to the plain integer when the Zeiss assembly is not
    loadable, which is what unit tests run against.
    """
    if name in _mode_cache:
        return _mode_cache[name]
    if name not in _MODE_INTS:
        raise ValueError(
            f"unknown MTBCmdSetModes name {name!r}; "
            f"known: {sorted(_MODE_INTS)}"
        )
    try:
        from ZEISS.MTB.Api import MTBCmdSetModes
        value = getattr(MTBCmdSetModes, name)
    except Exception:
        # No .NET here (unit tests, non-Zeiss machines). Fakes accept
        # the int, and the real path never reaches this branch.
        value = _MODE_INTS[name]
    _mode_cache[name] = value
    return value

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

    @property
    def typical_deviation(self) -> float:
        """Positioning error the axis normally achieves.

        Distinct from step width: a closed-loop piezo resolves 0.01 µm
        but servos with a dither an order of magnitude larger, so its
        readback wanders. Using step width as an arrival tolerance
        made legitimate settles look like refusals.
        """
        try:
            return abs(float(self._c.TypicalDeviation(self.unit)))
        except Exception:
            return 0.0

    @property
    def max_deviation(self) -> float:
        """Worst-case positioning error the axis admits to."""
        try:
            return abs(float(self._c.MaxDeviation(self.unit)))
        except Exception:
            return 0.0

    @property
    def arrival_tolerance(self) -> float:
        """How close counts as "there".

        Takes the largest of step width and the axis's own declared
        typical deviation — the hardware knows its precision better
        than we do.
        """
        return max(self.step, self.typical_deviation, 1e-6)

    # --- writes ---

    def move_to(self, position: float,
                mode_name: str = "Synchronous",
                clamp: bool = True,
                tolerance: float | None = None) -> float:
        """Move to an absolute position (µm). Returns where it landed.

        Absolute rather than MODE_RELATIVE by design: a tracking loop
        computes targets as baseline + cumulative_drift, so an absolute
        target is self-correcting — a dropped move does not permanently
        offset the series the way a dropped relative delta would.

        A target the axis has already reached is a NO-OP, not a move.
        MTB returns False for a zero-distance SetPosition, which means
        "nothing to do" rather than "refused"; commanding one happens
        constantly in practice, because a tracking run with zero drift
        targets exactly the current position. So we skip the call when
        already within `tolerance` (default: one step width), and if
        MTB still returns False we check whether we in fact arrived
        before treating it as an error.
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

        tol = (tolerance if tolerance is not None
               else self.arrival_tolerance)
        current = self.position
        if abs(current - target) <= tol:
            logger.debug(
                "%s: already at %.3f %s (target %.3f, tol %.3f) — "
                "no move commanded",
                self.label, current, self.unit, target, tol,
            )
            return current

        # Must be a real enum member — see resolve_mode().
        mode = resolve_mode(mode_name)

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
                landed = self.position
                if abs(landed - target) <= tol:
                    logger.debug(
                        "%s: SetPosition returned False but the axis "
                        "is at %.3f %s — treating as done",
                        self.label, landed, self.unit,
                    )
                    return landed
                # A servoing axis may sit outside the nominal tolerance
                # yet still be as close as it can hold. Accept up to
                # its declared worst case, but say so.
                slack = max(tol, self.max_deviation)
                if abs(landed - target) <= slack:
                    logger.info(
                        "%s: settled at %.3f %s against target %.3f "
                        "(within its %.3f max deviation, outside the "
                        "%.3f tolerance) — accepting",
                        self.label, landed, self.unit, target,
                        self.max_deviation, tol,
                    )
                    return landed
                raise MTBError(
                    f"{self.label}: SetPosition({target:.3f} "
                    f"{self.unit}) was refused and the axis is still "
                    f"at {landed:.3f}. Possible causes: an axis lock, "
                    f"a latched emergency stop, an active joystick, "
                    f"or — for the motorized focus — Definite Focus 2 "
                    f"holding the axis."
                )
        return self.position

    def move_by(self, delta: float, **kw) -> float:
        """Move by a relative delta (µm) via an absolute target."""
        return self.move_to(self.position + delta, **kw)

    def __repr__(self) -> str:
        return f"<MTBAxis {self.label} @ {self.position:.3f} {self.unit}>"


_MAG_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[xX×]")


def parse_magnification(text: str) -> float | None:
    """Pull the magnification out of an objective name.

    Zeiss objective names carry it, e.g.
      'Plan-Apochromat 20x/0.8 M27'      -> 20.0
      'LD C-Apochromat 40x/1.1 W Korr'   -> 40.0
      'EC Plan-Neofluar 10x/0.3 Ph1'     -> 10.0
    This is the fallback when the typed API is unavailable, and it is
    surprisingly reliable because the convention is universal.
    """
    if not text:
        return None
    m = _MAG_RE.search(str(text))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def sample_pixel_size_um(camera_pixel_um: float,
                         objective_mag: float,
                         adapter_mag: float = 1.0) -> float:
    """Pixel pitch projected onto the sample, in micrometres.

    The tracker measures shifts in pixels and must convert them to
    stage micrometres; get this wrong and every correction is
    systematically mis-scaled.

    Prime 95B pixels are 11 um. With a 20x objective and the 1x CSU
    camera adapter this microscope reports, that is 11/20 = 0.55 um.
    """
    total = float(objective_mag) * float(adapter_mag)
    if total <= 0:
        raise ValueError(
            f"total magnification must be positive, got {total}"
        )
    return float(camera_pixel_um) / total


class MTBObjective:
    """The nosepiece: which objective is in, and its magnification.

    MTB's typed objective interface is not something we have confirmed
    on this installation, so every read is attempted through several
    plausible routes and falls back to parsing the objective's name.
    `probe()` reports which route worked, so the guesswork can be
    removed once we know.
    """

    _MAG_ATTRS = ("Magnification", "magnification", "Mag")
    _NAME_ATTRS = ("Name", "name", "DisplayName", "Text")
    _APERTURE_ATTRS = ("Aperture", "NumericalAperture", "NA")

    def __init__(self, comp, label: str = "objective"):
        self.label = label
        self._raw = comp
        self._changer = self._cast_changer(comp)

    # Confirmed on the microscope 2026-09-04: GetComponent returns a
    # bare IMTBComponent; the nosepiece casts to IMTBObjectiveChanger,
    # IMTBChanger, IMTBBase, IMTBIdent, IMTBEventSink, IMTBComponent.
    # IMTBChanger provides Position (Int16), GetElement(n),
    # GetElementCount(); IMTBObjectiveChanger adds OilStop members.
    _CHANGER_IFACES = ("IMTBObjectiveChanger", "IMTBChanger")

    # IMTBChangerElement exposes ONLY ElementType, so the objective's
    # own data needs a further cast. IMTBObjective is confirmed
    # (2026-09-04) to carry Magnification, Aperture, Name,
    # ImmersionType, WorkingDistance, ContrastMethod. An EMPTY slot is
    # not castable to it and falls through to IMTBChangerElement, whose
    # ElementType reads 'None' — that is how empty slots present.
    _ELEMENT_IFACES = ("IMTBObjective", "IMTBChangerElement")

    @staticmethod
    def _cast(comp, names):
        """Cast through the first interface that yields a usable object."""
        for name in names:
            try:
                mod = __import__("ZEISS.MTB.Api", fromlist=[name])
                cast = getattr(mod, name)(comp)
            except Exception:
                continue
            if cast is not None:
                return cast, name
        return comp, None

    @classmethod
    def _cast_changer(cls, comp):
        # Prefer the specific interface, but do NOT probe for .Position
        # here: an earlier version did, and when the probe raised it
        # silently fell back to the raw IMTBComponent, which has no
        # changer members at all.
        cast, _ = cls._cast(comp, cls._CHANGER_IFACES)
        return cast

    @staticmethod
    def _first_attr(obj, names):
        for n in names:
            try:
                val = getattr(obj, n)
            except Exception:
                continue
            if val not in (None, ""):
                return val, n
        return None, None

    @property
    def position(self) -> int | None:
        val, _ = self._first_attr(self._changer, ("Position",))
        try:
            return int(val) if val is not None else None
        except Exception:
            return None

    def _element(self):
        """The changer element for the current position, cast onward.

        GetElement() hands back an IMTBChangerElement, which carries
        only ElementType — so cast it to whatever objective-specific
        interface exists before reading magnification off it.
        """
        pos = self.position
        if pos is None:
            return None
        for getter in ("GetElement", "GetElementAt", "Element"):
            try:
                fn = getattr(self._changer, getter)
            except Exception:
                continue
            try:
                el = fn(pos) if callable(fn) else fn[pos]
            except Exception:
                continue
            if el is not None:
                cast, _ = self._cast(el, self._ELEMENT_IFACES)
                return cast
        return None

    @property
    def element_count(self) -> int | None:
        try:
            return int(self._changer.GetElementCount())
        except Exception:
            return None

    @property
    def name(self) -> str | None:
        el = self._element()
        for target in (el, self._changer, self._raw):
            if target is None:
                continue
            val, _ = self._first_attr(target, self._NAME_ATTRS)
            if val:
                return str(val)
        return None

    @property
    def aperture(self) -> float | None:
        el = self._element()
        for target in (el, self._changer):
            if target is None:
                continue
            val, _ = self._first_attr(target, self._APERTURE_ATTRS)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
        return None

    @property
    def magnification(self) -> float | None:
        """Magnification, from the typed API or parsed from the name."""
        el = self._element()
        for target in (el, self._changer):
            if target is None:
                continue
            val, _ = self._first_attr(target, self._MAG_ATTRS)
            if val is not None:
                try:
                    mag = float(val)
                    if mag > 0:
                        return mag
                except Exception:
                    pass
        # Fall back to the name, which conventionally contains it.
        return parse_magnification(self.name or "")

    def slots(self) -> list[dict]:
        """Every nosepiece slot: index, name, magnification, aperture.

        Reported in full rather than just the current one, because the
        indexing convention is not fully settled: GetElementCount()
        returns 6 on this scope while GetElement(6) raises "Index was
        outside the bounds of the array", which hints the underlying
        array may be 0-based. Showing all slots lets the operator
        confirm against what is physically in the light path instead of
        trusting a possibly off-by-one lookup.
        """
        out = []
        count = self.element_count or 0
        for idx in range(1, count + 1):
            entry = {"index": idx, "name": None, "magnification": None,
                     "aperture": None, "empty": True, "error": None}
            try:
                raw = self._changer.GetElement(idx)
            except Exception as e:
                entry["error"] = f"{type(e).__name__}"
                out.append(entry)
                continue
            if raw is None:
                out.append(entry)
                continue
            el, _ = self._cast(raw, self._ELEMENT_IFACES)
            name, _ = self._first_attr(el, self._NAME_ATTRS)
            entry["name"] = str(name) if name else None
            # 'None' is how MTB labels an unpopulated slot.
            etype = None
            try:
                etype = str(el.ElementType)
            except Exception:
                pass
            entry["empty"] = (etype == "None"
                              or (entry["name"] or "").lower() == "none")
            mag, _ = self._first_attr(el, self._MAG_ATTRS)
            if mag is not None:
                try:
                    entry["magnification"] = float(mag)
                except Exception:
                    pass
            if entry["magnification"] is None and entry["name"]:
                entry["magnification"] = parse_magnification(
                    entry["name"]
                )
            ap, _ = self._first_attr(el, self._APERTURE_ATTRS)
            if ap is not None:
                try:
                    entry["aperture"] = float(ap)
                except Exception:
                    pass
            out.append(entry)
        return out

    def set_position(self, index: int, timeout_ms: int = 30000) -> int:
        """Rotate the nosepiece to `index`. Returns the position after.

        IMTBChanger.SetPosition takes an Int16 and the same
        MTBCmdSetModes enum the axes use, so it must be a real enum
        member — pythonnet >= 3.0 rejects a bare int.

        Actually turning the turret beats asking the operator which
        objective they believe is fitted: afterwards the magnification
        is read from the hardware rather than assumed.
        """
        mode = resolve_mode("Synchronous")
        try:
            ok = self._changer.SetPosition(int(index), mode, timeout_ms)
        except Exception as e:
            raise MTBError(
                f"{self.label}: could not rotate to slot {index}"
            ) from e
        if ok is False:
            # Same convention as the axes: False can mean "already
            # there" rather than "refused".
            if self.position == int(index):
                logger.debug(
                    "%s: already at slot %s", self.label, index
                )
            else:
                raise MTBError(
                    f"{self.label}: rotation to slot {index} was "
                    f"refused; the turret is still at "
                    f"{self.position}. It may be locked, or the stand "
                    f"may be blocking the change."
                )
        return self.position

    @property
    def is_empty(self) -> bool:
        """True when the current slot holds no objective."""
        el = self._element()
        if el is None:
            return True
        try:
            if str(el.ElementType) == "None":
                return True
        except Exception:
            pass
        return (self.name or "").lower() in ("", "none")

    def probe(self) -> dict:
        """Report what is readable and how — for diagnosing the API."""
        el = self._element()
        info = {
            "changer_type": type(self._changer).__name__,
            "position": self.position,
            "element_found": el is not None,
            "element_type": type(el).__name__ if el is not None else None,
            "name": self.name,
            "magnification": self.magnification,
            "aperture": self.aperture,
        }
        for target, tag in ((el, "element"), (self._changer, "changer")):
            if target is None:
                continue
            try:
                info[f"{tag}_attrs"] = sorted(
                    a for a in dir(target) if not a.startswith("_")
                )[:60]
            except Exception:
                pass
        return info


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

    def objective(self) -> "MTBObjective":
        """The nosepiece wrapper, for reading magnification."""
        if "_objective" not in self._axes:
            self._axes["_objective"] = MTBObjective(
                self.component("objective")
            )
        return self._axes["_objective"]

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
                f"[{lo:.1f}, {hi:.1f}]  step {ax.step:g}  "
                f"dev typ {ax.typical_deviation:g} / max "
                f"{ax.max_deviation:g}  tol {ax.arrival_tolerance:g}"
            )
        return "\n".join(lines)
