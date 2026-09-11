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
    # Light path — all confirmed present 2026-09-04. The camera sits on
    # the LEFT sideport, so a sideport on '100% R' starves it; and a
    # condenser on a phase ring (PH1/2/3) instead of brightfield throws
    # away most of the transmitted light. Both were responsible for
    # exposure-independent read-noise frames.
    "tl_shutter": "MTBTLShutter",
    "tl_lamp": "MTBTLHalogenLamp",
    "sideport": "MTBSideportChanger",
    "condenser": "MTBCondenserContrastChanger",
}

# Roles that make up the transmitted-light path to the camera, in the
# order light travels. Used to build the light-path UI.
LIGHT_PATH_ROLES = ("tl_shutter", "condenser", "objective",
                    "reflector", "sideport")

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


#: A Zeiss order number, e.g. 420660-9970-000.
_ORDER_RE = re.compile(r"\d{6}-\d{3,4}-\d{3}")


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


class MTBLED:
    """One Colibri LED: on/off plus intensity.

    Confirmed on the microscope 2026-09-07. Each element of
    IMTBFluorescenceLEDControl.LED casts to BOTH:

      IMTBFluorescenceLED  -> Wavelength, HasOnOff, GetOnOff/SetOnOff,
                              IsActive, GetExcitationFilter, LampType
      IMTBContinual        -> GetPosition/SetPosition, Get{Min,Max}Position,
                              GetPositionUnit  (this is the INTENSITY axis)

    So intensity is an ordinary continual axis — the same interface the
    stage uses — and on/off is a separate explicit call. An earlier
    reading of a filtered dump concluded there was no brightness member
    and that the trigger buffer was the only route; that was wrong.
    """

    def __init__(self, led, index: int, timeout_ms: int = 10000):
        self.index = index
        self.timeout_ms = timeout_ms
        self._led = self._as(led, "IMTBFluorescenceLED") or led
        self._cont = self._as(led, "IMTBContinual")
        self.unit = self._discover_unit()

    @staticmethod
    def _as(obj, iface_name):
        try:
            mod = __import__("ZEISS.MTB.Api", fromlist=[iface_name])
            return getattr(mod, iface_name)(obj)
        except Exception:
            return None

    def _discover_unit(self) -> str:
        """Intensity unit — '%' on this hardware, but ask rather than
        assume; MTBAxis insists on micrometres, which is wrong here."""
        if self._cont is None:
            return "%"
        try:
            for i in range(self._cont.GetPositionUnitCount()):
                u = self._cont.GetPositionUnit(i)
                if u:
                    return u
        except Exception:
            pass
        return "%"

    @property
    def wavelength(self) -> int | None:
        try:
            return int(self._led.Wavelength)
        except Exception:
            return None

    @property
    def has_on_off(self) -> bool:
        try:
            return bool(self._led.HasOnOff)
        except Exception:
            return False

    @property
    def is_on(self) -> bool | None:
        for getter in ("GetOnOff", "IsActive"):
            try:
                val = getattr(self._led, getter)
                val = val() if callable(val) else val
            except Exception:
                continue
            if isinstance(val, bool):
                return val
            # Some builds return an MTBOnOff-style enum.
            text = str(val).lower()
            if "on" in text and "off" not in text:
                return True
            if "off" in text:
                return False
        return None

    def signature_of(self, method_name: str) -> list[str]:
        """Reflected parameter types of every overload. Diagnostic."""
        out = []
        try:
            for m in self._led.GetType().GetMethods():
                if m.Name != method_name:
                    continue
                params = ", ".join(
                    f"{p.ParameterType.Name} {p.Name}"
                    for p in m.GetParameters()
                )
                out.append(f"{m.ReturnType.Name} {m.Name}({params})")
        except Exception:
            pass
        return out

    def _args_for(self, method_name: str, value: bool):
        """Build argument tuples from the REFLECTED signature.

        Guessing shapes did not work — three plausible ones all failed
        with "SetOnOff failed" on real hardware. Every other MTB setter
        takes (value, mode, timeout), so rather than guess a fourth
        time, read the parameter types and construct arguments to match:
        a two-state enum gets On/Off, an MTBCmdSetModes gets
        Synchronous, and a trailing integer gets the timeout.
        """
        try:
            from System import Enum
        except ImportError:
            # No .NET here (unit tests, non-Zeiss machines) — the
            # caller falls back to the naive argument shapes.
            return []

        candidates = []
        try:
            methods = [m for m in self._led.GetType().GetMethods()
                       if m.Name == method_name]
        except Exception:
            methods = []

        for m in methods:
            params = list(m.GetParameters())
            args, usable = [], True
            for idx, p in enumerate(params):
                pt = p.ParameterType
                tname = pt.Name or ""
                try:
                    is_enum = bool(pt.IsEnum)
                except Exception:
                    is_enum = False

                if is_enum:
                    try:
                        names = list(Enum.GetNames(pt))
                    except Exception:
                        usable = False
                        break
                    lowered = [n.lower() for n in names]
                    if "on" in lowered and "off" in lowered:
                        want = names[lowered.index(
                            "on" if value else "off"
                        )]
                        args.append(Enum.Parse(pt, want))
                    elif "cmdsetmodes" in tname.lower():
                        args.append(resolve_mode("Synchronous"))
                    else:
                        args.append(Enum.Parse(pt, names[0]))
                elif tname == "Boolean":
                    args.append(bool(value))
                elif tname in ("Int32", "Int16", "Byte", "UInt32"):
                    # A trailing integer is the timeout.
                    args.append(int(self.timeout_ms)
                                if idx == len(params) - 1 else 0)
                elif tname == "String":
                    # clientID overloads — skip, the short forms work.
                    usable = False
                    break
                else:
                    usable = False
                    break
            if usable:
                candidates.append((len(args), tuple(args)))

        # Prefer the richest overload; the bare one often is not
        # implemented on these devices.
        candidates.sort(reverse=True)
        return [a for _, a in candidates]

    def set_on(self, on: bool) -> None:
        """Switch the LED on or off."""
        attempts = self._args_for("SetOnOff", on)
        # Fall back to naive shapes if reflection yielded nothing.
        if not attempts:
            attempts = [(bool(on),), (1 if on else 0,)]

        last = None
        for args in attempts:
            try:
                self._led.SetOnOff(*args)
                return
            except Exception as e:
                last = e
        sig = self.signature_of("SetOnOff") or ["<unavailable>"]
        raise MTBError(
            f"LED {self.wavelength}nm: SetOnOff rejected every "
            f"argument form. Reflected signature(s): {sig}"
        ) from last

    @property
    def intensity(self) -> float | None:
        if self._cont is None:
            return None
        try:
            return float(self._cont.GetPosition(self.unit))
        except Exception:
            return None

    @property
    def intensity_limits(self) -> tuple[float, float]:
        if self._cont is None:
            return (0.0, 100.0)
        try:
            return (float(self._cont.GetMinPosition(self.unit)),
                    float(self._cont.GetMaxPosition(self.unit)))
        except Exception:
            return (0.0, 100.0)

    def set_intensity(self, value: float) -> float | None:
        if self._cont is None:
            raise MTBError(
                f"LED {self.wavelength}nm has no intensity axis"
            )
        lo, hi = self.intensity_limits
        target = max(lo, min(hi, float(value)))
        mode = resolve_mode("Synchronous")
        try:
            self._cont.SetPosition(target, self.unit, mode,
                                   self.timeout_ms)
        except Exception as e:
            raise MTBError(
                f"LED {self.wavelength}nm: could not set intensity "
                f"to {target}{self.unit}"
            ) from e
        return self.intensity

    def __repr__(self) -> str:
        return (f"<MTBLED {self.wavelength}nm "
                f"on={self.is_on} {self.intensity}{self.unit}>")


class MTBColibri:
    """The Colibri fluorescence LED controller.

    Six LEDs on this unit (385/430/475/511/567/630 nm). Without
    excitation light a reflector cube change alone produces no
    fluorescence — every "channel" is just the transmitted-light image
    through a different filter, which is what was observed before this
    existed.
    """

    def __init__(self, comp, timeout_ms: int = 10000):
        self._raw = comp
        self.timeout_ms = timeout_ms
        self._ctrl = MTBLED._as(comp, "IMTBFluorescenceLEDControl")
        if self._ctrl is None:
            raise MTBError(
                "component does not expose IMTBFluorescenceLEDControl"
            )
        self._leds: list[MTBLED] | None = None

    @property
    def n_leds(self) -> int:
        try:
            return int(self._ctrl.NumberOfLEDs)
        except Exception:
            return 0

    def leds(self) -> list[MTBLED]:
        if self._leds is None:
            out = []
            for i in range(self.n_leds):
                try:
                    out.append(MTBLED(self._ctrl.LED[i], i,
                                      self.timeout_ms))
                except Exception as e:
                    logger.warning("LED[%d] unavailable: %s", i, e)
            self._leds = out
        return self._leds

    def set_manual_mode(self) -> None:
        """Put the controller in continuous manual LED control.

        LEDLight + Continuous means "we drive the LEDs directly", as
        opposed to LEDTriggerBuffer, where brightness comes from the
        hardware trigger buffer instead.
        """
        try:
            from ZEISS.MTB.Api import (
                MTBFluorescenceLEDOperationMode,
                MTBFluorescenceLEDTriggerMode,
            )
        except Exception:
            return
        mode = resolve_mode("Synchronous")
        for args in (
            (MTBFluorescenceLEDOperationMode.LEDLight,
             MTBFluorescenceLEDTriggerMode.Continuous, mode,
             self.timeout_ms),
            (MTBFluorescenceLEDOperationMode.LEDLight,
             MTBFluorescenceLEDTriggerMode.Continuous, mode),
        ):
            try:
                self._ctrl.SetOperationMode(*args)
                return
            except Exception:
                continue
        logger.warning("could not set Colibri to manual LED mode")

    def all_off(self) -> None:
        for led in self.leds():
            try:
                led.set_on(False)
            except Exception as e:
                logger.warning("LED %s would not switch off: %s",
                               led.wavelength, e)

    def find(self, wavelength: int) -> MTBLED | None:
        """The LED closest to `wavelength`, or None."""
        best, best_gap = None, None
        for led in self.leds():
            wl = led.wavelength
            if wl is None:
                continue
            gap = abs(int(wl) - int(wavelength))
            if best_gap is None or gap < best_gap:
                best, best_gap = led, gap
        # Refuse a nonsensical match — 385 is not 630.
        if best_gap is not None and best_gap > 40:
            return None
        return best

    def select(self, wavelength: int, intensity: float) -> MTBLED:
        """Exclusively switch on one LED at a given intensity."""
        led = self.find(wavelength)
        if led is None:
            raise MTBError(
                f"no Colibri LED near {wavelength} nm; available: "
                f"{[l.wavelength for l in self.leds()]}"
            )
        self.set_manual_mode()
        for other in self.leds():
            if other.index != led.index:
                try:
                    other.set_on(False)
                except Exception:
                    pass
        led.set_intensity(intensity)
        led.set_on(True)
        return led

    def describe(self) -> str:
        lines = [f"Colibri: {self.n_leds} LED(s)"]
        for led in self.leds():
            lo, hi = led.intensity_limits
            lines.append(
                f"    [{led.index}] {led.wavelength} nm  "
                f"on={led.is_on}  has_on_off={led.has_on_off}  "
                f"{led.intensity}{led.unit} ({lo:g}..{hi:g})"
            )
        first = self.leds()[0] if self.leds() else None
        if first is not None:
            for meth in ("SetOnOff", "GetOnOff"):
                for sig in first.signature_of(meth):
                    lines.append(f"    sig: {sig}")
        return "\n".join(lines)


class MTBDefiniteFocus:
    """Definite Focus 2 — hardware focus stabilisation.

    DF2 holds a learned plane against thermal drift optically, which
    over a multi-hour run beats correcting Z from images: it costs no
    light, no exposure time, and does not depend on the specimen
    having trackable structure.

    API confirmed on the microscope 2026-09-11 via
    `tools/mtb_smoke_test.py --df2`. The component casts to
    IMTBFocusStabilizer2, whose useful members are:

        InitOnCurrentFocusPosition(mode, timeout)      -> Boolean
        InitOnCurrentFocusPosition(out Byte[] data, timeout)
        InitWithStabilizingData(mode, Byte[] data, timeout)
        GetStabilizerData()                            -> Byte[]
        StabilizeNow(...)          one-shot correction
        StabilizePeriodically(...) continuous
        FindFocus(...)             search for a surface
        Stop(...)
        Get/SetOperationMode()     Exact | Balanced | Fast
        ObjectiveSuitability(objectiveOrderNumber)
        IsBusy, Period, SignalQuality
        Get/SetPiezoFocusUsage()   DF2 can drive the piezo directly

    The `Byte[]` blob is the key to multi-position work: learn a plane
    at each position, keep its blob, and restore that exact plane on
    every later visit with InitWithStabilizingData. One global
    reference cannot serve several positions at different heights.
    """

    #: DF2 measurement is slow; the Micro-Manager wiki recommends a
    #: 25 s Core timeout for exactly this reason.
    DEFAULT_TIMEOUT_MS = 30000

    def __init__(self, comp, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self._raw = comp
        self.timeout_ms = timeout_ms
        self._c = None
        self.interface = None
        for name in ("IMTBFocusStabilizer2", "IMTBAutoFocus"):
            cast = MTBLED._as(comp, name)
            if cast is not None:
                self._c = cast
                self.interface = name
                break
        if self._c is None:
            raise MTBError(
                "component does not expose IMTBFocusStabilizer2"
            )

    # --- state ---

    @property
    def is_busy(self) -> bool:
        try:
            return bool(self._c.IsBusy)
        except Exception:
            return False

    @property
    def signal_quality(self):
        try:
            return self._c.SignalQuality
        except Exception:
            return None

    @property
    def period(self):
        try:
            return self._c.Period
        except Exception:
            return None

    def operation_mode(self):
        try:
            return str(self._c.GetOperationMode())
        except Exception:
            return None

    def set_operation_mode(self, name: str = "Balanced") -> bool:
        """Exact (slowest, most precise) | Balanced | Fast."""
        try:
            from ZEISS.MTB.Api import MTBFocusStabilizer2OperationMode
            value = getattr(MTBFocusStabilizer2OperationMode, name)
        except Exception as e:
            raise MTBError(
                f"unknown DF2 operation mode {name!r}"
            ) from e
        for args in ((value,), (None, value)):
            try:
                return bool(self._c.SetOperationMode(*args))
            except Exception:
                continue
        return False

    def objective_suitable(self, order_number: str):
        """Whether DF2 can stabilise with a given objective.

        Zeiss objectives carry an order number, e.g. the
        'Objective.420660-9970-000' unique name reported by the
        nosepiece.
        """
        try:
            return str(self._c.ObjectiveSuitability(str(order_number)))
        except Exception:
            return None

    # --- learning and restoring a plane ---

    def learn(self) -> bytes | None:
        """Learn the CURRENT focus plane. Returns its reference blob.

        Keep the blob with the position it was learned at: restoring it
        later reproduces that exact plane. Returns None if this build
        only offers the mode-based overload, in which case the
        reference lives inside DF2 and only one position can use it.
        """
        mode = resolve_mode("Synchronous")
        # Prefer the overload that hands back the data.
        for args in ((None, self.timeout_ms), (None,)):
            try:
                result = self._c.InitOnCurrentFocusPosition(*args)
            except Exception:
                continue
            data = self._extract_bytes(result)
            if data is not None:
                return data
        for args in ((mode, self.timeout_ms), (mode,)):
            try:
                if self._c.InitOnCurrentFocusPosition(*args) is not False:
                    return self.stabilizer_data()
            except Exception:
                continue
        raise MTBError("DF2 could not learn the current focus plane")

    @staticmethod
    def _extract_bytes(result):
        """pythonnet returns out-params in a tuple; find the Byte[]."""
        if result is None or isinstance(result, bool):
            return None
        candidates = result if isinstance(result, tuple) else (result,)
        for item in candidates:
            if isinstance(item, (bytes, bytearray)):
                return bytes(item)
            try:
                if len(item) and not isinstance(item, str):
                    return bytes(bytearray(item))
            except Exception:
                continue
        return None

    def stabilizer_data(self) -> bytes | None:
        try:
            return self._extract_bytes(self._c.GetStabilizerData())
        except Exception:
            return None

    def restore(self, data: bytes) -> bool:
        """Re-arm DF2 on a previously learned plane."""
        if not data:
            raise MTBError("no stabilizing data to restore")
        mode = resolve_mode("Synchronous")
        payload = bytearray(data)
        for args in ((mode, payload, self.timeout_ms),
                     (mode, payload)):
            try:
                return self._c.InitWithStabilizingData(*args) is not False
            except Exception:
                continue
        raise MTBError("DF2 rejected the stored stabilizing data")

    # --- acting ---

    def stabilize_now(self) -> bool:
        """One correction back to the learned plane.

        Called before acquiring at a position: it costs a moment but no
        light, and it is what keeps a long run in focus.
        """
        mode = resolve_mode("Synchronous")
        for args in ((mode, self.timeout_ms), (mode,)):
            try:
                return self._c.StabilizeNow(*args) is not False
            except Exception:
                continue
        raise MTBError("DF2 StabilizeNow failed")

    def find_focus(self) -> bool:
        """Search for a surface — use when no plane is learned yet."""
        mode = resolve_mode("Synchronous")
        for args in ((mode, self.timeout_ms), (mode,)):
            try:
                return self._c.FindFocus(*args) is not False
            except Exception:
                continue
        raise MTBError("DF2 FindFocus failed")

    def stabilize_periodically(self, on: bool, period_s: float = 0.0):
        """Continuous stabilisation.

        Useful for a single position, but it fights a multi-position
        run: DF2 would keep pulling the focus back to one plane while
        the stage visits others. Prefer stabilize_now() per visit.
        """
        try:
            from ZEISS.MTB.Api import MTBOnOff
            flag = MTBOnOff.On if on else MTBOnOff.Off
        except Exception:
            flag = bool(on)
        for args in ((flag, period_s) if period_s else (flag,), (flag,)):
            try:
                return self._c.StabilizePeriodically(*args) is not False
            except Exception:
                continue
        raise MTBError("DF2 StabilizePeriodically failed")

    def stop(self) -> bool:
        mode = resolve_mode("Synchronous")
        for args in ((mode, self.timeout_ms), (mode,)):
            try:
                return self._c.Stop(*args) is not False
            except Exception:
                continue
        return False

    def describe(self) -> str:
        return (
            f"Definite Focus 2 via {self.interface}\n"
            f"    mode={self.operation_mode()}  busy={self.is_busy}\n"
            f"    signal quality={self.signal_quality}  "
            f"period={self.period}"
        )


class MTBChangerDevice:
    """Any discrete-position MTB changer.

    Covers the sideport turret, condenser, reflector wheel and TL
    shutter — all of which cast to IMTBChanger and expose
    Position/MinPosition/MaxPosition (Int16, 1-based),
    GetElement(n) and SetPosition(position, mode, timeout).

    Confirmed present on this stand (2026-09-04):

      MTBSideportChanger      1..3   '100% L' / '100% R'
      MTBReflectorChanger     1..6   named filter cubes / 'none'
      MTBCondenserContrast    1..6   'PH1' / 'PH2' / 'PH3' / 'none'
      MTBTLShutter            1..2   'Open' / ...

    CAUTION on element names: GetElement(MaxPosition) raises on every
    changer here, which suggests GetElement indexes a 0-BASED array
    while Position is 1-based. SetPosition is unambiguous (it takes the
    1-based position), but a NAME read at GetElement(p) may belong to a
    neighbouring slot. Element listings therefore report the index they
    were read at so a mismatch is visible rather than silent.
    """

    _IFACES = ("IMTBChanger",)

    def __init__(self, comp, label: str, timeout_ms: int = 30000):
        self.label = label
        self.timeout_ms = timeout_ms
        self._raw = comp
        self._c = self._cast(comp)

    @classmethod
    def _cast(cls, comp):
        for name in cls._IFACES:
            try:
                mod = __import__("ZEISS.MTB.Api", fromlist=[name])
                cast = getattr(mod, name)(comp)
            except Exception:
                continue
            if cast is not None:
                return cast
        return comp

    @property
    def name(self) -> str:
        try:
            return str(self._raw.Name)
        except Exception:
            return self.label

    @property
    def position(self) -> int | None:
        try:
            return int(self._c.Position)
        except Exception:
            return None

    @property
    def bounds(self) -> tuple[int, int] | None:
        try:
            return (int(self._c.MinPosition), int(self._c.MaxPosition))
        except Exception:
            return None

    def elements(self) -> list[dict]:
        """All positions, each with the name of the element in it.

        CONFIRMED against the stand's own display 2026-09-08:
        `GetElement` indexes a 0-BASED array while `Position` is
        1-based, so the element sitting at position p is at
        **GetElement(p - 1)**.

        The sideport turret proved it — the panel reports

            position 1  VIS 100%, Sideport 0%
            position 2  Sideport 100% Left
            position 3  Sideport 100% Right

        while GetElement(1) returned '100% L' and GetElement(2)
        returned '100% R'. Reading GetElement(p) therefore labelled
        every position with its NEIGHBOUR's element, which mislabelled
        the whole light path and made the objective's magnification —
        and hence the pixel pitch — wrong.
        """
        out = []
        bounds = self.bounds
        if bounds is None:
            return out
        lo, hi = bounds
        cur = self.position
        for pos in range(lo, hi + 1):
            entry = {"index": pos, "name": None,
                     "current": pos == cur, "error": None}
            try:
                el = self._c.GetElement(pos - 1)
                entry["name"] = str(el.Name)
            except Exception as e:
                entry["error"] = type(e).__name__
            out.append(entry)
        return out

    def set_position(self, index: int) -> int | None:
        """Move to a 1-based position. Returns the position after."""
        mode = resolve_mode("Synchronous")
        try:
            ok = self._c.SetPosition(int(index), mode, self.timeout_ms)
        except Exception as e:
            raise MTBError(
                f"{self.label}: could not move to position {index}"
            ) from e
        if ok is False and self.position != int(index):
            raise MTBError(
                f"{self.label}: move to position {index} was refused; "
                f"still at {self.position}"
            )
        return self.position

    def find_position(self, *keywords: str) -> int | None:
        """The position whose element name contains any keyword.

        Element names vary between stands, so 'which position closes
        the shutter' has to be looked up rather than hard-coded.
        """
        for el in self.elements():
            name = (el["name"] or "").lower()
            if any(k.lower() in name for k in keywords):
                return el["index"]
        return None

    def set_open(self, open_it: bool) -> int | None:
        """Open or close a two-state changer used as a shutter.

        Matters for fluorescence: with the transmitted-light shutter
        OPEN, brightfield floods the sensor and the fluorescence image
        has a bright background instead of a dark one. Excitation-only
        imaging needs TL closed.
        """
        want = (("open",) if open_it
                else ("closed", "close", "shut", "blocked"))
        pos = self.find_position(*want)
        if pos is None:
            # Only two positions and one is named 'Open': the other
            # one is closed, whatever it is called.
            open_pos = self.find_position("open")
            bounds = self.bounds
            if open_pos is not None and bounds and bounds[1] == 2:
                pos = open_pos if open_it else (
                    1 if open_pos == 2 else 2
                )
        if pos is None:
            raise MTBError(
                f"{self.label}: cannot tell which position is "
                f"{'open' if open_it else 'closed'}; elements are "
                f"{[e['name'] for e in self.elements()]}"
            )
        return self.set_position(pos)

    @property
    def is_open(self) -> bool | None:
        for el in self.elements():
            if el["current"]:
                name = (el["name"] or "").lower()
                if "open" in name:
                    return True
                if any(k in name for k in ("closed", "close", "shut")):
                    return False
        return None

    def describe(self) -> str:
        bounds = self.bounds
        rng = f"{bounds[0]}..{bounds[1]}" if bounds else "?"
        parts = [f"{self.label} ({self.name}) position "
                 f"{self.position} of {rng}"]
        for e in self.elements():
            mark = " <=" if e["current"] else ""
            nm = e["name"] or f"<{e['error']}>"
            parts.append(f"    {e['index']}: {nm}{mark}")
        return "\n".join(parts)


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
    #: Zeiss order numbers look like 420660-9970-000, and DF2's
    #: ObjectiveSuitability() is the only API that wants one.
    _ORDER_ATTRS = ("ArticleNo", "ArticleNumber", "OrderNumber",
                    "PartNumber", "ArtNo", "UniqueName", "Id")

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

    def _element(self, position: int | None = None):
        """The element at `position` (default: current), cast onward.

        Uses GetElement(position - 1): confirmed 2026-09-08 that
        GetElement indexes a 0-based array while Position is 1-based.
        Reading GetElement(position) returned the NEXT slot's
        objective, so the magnification — and therefore the pixel
        pitch and every drift correction — was silently wrong.

        GetElement hands back an IMTBChangerElement carrying only
        ElementType, so cast it to IMTBObjective before reading
        magnification.
        """
        pos = self.position if position is None else position
        if pos is None:
            return None
        try:
            el = self._changer.GetElement(int(pos) - 1)
        except Exception:
            return None
        if el is None:
            return None
        cast, _ = self._cast(el, self._ELEMENT_IFACES)
        return cast

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
    def order_number(self) -> str | None:
        """The objective's Zeiss order number, if it can be found.

        DF2's ObjectiveSuitability() takes one. No attribute is
        guaranteed, so try the plausible names and then fall back to
        digging the pattern out of whatever strings the element has —
        the unique name is typically 'Objective.420660-9970-000'.
        """
        el = self._element()
        for target in (el, self._changer, self._raw):
            if target is None:
                continue
            val, _ = self._first_attr(target, self._ORDER_ATTRS)
            if val:
                found = _ORDER_RE.search(str(val))
                return found.group(0) if found else str(val)
        for target in (el, self._changer, self._raw):
            if target is None:
                continue
            for attr in ("UniqueName", "Name", "Id"):
                try:
                    found = _ORDER_RE.search(str(getattr(target, attr)))
                except Exception:
                    continue
                if found:
                    return found.group(0)
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
        """Every nosepiece POSITION with the objective actually in it.

        Iterates 1-based positions and reads each element at
        GetElement(pos - 1) — confirmed 2026-09-08 against the stand's
        own display that GetElement is 0-based while Position is
        1-based. The previous version listed raw element indices, which
        both mislabelled the objectives and offered index 0 as a
        selectable slot, producing
          rotation to slot 0 was refused; the turret is still at 2
        """
        out = []
        count = self.element_count or 0
        cur = self.position
        for pos in range(1, count + 1):
            entry = {"index": pos, "name": None, "magnification": None,
                     "aperture": None, "empty": True, "error": None,
                     "current": pos == cur}
            el = self._element(pos)
            if el is None:
                entry["error"] = "unreadable"
                out.append(entry)
                continue

            name, _ = self._first_attr(el, self._NAME_ATTRS)
            entry["name"] = str(name) if name else None
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

    @property
    def is_empty(self) -> bool:
        """True when the current position holds no objective."""
        el = self._element()
        if el is None:
            return True
        try:
            if str(el.ElementType) == "None":
                return True
        except Exception:
            pass
        return (self.name or "").lower() in ("", "none")

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

    def colibri(self) -> "MTBColibri":
        """The fluorescence LED controller, cached."""
        if "_colibri" not in self._axes:
            self._axes["_colibri"] = MTBColibri(
                self.component("leds"), self.timeout_ms
            )
        return self._axes["_colibri"]

    def changer(self, role: str) -> "MTBChangerDevice":
        """A discrete-position changer wrapper, cached per role."""
        key = f"_changer_{role}"
        if key not in self._axes:
            self._axes[key] = MTBChangerDevice(
                self.component(role), role, self.timeout_ms
            )
        return self._axes[key]

    def light_path(self) -> dict[str, "MTBChangerDevice"]:
        """Every reachable light-path changer, keyed by role."""
        out = {}
        for role in LIGHT_PATH_ROLES:
            try:
                out[role] = self.changer(role)
            except Exception as e:
                logger.info("light-path role %s unavailable: %s",
                            role, e)
        return out

    def objective(self) -> "MTBObjective":
        """The nosepiece wrapper, for reading magnification."""
        if "_objective" not in self._axes:
            self._axes["_objective"] = MTBObjective(
                self.component("objective")
            )
        return self._axes["_objective"]

    def definite_focus(self) -> "MTBDefiniteFocus":
        """The Definite Focus 2 stabilizer, cached."""
        if "_df2" not in self._axes:
            # DF2 measurement takes many seconds; never inherit the
            # session's short motion timeout here.
            self._axes["_df2"] = MTBDefiniteFocus(
                self.component("df2"),
                max(self.timeout_ms, MTBDefiniteFocus.DEFAULT_TIMEOUT_MS),
            )
        return self._axes["_df2"]

    def has_definite_focus(self) -> bool:
        """Whether DF2 is fitted and castable. Never raises.

        The panel asks this before offering the controls; a stand
        without DF2 should show nothing rather than an error.
        """
        try:
            self.definite_focus()
            return True
        except Exception as e:
            logger.info("Definite Focus 2 unavailable: %s", e)
            return False

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
