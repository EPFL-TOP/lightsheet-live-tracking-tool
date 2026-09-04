from .MicroscopeInterface import (
    MicroscopeInterface_LS1,
    SimulatedMicroscopeInterface_LS1,
    SimulatedMicroscopeInterface_General,
    SimulatedMicroscopeInterface_Zeiss,
    MicroscopeInterface_Zeiss,
    MicroscopeInterface_Files,
)

try:
    from .MicroscopeInterface import MicroscopeInterface_Micromanager
except ImportError:
    MicroscopeInterface_Micromanager = None

# --- MTB (Zeiss Axio Observer 7) -------------------------------------
# Motion via Zeiss MTB 2011, camera via Micro-Manager/PVCAM. Guarded
# because pythonnet and the Zeiss stack exist only on the Zeiss PC.
try:
    from .mtb import (
        MTBAxis,
        MTBError,
        MTBMotion,
        MTBSession,
    )
    from .mtb_backend import MicroscopeInterface_MTB
except ImportError:  # pragma: no cover - platform dependent
    MTBAxis = None
    MTBError = None
    MTBMotion = None
    MTBSession = None
    MicroscopeInterface_MTB = None
