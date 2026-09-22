"""OME companion metadata for movies that already exist on disk."""
from .companion import (  # noqa: F401
    COMPANION_NAME,
    CompanionError,
    FileRec,
    build_ome,
    scan_position,
    write_companion,
)
