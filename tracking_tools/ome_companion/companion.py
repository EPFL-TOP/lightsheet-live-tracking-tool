"""Describe an existing movie as OME, without touching a single pixel.

WHAT THIS IS

An OME *companion* is a small XML file that sits beside TIFFs already on
disk and tells a reader how they compose into a 5-D (T, C, Z, Y, X)
series. The pixel data is never copied, re-encoded or renamed — the
companion only indexes it. For this project that matters twice over: a
single Viventis timepoint is 1.06 GB, so consolidating would mean
rewriting hundreds of gigabytes; and the per-timepoint filenames are
load-bearing (the ROI selection dashboard parses channels out of
``t{NNNN}_{channel}.tif`` with a regex anchored on ``.tif``), so they
must not change.

THE NAME IS NOT COSMETIC

The file must be called ``ome-tiff.companion.ome``, for two independent
reasons, both verified against Bio-Formats 8.0.1:

  * The ``.companion.ome`` SUFFIX is what routes the file to
    ``OMETiffReader``, the only reader that follows
    ``<TiffData><UUID FileName=...>``. Named ``.ome``, ``.ome.xml`` or
    ``.xml`` it goes to ``OMEXMLReader`` instead, which reports the
    dimensions and then cannot read any pixels.
  * The ``ome-tiff`` PREFIX is what the stand's own ZEN-written TIFFs
    already point at: every one of them carries
    ``<BinaryOnly MetadataFile="ome-tiff.companion.ome"/>`` and nothing
    else. Those files are unreadable as z-stacks today because the
    companion they name was never written. Writing ours under that
    exact name repairs them in place.

WHAT WE DELIBERATELY DO NOT DO

  * We do not modify the TIFFs. Bio-Formats resolves a companion from
    ``FileName`` alone — verified against plain ImageJ TIFFs carrying no
    OME block, no BinaryOnly and no UUID, with companion UUIDs that
    match nothing on disk. So no rewrite is needed, and the user asked
    for none.
  * We do not emit ``<Plane>`` elements. The only per-plane quantity
    this project has is the stage position, which is per POSITION and
    belongs in ``<StageLabel>``; a regular interval is already covered
    by ``TimeIncrement``. A full (z, c, t) Plane grid for a
    20-position, 500-timepoint, 3-channel, 100-plane run would be three
    million elements describing nothing we know.
  * We do not import ome_types, lxml or aicsimageio here. Everything
    under ``tracking_tools/`` must declare its third-party imports
    (``tests/test_dependencies_declared.py``), and this needs only the
    standard library plus tifffile, which is already a dependency. The
    tests validate the output against the real OME XSD; the writer does
    not need the schema machinery at runtime.

SCHEMA RULES THAT BITE

Child order inside ``<Pixels>`` is an ``xs:sequence``, so it is not
optional: ``Channel*``, then ``TiffData+``, then ``Plane*``. Bio-Formats'
SAX parser happens to tolerate a wrong order, but the document is then
XSD-invalid and OMERO import rejects it. Inside ``<Image>``,
``<StageLabel>`` comes BEFORE ``<Pixels>``.

The ``<UUID>`` child of ``<TiffData>`` is schema-optional but
reader-fatal: omitting the element entirely makes Bio-Formats throw a
NullPointerException, and an empty text node fails the XSD pattern. So
it is always emitted with a real ``urn:uuid:`` value.

``FileName`` must be RELATIVE and use forward slashes. An absolute path
is XSD-valid and fatal — Bio-Formats concatenates it onto the companion's
own directory and reports FileNotFoundException on the resulting
nonsense. Relative paths also mean the whole folder can be moved or
handed to a collaborator and still open.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: The only acceptable filename. See the module docstring.
COMPANION_NAME = "ome-tiff.companion.ome"

_OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

#: ``t0012_BF.tif`` -> timepoint 12, channel 'BF'. Also matches
#: ``t007.tif`` (no channel), which is how the Cell Tracking Challenge
#: sets are laid out. Anchored on ``.tif`` at the end, exactly like the
#: regexes the rest of the project already uses, so the two agree.
_TNAME = re.compile(r"^t(\d+)(?:_(.*))?\.tiff?$", re.IGNORECASE)

#: numpy dtype name -> OME PixelType. OME has no 64-bit integer type,
#: so those are rejected rather than silently narrowed.
_OME_TYPES = {
    "int8": "int8", "int16": "int16", "int32": "int32",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32",
    "float32": "float", "float64": "double",
    "complex64": "complex", "complex128": "double-complex",
}

#: U+00B5 MICRO SIGN, which is what the OME UnitsLength enumeration
#: uses. Writing 'um' instead makes the attribute invalid.
MICRON = "µm"


class CompanionError(Exception):
    """The companion could not be built, with a reason worth reading."""


@dataclass
class FileRec:
    """One TIFF on disk, and what it contributes to the series."""

    path: str                 #: absolute path
    name: str                 #: basename
    timepoint: int
    channel: str              #: '' when the filename carries none
    n_planes: int
    height: int
    width: int
    dtype: str
    #: Set when the file is readable but suspect. Such files are still
    #: indexed — a warning is far better than dropping a timepoint.
    note: str = ""

    @property
    def shape_key(self) -> tuple:
        """Files only belong in one Image if these all agree."""
        return (self.n_planes, self.height, self.width, self.dtype)


@dataclass
class Group:
    """One ``<Image>``: a position, or one shape-variant within it."""

    position: str
    label: str
    recs: list = field(default_factory=list)
    #: Discovered or supplied calibration, all optional.
    pixel_size_um: float | None = None
    z_step_um: float | None = None
    interval_s: float | None = None
    stage_xyz_um: tuple | None = None
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- scan

def _probe(path: str, tifffile, verify: bool = True) -> FileRec | None:
    """Read one TIFF's header. Returns None if it is not usable.

    Pixels are not read, only the header — a position may hold hundreds
    of gigabytes and the companion needs nothing but shape. The one
    exception is `verify`, which decodes the LAST page; see below.

    WHY THE LAST PAGE AND NOT THE FILE SIZE. A run killed mid-write
    leaves a damaged file, and Bio-Formats fails the WHOLE series on one
    bad member, so it is worth detecting. The obvious test — compare the
    file size against the declared pixel count — does not work, measured
    on a deliberately truncated 5-plane stack:

        cut to 50%:  tifffile reports 1 page, not 5, so the declared
                     count shrinks to match and the size looks fine
        cut to 99%:  5 pages, and 11143 B on disk against 10240 B of
                     pixels, so the size STILL looks fine (IFD overhead)
                     - but decoding the last page raises TiffFileError

    It is also wrong for compressed data, which is legitimately far
    smaller than its pixel count: applied that way it rejected both real
    Cell-Tracking-Challenge folders in data/. Decoding the final page
    catches the case the size test misses and costs one page, not one
    stack.
    """
    name = os.path.basename(path)
    m = _TNAME.match(name)
    if not m:
        return None
    timepoint = int(m.group(1))
    channel = (m.group(2) or "").strip()

    note = ""
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            n_planes = len(tf.pages)
            height, width = int(page.imagelength), int(page.imagewidth)
            dtype = str(page.dtype)
            if verify and n_planes:
                try:
                    tf.pages[n_planes - 1].asarray()
                except Exception as e:
                    note = (f"damaged: the last of {n_planes} plane(s) "
                            f"will not decode ({type(e).__name__}) - "
                            f"likely a run killed mid-write")
    except Exception as e:
        logger.warning("skipping unreadable %s: %s", name, e)
        return None

    return FileRec(path=path, name=name, timepoint=timepoint,
                   channel=channel, n_planes=n_planes, height=height,
                   width=width, dtype=dtype, note=note)


def scan_position(pos_dir: str, header_cache=None,
                  verify: bool = True) -> list:
    """Every indexable TIFF directly inside one position folder.

    Non-recursive on purpose. ``max_proj/`` and ``embryo_tracking/``
    hold DERIVED images; indexing them into the raw series would
    silently mix a projection into the z-stack.

    `header_cache` is an optional dict that makes a LIVE run affordable.
    A run refreshes the companion once per timepoint, and by the end of
    a 20-position, 500-timepoint, 3-channel run that would mean opening
    30 000 files every time — quadratic, and fatal to the acquisition
    loop. Keyed on (path, size, mtime) only new or changed files are
    opened; the rest cost one ``stat`` each.
    """
    import tifffile

    out = []
    try:
        entries = sorted(os.listdir(pos_dir))
    except OSError as e:
        logger.warning("cannot list %s: %s", pos_dir, e)
        return out

    for entry in entries:
        path = os.path.join(pos_dir, entry)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not os.path.isfile(path):
            continue

        if header_cache is not None:
            # size and mtime both, so a file rewritten at the same
            # length within the mtime resolution is still re-read.
            key = (path, st.st_size, st.st_mtime_ns)
            hit = header_cache.get(key)
            if hit is not None:
                out.append(hit)
                continue

        rec = _probe(path, tifffile, verify=verify)
        if rec is not None:
            out.append(rec)
            if header_cache is not None:
                header_cache[(path, st.st_size, st.st_mtime_ns)] = rec
    # Numeric order. Lexical would put t10 before t9 as soon as the
    # timepoint count crosses a power of ten.
    out.sort(key=lambda r: (r.channel, r.timepoint))
    return out


def group_position(position: str, recs: list) -> list:
    """Split one position's files into Images of uniform shape.

    Normally this yields exactly one group. It yields more when the
    operator changed the plane count or binning partway through a run —
    a single ``<Pixels>`` declares ONE SizeZ/SizeX/SizeY, so files that
    disagree cannot share an Image without the reader returning
    zero-filled or truncated planes.
    """
    by_shape: dict = {}
    for rec in recs:
        by_shape.setdefault(rec.shape_key, []).append(rec)

    groups = []
    dominant_planes = max(
        (len(v), v[0].n_planes) for v in by_shape.values()
    )[1] if by_shape else 0
    # Most files first, then deepest: the real series should be
    # Image:0, not whichever shape key happens to sort first as a
    # string. With one file each that tie-break put a truncated
    # single-page file ahead of the intact stack.
    for i, (_, members) in enumerate(
            sorted(by_shape.items(),
                   key=lambda kv: (-len(kv[1]), -kv[1][0].n_planes))):
        label = position if len(by_shape) == 1 else f"{position} #{i}"
        g = Group(position=position, label=label, recs=members)
        if len(by_shape) > 1:
            why = ""
            # Heavy truncation does not make a file unreadable, it makes
            # tifffile report FEWER pages - so a minority group that is
            # shallower than the dominant one is very likely damaged
            # rather than a deliberate settings change.
            # Position in the sort is not the signal — being shallower
            # than the deepest group is.
            if members[0].n_planes < dominant_planes:
                why = (f" - these have only {members[0].n_planes} of "
                       f"{dominant_planes} plane(s), which usually "
                       f"means a run killed mid-write")
            g.warnings.append(
                f"{position}: files do not all share one shape; split "
                f"into {len(by_shape)} Image(s) so no plane is "
                f"zero-filled{why}"
            )
        groups.append(g)
    return groups


# --------------------------------------------------------------- build

def _axes_of(group: Group) -> tuple:
    """(timepoints, channels) for this group, in index order."""
    timepoints = sorted({r.timepoint for r in group.recs})
    channels = sorted({r.channel for r in group.recs})
    return timepoints, channels


def _ome_type(dtype: str) -> str:
    try:
        return _OME_TYPES[dtype]
    except KeyError:
        raise CompanionError(
            f"OME has no pixel type for dtype {dtype!r}"
        ) from None


def _sub(parent, tag, **attrs):
    clean = {k: v for k, v in attrs.items() if v is not None}
    return ET.SubElement(parent, tag, {k: str(v) for k, v in clean.items()})


def build_ome(groups: list, path_prefix=None) -> str:
    """Serialise groups into an OME companion document.

    `path_prefix` maps a group to the directory its files sit in,
    relative to where the companion will be written. Return '' for a
    companion written inside the position folder itself.
    """
    if not groups:
        # XSD-valid and fatal: Bio-Formats throws
        # IndexOutOfBoundsException on a companion with zero Images.
        # Eight real position folders in data/ contain no TIFFs at all,
        # so this is reachable, not theoretical.
        raise CompanionError("refusing to write a companion with no images")

    ET.register_namespace("", _OME_NS)
    ET.register_namespace("xsi", _XSI_NS)
    root = ET.Element(
        f"{{{_OME_NS}}}OME",
        {
            f"{{{_XSI_NS}}}schemaLocation": (
                f"{_OME_NS} {_OME_NS}/ome.xsd"
            ),
            "UUID": f"urn:uuid:{uuid.uuid4()}",
            "Creator": "lightsheet-live-tracking-tool",
        },
    )

    for idx, g in enumerate(groups):
        timepoints, channels = _axes_of(g)
        first = g.recs[0]
        prefix = "" if path_prefix is None else path_prefix(g)

        image = _sub(root, f"{{{_OME_NS}}}Image",
                     ID=f"Image:{idx}", Name=g.label)

        # BEFORE <Pixels>: the schema sequence puts StageLabel first,
        # and this is the natural home for a position's stage
        # coordinates — which is exactly the per-position quantity that
        # would otherwise tempt us into a Plane grid.
        if g.stage_xyz_um is not None:
            x, y, z = g.stage_xyz_um
            _sub(image, f"{{{_OME_NS}}}StageLabel", Name=g.position,
                 X=float(x), XUnit=MICRON,
                 Y=float(y), YUnit=MICRON,
                 Z=float(z), ZUnit=MICRON)

        pixels = _sub(
            image, f"{{{_OME_NS}}}Pixels",
            ID=f"Pixels:{idx}",
            # XYZCT: within one file the planes run over Z, and each
            # file is one (T, C). This is the order the TiffData blocks
            # below assume.
            DimensionOrder="XYZCT",
            Type=_ome_type(first.dtype),
            SizeX=first.width, SizeY=first.height,
            SizeZ=first.n_planes,
            SizeC=len(channels), SizeT=len(timepoints),
            PhysicalSizeX=g.pixel_size_um,
            PhysicalSizeXUnit=MICRON if g.pixel_size_um else None,
            PhysicalSizeY=g.pixel_size_um,
            PhysicalSizeYUnit=MICRON if g.pixel_size_um else None,
            # A fabricated calibration is worse than none, so a single
            # plane gets no Z spacing and a single timepoint no interval.
            PhysicalSizeZ=(g.z_step_um if first.n_planes > 1 else None),
            PhysicalSizeZUnit=(
                MICRON if (g.z_step_um and first.n_planes > 1) else None
            ),
            TimeIncrement=(
                g.interval_s if len(timepoints) > 1 else None
            ),
            TimeIncrementUnit=(
                "s" if (g.interval_s and len(timepoints) > 1) else None
            ),
        )

        # 1. Channels, before any TiffData.
        for ci, chan in enumerate(channels):
            _sub(pixels, f"{{{_OME_NS}}}Channel",
                 ID=f"Channel:{idx}:{ci}",
                 Name=chan or None,
                 SamplesPerPixel=1)

        # 2. TiffData, one per file.
        t_index = {t: i for i, t in enumerate(timepoints)}
        c_index = {c: i for i, c in enumerate(channels)}
        for rec in sorted(g.recs, key=lambda r: (r.timepoint, r.channel)):
            td = _sub(pixels, f"{{{_OME_NS}}}TiffData",
                      FirstT=t_index[rec.timepoint],
                      FirstC=c_index[rec.channel],
                      FirstZ=0, IFD=0, PlaneCount=rec.n_planes)
            rel = f"{prefix}{rec.name}" if prefix else rec.name
            # Forward slashes even on Windows: the value is a URI-ish
            # reference, not a local path.
            u = _sub(td, f"{{{_OME_NS}}}UUID",
                     FileName=rel.replace(os.sep, "/"))
            # Schema-optional, reader-fatal — an absent or empty value
            # makes Bio-Formats throw.
            u.text = f"urn:uuid:{uuid.uuid4()}"

        # 3. No <Plane> elements. See the module docstring.

    _indent(root)
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def _indent(elem, level=0):
    """Pretty-print in place (ET.indent is 3.9+, and we support 3.8)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
            if not (child.tail or "").strip():
                child.tail = pad + "  "
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


# --------------------------------------------------------------- write

def _diagnose(root: str) -> str:
    """Explain what IS in a folder when nothing indexable was found.

    "expected <root>/<position>/t0000_<channel>.tif" is useless on its
    own: it does not say whether the folder is empty, holds images
    under names we do not recognise, or is simply one level off. Naming
    the files actually present answers all three at a glance.
    """
    try:
        entries = sorted(os.listdir(root))
    except OSError as e:
        return f"cannot read {root}: {e}"

    tifs = [e for e in entries if e.lower().endswith((".tif", ".tiff"))]
    subdirs = [e for e in entries
               if os.path.isdir(os.path.join(root, e))
               and not e.startswith(".")]

    if tifs:
        shown = ", ".join(tifs[:5]) + (" ..." if len(tifs) > 5 else "")
        return (
            f"{len(tifs)} TIFF(s) here, but none are named "
            f"t<number>_<channel>.tif — found: {shown}"
        )
    if subdirs:
        shown = ", ".join(subdirs[:8]) + (" ..." if len(subdirs) > 8 else "")
        return (
            f"no TIFFs here, and none in the {len(subdirs)} "
            f"subfolder(s) either: {shown}"
        )
    return "the folder is empty"


def looks_like_position(root: str) -> bool:
    """True when `root` IS a position folder rather than holding them.

    Unambiguous: a folder either has t<number>_<channel>.tif files
    directly inside it or it does not.
    """
    try:
        return any(_TNAME.match(e) for e in os.listdir(root))
    except OSError:
        return False


def discover_positions(root: str, pattern: str = "*") -> list:
    """Position folders directly under an experiment root."""
    import fnmatch

    out = []
    try:
        entries = sorted(os.listdir(root))
    except OSError as e:
        raise CompanionError(f"cannot read {root}: {e}") from None
    for entry in entries:
        if not os.path.isdir(os.path.join(root, entry)):
            continue
        if entry.startswith(".") or entry in ("embryo_tracking", "max_proj"):
            continue
        if fnmatch.fnmatch(entry, pattern):
            out.append(entry)
    return out


def write_companion(root: str, pattern: str = "*",
                    per_position: bool = True,
                    root_level: bool = True,
                    metadata=None,
                    dry_run: bool = False,
                    header_cache=None,
                    verify: bool = True) -> dict:
    """Index an experiment root and write its companion file(s).

    This is the ONE implementation. A live run and the a-posteriori
    tool both call it, so there is no second code path to drift.

    Two companions are written by default because two readers resolve
    paths differently: Bio-Formats follows subdirectory ``FileName``s
    from a root-level companion, while tifffile resolves them against
    the directory of the member TIFF you opened and therefore needs one
    inside each position folder. Both are a few kilobytes.

    `metadata` supplies calibration the files cannot report: a dict of
    {pixel_size_um, z_step_um, interval_s, stage_xyz_um: {position:
    (x, y, z)}}. Everything in it is optional.

    `root` may be EITHER an experiment root holding position folders,
    OR a single position folder holding the frames itself — pointing at
    one position is the obvious thing to do and used to fail with an
    unhelpful message. Which one it is gets decided by looking, not by
    a flag: a folder either has t<number>_<channel>.tif files directly
    in it or it does not.

    Returns a report dict; writes nothing when `dry_run`.
    """
    meta = dict(metadata or {})
    stage = meta.get("stage_xyz_um") or {}

    report = {"root": root, "written": [], "positions": {},
              "warnings": [], "skipped": [], "single_position": False}

    if looks_like_position(root):
        return _write_single_position(
            root, meta, stage, report, dry_run, header_cache, verify
        )

    groups_all = []
    for position in discover_positions(root, pattern):
        pos_dir = os.path.join(root, position)
        recs = scan_position(pos_dir,
                             header_cache=header_cache,
                             verify=verify)
        if not recs:
            report["skipped"].append(position)
            continue

        groups = group_position(position, recs)
        for g in groups:
            g.pixel_size_um = meta.get("pixel_size_um")
            g.z_step_um = meta.get("z_step_um")
            g.interval_s = meta.get("interval_s")
            g.stage_xyz_um = stage.get(position)
            report["warnings"].extend(g.warnings)
            for rec in g.recs:
                if rec.note:
                    report["warnings"].append(f"{position}/{rec.name}: {rec.note}")

        timepoints, channels = _axes_of(groups[0])
        report["positions"][position] = {
            "files": len(recs),
            "timepoints": len(timepoints),
            "channels": channels,
            "planes": groups[0].recs[0].n_planes,
            "shape": (groups[0].recs[0].height, groups[0].recs[0].width),
            "dtype": groups[0].recs[0].dtype,
            "images": len(groups),
        }

        if per_position:
            xml = build_ome(groups, path_prefix=lambda g: "")
            target = os.path.join(pos_dir, COMPANION_NAME)
            if not dry_run:
                _atomic_write(target, xml)
            report["written"].append(target)

        groups_all.extend(groups)

    if not groups_all:
        raise CompanionError(
            f"nothing indexable under {root}.\n"
            f"  {_diagnose(root)}\n"
            f"  Point this at either an experiment root (holding "
            f"position folders) or a single position folder (holding "
            f"t0000_BF.tif and friends)."
        )

    if root_level:
        xml = build_ome(groups_all,
                        path_prefix=lambda g: f"{g.position}/")
        target = os.path.join(root, COMPANION_NAME)
        if not dry_run:
            _atomic_write(target, xml)
        report["written"].append(target)

    return report


def _write_single_position(root, meta, stage, report, dry_run,
                           header_cache, verify) -> dict:
    """`root` is itself a position folder: one companion, bare names.

    There is no experiment level to write, because the position IS the
    root — a second companion here would be the same file under the
    same name.
    """
    position = os.path.basename(os.path.normpath(root)) or "position"
    report["single_position"] = True

    recs = scan_position(root, header_cache=header_cache, verify=verify)
    if not recs:                                  # looks_like_position
        raise CompanionError(                     # matched but nothing
            f"nothing indexable in {root}.\n  {_diagnose(root)}"
        )

    # A folder with frames AND position subfolders is ambiguous; say so
    # rather than silently ignoring half the data.
    nested = [d for d in os.listdir(root)
              if os.path.isdir(os.path.join(root, d))
              and not d.startswith(".")
              and d not in ("embryo_tracking", "max_proj")
              and looks_like_position(os.path.join(root, d))]
    if nested:
        report["warnings"].append(
            f"{root} holds frames of its own AND subfolder(s) with "
            f"frames ({', '.join(nested[:5])}); indexing only the "
            f"frames here. Point at the parent to index those instead."
        )

    groups = group_position(position, recs)
    for g in groups:
        g.pixel_size_um = meta.get("pixel_size_um")
        g.z_step_um = meta.get("z_step_um")
        g.interval_s = meta.get("interval_s")
        g.stage_xyz_um = stage.get(position)
        report["warnings"].extend(g.warnings)
        for rec in g.recs:
            if rec.note:
                report["warnings"].append(f"{rec.name}: {rec.note}")

    timepoints, channels = _axes_of(groups[0])
    report["positions"][position] = {
        "files": len(recs),
        "timepoints": len(timepoints),
        "channels": channels,
        "planes": groups[0].recs[0].n_planes,
        "shape": (groups[0].recs[0].height, groups[0].recs[0].width),
        "dtype": groups[0].recs[0].dtype,
        "images": len(groups),
    }

    xml = build_ome(groups, path_prefix=lambda g: "")
    target = os.path.join(root, COMPANION_NAME)
    if not dry_run:
        _atomic_write(target, xml)
    report["written"].append(target)
    return report


def _atomic_write(path: str, text: str) -> None:
    """Write via a temporary file and rename.

    A live run rewrites the companion after every timepoint, so a
    reader must never catch it half-written. Rename is atomic on POSIX;
    on Windows ``os.replace`` still beats a truncate-and-write.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
