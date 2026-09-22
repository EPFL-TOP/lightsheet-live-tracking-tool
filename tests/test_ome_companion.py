"""The OME companion must be schema-valid AND actually readable.

Those are two different things and each catches what the other misses.
Bio-Formats' SAX parser silently tolerates a wrong element order, so
only XSD validation catches that; and a document can be perfectly valid
while pointing at the wrong file, so only reading the pixels back
catches that. Every important test here does both.

Validation goes through raw ``lxml.etree.XMLSchema`` against the OME
2016-06 XSD vendored in ome_types — deliberately NOT
``ome_types.validate_xml``, which upgrades the namespace before
checking and so cannot catch a wrong ``xmlns``.

ome_types and lxml are test-only: the writer itself is standard library
plus tifffile, because everything under ``tracking_tools/`` has to
declare its third-party imports (test_dependencies_declared.py).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from tracking_tools.ome_companion import (  # noqa: E402
    COMPANION_NAME,
    CompanionError,
    build_ome,
    scan_position,
    write_companion,
)
from tracking_tools.ome_companion.companion import (  # noqa: E402
    Group,
    group_position,
)


# ------------------------------------------------------------ helpers

def _xsd():
    """The real OME 2016-06 schema, or skip."""
    ome_types = pytest.importorskip("ome_types")
    etree = pytest.importorskip("lxml.etree")
    path = os.path.join(os.path.dirname(ome_types.__file__),
                        "ome-2016-06.xsd")
    if not os.path.exists(path):
        pytest.skip("OME XSD not vendored in this ome_types build")
    return etree, etree.XMLSchema(etree.parse(path))


def _assert_valid(path):
    etree, schema = _xsd()
    doc = etree.parse(str(path))
    if not schema.validate(doc):
        raise AssertionError(
            "companion is not XSD-valid:\n  "
            + "\n  ".join(str(e.message) for e in schema.error_log)
        )


def _write_stack(path, value, planes=5, size=32, dtype=np.uint16,
                 **kw):
    """A frame written exactly as MicroscopeInterface_MTB._save does."""
    shape = (planes, size, size) if planes > 1 else (size, size)
    arr = np.full(shape, value, dtype)
    meta = {"axes": "ZYX" if planes > 1 else "YX"}
    if planes > 1:
        meta.update(spacing=1.5, unit="um")
    tifffile.imwrite(str(path), arr, imagej=True, metadata=meta, **kw)
    return arr


@pytest.fixture
def experiment(tmp_path):
    """2 positions x 3 timepoints x 2 channels x 5 planes, plus a
    single-frame position and an empty one.

    Each pixel value encodes (position, timepoint, channel) so a test
    can prove the right FILE landed at the right (T, C) index rather
    than merely that the shape came out right.
    """
    root = tmp_path / "exp"
    for pi, pos in enumerate(("scene_000", "scene_001")):
        d = root / pos
        d.mkdir(parents=True)
        for t in range(3):
            for ci, chan in enumerate(("BF", "GFP")):
                _write_stack(d / f"t{t:04d}_{chan}.tif",
                             pi * 100 + t * 10 + ci)
    d = root / "scene_002"
    d.mkdir()
    _write_stack(d / "t0000_BF.tif", 7, planes=1)
    (root / "scene_003").mkdir()          # no images at all
    return root


def _value(pi, t, ci):
    return pi * 100 + t * 10 + ci


# ------------------------------------------------------------ the name

def test_the_companion_is_named_exactly_ome_tiff_companion_ome():
    """Two independent reasons, both load-bearing.

    The `.companion.ome` SUFFIX routes the file to Bio-Formats'
    OMETiffReader, the only reader that follows
    <TiffData><UUID FileName=...>; named `.ome` or `.xml` it goes to
    OMEXMLReader, which reports dimensions and then cannot read a
    pixel. The `ome-tiff` PREFIX is what the stand's own ZEN-written
    TIFFs already point at in their <BinaryOnly MetadataFile=...>.
    """
    assert COMPANION_NAME == "ome-tiff.companion.ome"
    assert COMPANION_NAME.endswith(".companion.ome")


def test_companions_are_written_at_both_levels(experiment):
    report = write_companion(str(experiment))
    names = {os.path.relpath(p, experiment) for p in report["written"]}
    assert COMPANION_NAME in names, "no experiment-level companion"
    assert f"scene_000/{COMPANION_NAME}" in names, (
        "no per-position companion — tifffile resolves FileName "
        "against the member file's directory and needs one"
    )


# ------------------------------------------------------------ validity

def test_every_companion_is_xsd_valid(experiment):
    report = write_companion(
        str(experiment),
        metadata={"pixel_size_um": 0.325, "z_step_um": 1.5,
                  "interval_s": 60.0,
                  "stage_xyz_um": {"scene_000": (10.0, 20.0, 30.0)}},
    )
    assert report["written"]
    for path in report["written"]:
        _assert_valid(path)


def test_REGRESSION_element_order_inside_pixels(experiment):
    """Channel* then TiffData+ then Plane* is an xs:sequence, so the
    order is not optional. Bio-Formats tolerates a wrong one but the
    document is then XSD-invalid and OMERO rejects it."""
    write_companion(str(experiment))
    text = (experiment / "scene_000" / COMPANION_NAME).read_text()
    first_channel = text.index("<Channel")
    first_tiffdata = text.index("<TiffData")
    assert first_channel < first_tiffdata, (
        "TiffData precedes Channel — XSD-invalid"
    )


def test_stage_label_precedes_pixels(experiment):
    """Inside <Image> the schema puts StageLabel before Pixels."""
    write_companion(
        str(experiment),
        metadata={"stage_xyz_um": {"scene_000": (1.0, 2.0, 3.0)}},
    )
    text = (experiment / "scene_000" / COMPANION_NAME).read_text()
    assert text.index("<StageLabel") < text.index("<Pixels"), (
        "StageLabel after Pixels — XSD-invalid"
    )
    _assert_valid(experiment / "scene_000" / COMPANION_NAME)


def test_tiffdata_carries_a_real_uuid(experiment):
    """Schema-optional, reader-fatal: an absent <UUID> child makes
    Bio-Formats throw a NullPointerException, and an empty text node
    fails the XSD pattern."""
    write_companion(str(experiment))
    text = (experiment / "scene_000" / COMPANION_NAME).read_text()
    assert "<UUID FileName=" in text
    assert ">urn:uuid:" in text, "UUID element has no value"


def test_filenames_are_relative_and_use_forward_slashes(experiment):
    """An absolute FileName is XSD-valid and fatal: Bio-Formats
    concatenates it onto the companion's own directory."""
    write_companion(str(experiment))
    for path in (experiment / COMPANION_NAME,
                 experiment / "scene_000" / COMPANION_NAME):
        text = path.read_text()
        for line in text.splitlines():
            if "FileName=" in line:
                value = line.split('FileName="')[1].split('"')[0]
                assert not os.path.isabs(value), f"absolute: {value}"
                assert "\\" not in value, f"backslash: {value}"


# ------------------------------------------------- does it actually read

def test_the_companion_reads_back_as_a_5d_series(experiment):
    write_companion(str(experiment))
    member = experiment / "scene_000" / "t0000_BF.tif"
    xml = (experiment / "scene_000" / COMPANION_NAME).read_text()
    with tifffile.TiffFile(str(member), omexml=xml, is_ome=True) as tf:
        series = tf.series[0]
        assert series.shape == (3, 2, 5, 32, 32)
        assert series.axes == "TCZYX"


def test_REGRESSION_each_file_lands_at_the_right_index(experiment):
    """Shape equality is not enough: a companion can be perfectly
    valid, the right shape, and still map files to the wrong (T, C).
    Only a per-index pixel check catches that, and it also catches the
    zero-filled padding you get from a wrong PlaneCount."""
    write_companion(str(experiment))
    for pi, pos in enumerate(("scene_000", "scene_001")):
        xml = (experiment / pos / COMPANION_NAME).read_text()
        member = experiment / pos / "t0000_BF.tif"
        with tifffile.TiffFile(str(member), omexml=xml,
                               is_ome=True) as tf:
            arr = tf.series[0].asarray()
        for t in range(3):
            for ci in range(2):
                got = int(arr[t, ci].ravel()[0])
                assert got == _value(pi, t, ci), (
                    f"{pos} T={t} C={ci}: got {got}, "
                    f"expected {_value(pi, t, ci)}"
                )


def test_the_tiffs_are_never_modified(experiment):
    """The entire premise. The user asked for an index, not a rewrite;
    a Viventis timepoint is 1.06 GB."""
    def digest():
        out = {}
        for dirpath, _, names in os.walk(experiment):
            for n in names:
                if n.endswith(".tif"):
                    p = os.path.join(dirpath, n)
                    out[p] = hashlib.md5(open(p, "rb").read()).hexdigest()
        return out

    before = digest()
    assert before
    write_companion(str(experiment))
    assert digest() == before


# ------------------------------------------------------ calibration

def test_calibration_is_recorded_when_supplied(experiment):
    write_companion(
        str(experiment),
        metadata={"pixel_size_um": 0.325, "z_step_um": 1.5,
                  "interval_s": 60.0},
    )
    text = (experiment / "scene_000" / COMPANION_NAME).read_text()
    assert 'PhysicalSizeX="0.325"' in text
    assert 'PhysicalSizeZ="1.5"' in text
    assert 'TimeIncrement="60.0"' in text
    # U+00B5 MICRO SIGN: the OME UnitsLength enumeration uses it, and
    # plain 'um' makes the attribute invalid.
    assert 'PhysicalSizeXUnit="µm"' in text


def test_no_calibration_is_recorded_when_not_supplied(experiment):
    """A fabricated calibration is worse than none — it silently
    mis-scales every downstream measurement."""
    write_companion(str(experiment))
    text = (experiment / "scene_000" / COMPANION_NAME).read_text()
    assert "PhysicalSizeX=" not in text
    assert "TimeIncrement=" not in text


def test_single_plane_gets_no_z_spacing(experiment):
    """scene_002 is one 2-D frame at one timepoint: a Z spacing or a
    time increment there would be invented."""
    write_companion(str(experiment),
                    metadata={"z_step_um": 1.5, "interval_s": 60.0})
    text = (experiment / "scene_002" / COMPANION_NAME).read_text()
    assert 'SizeZ="1"' in text
    assert "PhysicalSizeZ=" not in text
    assert "TimeIncrement=" not in text


def test_no_plane_elements_are_emitted(experiment):
    """A (z, c, t) Plane grid for a 20-position, 500-timepoint,
    3-channel, 100-plane run is three million elements describing
    nothing we know. Stage position is per POSITION and lives in
    StageLabel; a regular interval is TimeIncrement."""
    write_companion(str(experiment))
    assert "<Plane" not in (experiment / COMPANION_NAME).read_text()


# --------------------------------------------------- awkward reality

def test_LZW_compressed_files_are_indexed_not_rejected(tmp_path):
    """The truncation guard compares file size against declared pixel
    count. Applied to compressed data that is always 'too small', and
    it rejected both real Cell-Tracking-Challenge folders in data/."""
    d = tmp_path / "exp" / "01"
    d.mkdir(parents=True)
    for t in range(3):
        tifffile.imwrite(str(d / f"t{t:03d}.tif"),
                         np.zeros((64, 64), np.uint8),
                         compression="LZW")
    recs = scan_position(str(d))
    assert len(recs) == 3, "compressed files were dropped"
    assert all(not r.note for r in recs), (
        f"false truncation warning: {[r.note for r in recs]}"
    )


def test_three_digit_timepoints_with_no_channel(tmp_path):
    """The Cell Tracking Challenge layout: t000.tif, no suffix."""
    d = tmp_path / "exp" / "01"
    d.mkdir(parents=True)
    for t in range(4):
        _write_stack(d / f"t{t:03d}.tif", t, planes=1)
    recs = scan_position(str(d))
    assert [r.timepoint for r in recs] == [0, 1, 2, 3]
    assert {r.channel for r in recs} == {""}
    write_companion(str(tmp_path / "exp"))
    _assert_valid(tmp_path / "exp" / "01" / COMPANION_NAME)


def test_a_channel_name_containing_a_space_round_trips(tmp_path):
    """The lightsheet writes 't0001_Channel 1.tif'."""
    d = tmp_path / "exp" / "pos1"
    d.mkdir(parents=True)
    for t in range(2):
        _write_stack(d / f"t{t:04d}_Channel 1.tif", t, planes=3)
    recs = scan_position(str(d))
    assert {r.channel for r in recs} == {"Channel 1"}
    write_companion(str(tmp_path / "exp"))
    _assert_valid(tmp_path / "exp" / "pos1" / COMPANION_NAME)
    assert 'Name="Channel 1"' in (d / COMPANION_NAME).read_text()


def test_timepoints_sort_numerically_not_lexically(tmp_path):
    """t10 must not precede t9. Reachable the moment a run passes ten
    timepoints."""
    d = tmp_path / "exp" / "pos"
    d.mkdir(parents=True)
    for t in (0, 1, 2, 9, 10, 11):
        _write_stack(d / f"t{t:04d}_BF.tif", t, planes=1)
    recs = scan_position(str(d))
    assert [r.timepoint for r in recs] == [0, 1, 2, 9, 10, 11]


def test_a_position_with_no_images_is_skipped(experiment):
    report = write_companion(str(experiment))
    assert "scene_003" in report["skipped"]
    assert not (experiment / "scene_003" / COMPANION_NAME).exists()


def test_REGRESSION_an_empty_experiment_writes_nothing(tmp_path):
    """A zero-Image companion is XSD-VALID and crashes Bio-Formats with
    IndexOutOfBoundsException. Eight real position folders in data/
    hold no TIFFs, so this is reachable, not theoretical."""
    root = tmp_path / "exp"
    (root / "scene_000").mkdir(parents=True)
    with pytest.raises(CompanionError):
        write_companion(str(root))
    assert not (root / COMPANION_NAME).exists()


def test_build_ome_refuses_zero_groups():
    with pytest.raises(CompanionError, match="no images"):
        build_ome([])


def test_mixed_shapes_split_into_separate_images(tmp_path):
    """One <Pixels> declares ONE SizeZ. Files that disagree cannot
    share an Image without the reader zero-filling or truncating."""
    d = tmp_path / "exp" / "pos"
    d.mkdir(parents=True)
    _write_stack(d / "t0000_BF.tif", 1, planes=5)
    _write_stack(d / "t0001_BF.tif", 2, planes=5)
    _write_stack(d / "t0002_BF.tif", 3, planes=9)     # operator changed it
    groups = group_position("pos", scan_position(str(d)))
    assert len(groups) == 2
    assert any("shape" in w for g in groups for w in g.warnings)
    write_companion(str(tmp_path / "exp"))
    _assert_valid(d / COMPANION_NAME)


def test_derived_subfolders_are_not_indexed(experiment):
    """max_proj/ and embryo_tracking/ hold DERIVED images; folding a
    projection into the raw z-stack would corrupt the series."""
    derived = experiment / "scene_000" / "max_proj"
    derived.mkdir()
    _write_stack(derived / "t0000.tif", 99, planes=1)
    report = write_companion(str(experiment))
    assert report["positions"]["scene_000"]["files"] == 6
    assert "max_proj" not in report["positions"]


@pytest.mark.parametrize("keep,expect", [
    # Light truncation: the page COUNT survives, and the file is even
    # larger than its raw pixel count thanks to IFD overhead - so a
    # size check sees nothing. Only decoding the last page catches it.
    (0.99, "damaged"),
    # Heavy truncation: tifffile reports FEWER pages, so the declared
    # count shrinks to match and the last page decodes fine. This one
    # surfaces as a shape split instead.
    (0.5, "killed mid-write"),
])
def test_a_damaged_file_is_reported_not_silently_indexed(
        tmp_path, keep, expect):
    """A run killed mid-write leaves a damaged file, and Bio-Formats
    fails the WHOLE series on one bad member — so it must be reported.
    Dropping the timepoint silently would be worse."""
    d = tmp_path / "exp" / "pos"
    d.mkdir(parents=True)
    _write_stack(d / "t0000_BF.tif", 1, planes=5)
    _write_stack(d / "t0001_BF.tif", 2, planes=5)
    victim = d / "t0001_BF.tif"
    data = victim.read_bytes()
    victim.write_bytes(data[: int(len(data) * keep)])

    report = write_companion(str(tmp_path / "exp"))
    assert any(expect in w for w in report["warnings"]), (
        f"damage not reported: {report['warnings']}"
    )


def test_REGRESSION_file_size_alone_does_not_detect_truncation(tmp_path):
    """Guard the guard. The obvious check — compare size against the
    declared pixel count — passes on a file truncated to 99%, because
    IFD overhead leaves it LARGER than its own pixels. Measured:
    11143 B on disk against 10240 B declared."""
    d = tmp_path / "pos"
    d.mkdir()
    _write_stack(d / "t0000_BF.tif", 1, planes=5)
    victim = d / "t0000_BF.tif"
    data = victim.read_bytes()
    victim.write_bytes(data[: int(len(data) * 0.99)])

    with tifffile.TiffFile(str(victim)) as tf:
        page = tf.pages[0]
        declared = (len(tf.pages) * page.imagelength * page.imagewidth
                    * np.dtype(page.dtype).itemsize)
    assert victim.stat().st_size >= declared, (
        "the size heuristic would have caught this, so the expensive "
        "last-page decode is unnecessary — simplify _probe"
    )
    # But the real check does catch it.
    rec = scan_position(str(d))[0]
    assert "damaged" in rec.note


def test_verification_can_be_switched_off(tmp_path):
    """Decoding a page of every file is fine for a live run behind the
    cache, but a one-off pass over a 200 GB archive may not want it."""
    d = tmp_path / "pos"
    d.mkdir()
    _write_stack(d / "t0000_BF.tif", 1, planes=5)
    victim = d / "t0000_BF.tif"
    data = victim.read_bytes()
    victim.write_bytes(data[: int(len(data) * 0.99)])
    assert scan_position(str(d), verify=False)[0].note == ''


# ------------------------------------------------------------- caching

def test_the_header_cache_avoids_reopening_known_files(experiment):
    """A live run refreshes once per timepoint. Without the cache that
    re-opens every file written so far — quadratic over a long run."""
    cache = {}
    write_companion(str(experiment), dry_run=True, header_cache=cache)
    assert len(cache) == 13

    opened = []
    real = tifffile.TiffFile

    class Spy(real):
        def __init__(self, path, *a, **kw):
            opened.append(path)
            super().__init__(path, *a, **kw)

    tifffile.TiffFile = Spy
    try:
        write_companion(str(experiment), dry_run=True,
                        header_cache=cache)
    finally:
        tifffile.TiffFile = real
    assert opened == [], f"re-opened {len(opened)} cached files"


def test_a_changed_file_is_re_read(experiment):
    """Keyed on size and mtime, so a rewritten file is not served
    stale from the cache."""
    cache = {}
    write_companion(str(experiment), dry_run=True, header_cache=cache)
    target = experiment / "scene_000" / "t0000_BF.tif"
    _write_stack(target, 5, planes=9)          # different plane count
    write_companion(str(experiment), dry_run=True, header_cache=cache)
    recs = scan_position(str(experiment / "scene_000"),
                         header_cache=cache)
    rec = next(r for r in recs if r.name == "t0000_BF.tif")
    assert rec.n_planes == 9, "served a stale header from the cache"


# --------------------------------------------------- backend integration

def test_backend_refreshes_the_companion_after_each_timepoint():
    """Not after each frame: a run killed at any moment should leave a
    companion describing every COMPLETE timepoint."""
    import inspect

    from tracking_tools.microscope_interface import mtb_backend

    src = inspect.getsource(mtb_backend.MicroscopeInterface_MTB
                            ._acquisition_loop)
    assert "refresh_companion" in src


def test_REGRESSION_the_save_fallback_writes_no_shaped_tag():
    """tifffile's default metadata stamps a 'shaped' JSON description,
    and that tag takes precedence over the OME companion for every
    Python reader — so one fallback frame would silently detach that
    timepoint from the series."""
    import inspect

    from tracking_tools.microscope_interface import mtb_backend

    src = inspect.getsource(mtb_backend.MicroscopeInterface_MTB._save)
    fallback = src[src.index("falling back"):]
    assert "metadata=None" in fallback, (
        "the fallback imwrite does not suppress tifffile's shaped tag"
    )
