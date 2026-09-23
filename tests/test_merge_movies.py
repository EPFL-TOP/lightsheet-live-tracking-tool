"""tools/merge_movies.py: several start/stop runs -> one movie from t0."""
from __future__ import annotations

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import merge_movies  # noqa: E402


def _touch(path, content=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)


def _run(root, name, positions, tps, channels=("BF",)):
    for pos in positions:
        for t in tps:
            for ch in channels:
                _touch(os.path.join(root, name, pos, f"t{t:04d}_{ch}.tif"),
                       f"{name}/{t}/{ch}")
    return os.path.join(root, name)


def _files(folder):
    return sorted(os.listdir(folder))


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_two_runs_renumbered_from_zero(tmp_path):
    root = str(tmp_path)
    _run(root, "20260918_1000_Experiment", ["Position 1", "Position 2"],
         range(1, 4), channels=("BF", "Channel 1"))
    _run(root, "20260918_1400_Experiment", ["Position 1", "Position 2"],
         range(0, 2), channels=("BF", "Channel 1"))
    out = os.path.join(root, "merged")

    assert merge_movies.main([os.path.join(root, "2026*"), "-o", out]) == 0

    for pos in ("Position 1", "Position 2"):
        names = _files(os.path.join(out, pos))
        assert names == [f"t{t:04d}_{ch}.tif" for t in range(5)
                         for ch in ("BF", "Channel 1")]
        # run order kept, channel follows its timepoint
        assert _read(os.path.join(out, pos, "t0002_Channel 1.tif")) == \
            "20260918_1000_Experiment/3/Channel 1"
        assert _read(os.path.join(out, pos, "t0003_BF.tif")) == \
            "20260918_1400_Experiment/0/BF"

    with open(os.path.join(out, merge_movies.LOG_NAME)) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20
    # copy, not move
    assert os.path.exists(os.path.join(
        root, "20260918_1000_Experiment", "Position 1", "t0001_BF.tif"))


def test_non_image_files_not_carried(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], range(2))
    _touch(os.path.join(a, "P1", "embryo_tracking", "tracks.pkl"))
    _touch(os.path.join(a, "P1", "tracking_RoIs.json"))
    b = _run(root, "b", ["P1"], range(2))
    out = os.path.join(root, "merged")

    assert merge_movies.main([a, b, "-o", out]) == 0
    assert _files(os.path.join(out, "P1")) == [
        "t0000_BF.tif", "t0001_BF.tif", "t0002_BF.tif", "t0003_BF.tif"]


def test_gap_closed_and_incomplete_dropped(tmp_path, capsys):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], [0, 1, 3], channels=("BF", "GFP"))
    # run a was stopped mid-timepoint 4: only BF written
    _touch(os.path.join(a, "P1", "t0004_BF.tif"), "partial")
    b = _run(root, "b", ["P1"], range(2), channels=("BF", "GFP"))
    out = os.path.join(root, "merged")

    assert merge_movies.main([a, b, "-o", out, "--drop-incomplete"]) == 0
    names = _files(os.path.join(out, "P1"))
    assert names == [f"t{t:04d}_{ch}.tif" for t in range(5)
                     for ch in ("BF", "GFP")]
    assert _read(os.path.join(out, "P1", "t0002_BF.tif")) == "a/3/BF"
    log = capsys.readouterr().out
    assert "missing, gap closed" in log
    assert "dropped" in log


def test_position_folders_as_inputs(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["scene_001"], range(2))
    b = _run(root, "b", ["scene_001"], range(3))
    out = os.path.join(root, "merged")

    assert merge_movies.main([os.path.join(a, "scene_001"),
                              os.path.join(b, "scene_001"), "-o", out]) == 0
    assert len(_files(os.path.join(out, "scene_001"))) == 5


def test_dry_run_writes_nothing(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], range(2))
    b = _run(root, "b", ["P1"], range(2))
    out = os.path.join(root, "merged")

    assert merge_movies.main([a, b, "-o", out, "--dry-run"]) == 0
    assert not os.path.exists(out)


def test_refuses_to_overwrite(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], range(2))
    b = _run(root, "b", ["P1"], range(2))
    out = os.path.join(root, "merged")
    _touch(os.path.join(out, "P1", "t0000_BF.tif"), "keep me")

    assert merge_movies.main([a, b, "-o", out]) == 1
    assert _read(os.path.join(out, "P1", "t0000_BF.tif")) == "keep me"


def test_link_and_move(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], range(2))
    b = _run(root, "b", ["P1"], range(2))

    assert merge_movies.main([a, b, "-o", os.path.join(root, "l"),
                              "--link"]) == 0
    assert os.path.samefile(os.path.join(root, "l", "P1", "t0002_BF.tif"),
                            os.path.join(b, "P1", "t0000_BF.tif"))

    assert merge_movies.main([a, b, "-o", os.path.join(root, "m"),
                              "--move"]) == 0
    assert _files(os.path.join(a, "P1")) == []
    assert len(_files(os.path.join(root, "m", "P1"))) == 4


def test_width_grows(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], range(6))
    b = _run(root, "b", ["P1"], range(6))
    out = os.path.join(root, "merged")

    assert merge_movies.main([a, b, "-o", out, "--width", "1"]) == 0
    assert "t11_BF.tif" in _files(os.path.join(out, "P1"))
    assert "t00_BF.tif" in _files(os.path.join(out, "P1"))


def test_mixed_levels_rejected(tmp_path):
    root = str(tmp_path)
    a = _run(root, "a", ["P1"], range(2))
    b = _run(root, "b", ["P1"], range(2))
    assert merge_movies.main([a, os.path.join(b, "P1"),
                              "-o", os.path.join(root, "m")]) == 1


@pytest.mark.parametrize("name,ok", [
    ("t0012_BF.tif", True), ("t007.tif", True), ("t0001_Channel 1.tif", True),
    ("t0000_BF_z003.TIFF", True), ("max_t0000.tif", False),
    ("t0000_BF.png", False),
])
def test_name_pattern(name, ok):
    assert bool(merge_movies._TNAME.match(name)) is ok
