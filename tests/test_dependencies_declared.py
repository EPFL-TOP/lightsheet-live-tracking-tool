"""Every third-party import under tracking_tools must be declared.

Motivation: on 2026-09-04 a Zeiss bring-up hit three separate missing
dependencies in a row, each discovered only when the tracker was
constructed — after positions had been captured, a full series
acquired, and ROIs drawn:

    ERROR No module named 'torch'
    ERROR No module named 'torchvision'
    ... and watchdog was silently degrading the ROI file watcher

Each cost a full acquisition cycle to find. This test walks the AST of
every module under tracking_tools and asserts each third-party import
is either pinned in a requirements file or explicitly listed below as
an optional backend dependency. It turns "crash, install, repeat" into
one check.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Provided by a vendor SDK or a specific backend, deliberately NOT in
# requirements.txt — each maps to hardware not every install has.
OPTIONAL: dict[str, str] = {
    # pythonnet, for the Zeiss MTB backend
    "clr": "pythonnet (MTB backend)",
    "System": "pythonnet/.NET (MTB backend)",
    "ZEISS": "Zeiss MTBApi.dll (MTB backend)",
    # Micro-Manager backend — pinned in requirements-mm.txt
    "pymmcore_plus": "requirements-mm.txt",
    "useq": "requirements-mm.txt",
    # ZEN gRPC backend, kept for the LSM/Lightsheet 7 path
    "zen_api": "Zeiss ZEN API SDK (ZEN backend)",
    "grpclib": "ZEN backend",
    "pylibCZIrw": "Zeiss CZI reader (ZEN backend)",
    "pymcs": "Viventis LS1 SDK (LS1 backend)",
}

# Import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "skimage": "scikit-image",
    "imaging_server_kit": "imaging-server-kit",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
}


def _declared_packages() -> set[str]:
    """Distribution names actually required, comments excluded.

    Comments must be stripped: an earlier version substring-matched the
    raw file, so a package merely NAMED in a comment counted as
    declared. Deleting `torchvision` from the requirement list left the
    word in an explanatory comment and the test still passed — a guard
    that cannot fail is worse than no guard.
    """
    names: set[str] = set()
    for fname in ("requirements.txt", "requirements-mm.txt"):
        path = _ROOT / fname
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()      # drop comments
            if not line or line.startswith("-"):     # -r, --index-url
                continue
            # Strip extras, version specifiers and environment markers.
            for sep in (";", "==", ">=", "<=", "~=", "!=", ">", "<",
                        "["):
                line = line.split(sep, 1)[0]
            line = line.strip()
            if line:
                names.add(line.lower())
    return names


def _requirement_text() -> str:
    """Kept for the per-package assertions; comment-free."""
    return "\n".join(sorted(_declared_packages()))


def _third_party_imports() -> dict[str, set[str]]:
    """Top-level third-party module -> files that import it."""
    stdlib = set(sys.stdlib_module_names)
    local = {"tracking_tools", "interactive_tools", "training_tools",
             "tools", "tests"}
    found: dict[str, set[str]] = {}

    for path in (_ROOT / "tracking_tools").rglob("*.py"):
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="ignore")
            )
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:            # relative import
                    continue
                names = [node.module or ""]
            for mod in names:
                top = mod.split(".")[0]
                if not top or top in stdlib or top in local:
                    continue
                rel = path.relative_to(_ROOT).as_posix()
                found.setdefault(top, set()).add(rel)
    return found


def test_every_third_party_import_is_declared_or_optional():
    declared = _declared_packages()
    imports = _third_party_imports()

    undeclared = []
    for mod, files in sorted(imports.items()):
        if mod in OPTIONAL:
            continue
        dist = IMPORT_TO_DIST.get(mod, mod).lower()
        # Exact membership: substring matching let a package named only
        # in a comment count as declared.
        if dist in declared or mod.lower() in declared:
            continue
        undeclared.append(f"{mod} (imported by {sorted(files)[0]})")

    assert not undeclared, (
        "third-party imports missing from requirements*.txt — these "
        "surface as 'No module named X' only when the code path runs:\n  "
        + "\n  ".join(undeclared)
    )


@pytest.mark.parametrize("pkg", ["torch", "torchvision", "scipy",
                                 "tifffile", "watchdog"])
def test_tracker_dependencies_are_pinned(pkg):
    """The tracker's own imports, named individually so a regression
    points straight at the package that went missing."""
    assert pkg in _declared_packages(), (
        f"{pkg} is imported by the tracker but not declared as a "
        f"requirement (a mention in a comment does not count)"
    )


def test_optional_backends_are_documented_not_silently_missing():
    """Anything intentionally absent from requirements must be listed
    in OPTIONAL with a reason, so 'not declared' never means
    'forgotten'."""
    for mod, reason in OPTIONAL.items():
        assert reason, f"{mod} has no explanation"


def test_declared_packages_ignores_comments():
    """The guard must not credit a package named only in a comment."""
    declared = _declared_packages()
    assert "torch" in declared
    assert "torchvision" in declared
    # Words that appear only in prose in requirements.txt.
    assert "imported" not in declared
    assert "cuda" not in declared
    assert "install" not in declared


def test_scan_actually_finds_imports():
    """Guard the guard: an AST walk that silently matched nothing would
    make this whole file vacuously pass."""
    imports = _third_party_imports()
    assert "numpy" in imports, "scan found no numpy — walker is broken"
    assert len(imports) > 5, f"suspiciously few imports: {imports}"
