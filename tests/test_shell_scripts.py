"""Shipped shell scripts must parse on the machine that runs them.

Windows PowerShell 5.1 reads a `.ps1` with no byte-order mark as ANSI,
not UTF-8. A single multi-byte character therefore arrives as several
bytes of nonsense, terminates whatever string it sits in, and cascades
into a wall of unrelated parse errors elsewhere in the file:

    Unexpected token 'use' in expression or statement.
    Missing type name after '['.
    The '<' operator is reserved for future use.
    Missing closing ')' in expression.

One em dash in a comment produced all of the above and made the
hardware audit unusable. These checks are cheap and catch it here
rather than in front of the microscope.
"""
from __future__ import annotations

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PS1 = sorted(_ROOT.rglob("*.ps1"))


def test_there_are_powershell_scripts_to_check():
    """Guard the guard: an empty glob would make this file vacuous."""
    assert _PS1, "no .ps1 files found — has the glob broken?"


@pytest.mark.parametrize(
    "path", _PS1, ids=[p.name for p in _PS1]
)
def test_powershell_script_is_pure_ascii(path):
    raw = path.read_bytes()
    offenders = []
    line = 1
    for byte in raw:
        if byte == 0x0A:
            line += 1
        elif byte > 0x7F:
            offenders.append(line)
    assert not offenders, (
        f"{path.name} has non-ASCII bytes on line(s) "
        f"{sorted(set(offenders))[:10]}. Windows PowerShell 5.1 reads a "
        f"BOM-less .ps1 as ANSI, so these corrupt the parse. Use ASCII "
        f"punctuation: '-' not an em dash, '->' not an arrow."
    )


@pytest.mark.parametrize(
    "path", _PS1, ids=[p.name for p in _PS1]
)
def test_powershell_braces_are_balanced(path):
    src = path.read_text(encoding="ascii")
    assert src.count("{") == src.count("}"), (
        f"{path.name}: unbalanced braces"
    )


@pytest.mark.parametrize(
    "path", _PS1, ids=[p.name for p in _PS1]
)
def test_powershell_here_strings_are_closed(path):
    src = path.read_text(encoding="ascii")
    assert src.count('@"') == src.count('"@'), (
        f"{path.name}: unbalanced here-string delimiters"
    )
    assert src.count("<#") == src.count("#>"), (
        f"{path.name}: unbalanced block comment"
    )


@pytest.mark.parametrize(
    "path", _PS1, ids=[p.name for p in _PS1]
)
def test_powershell_avoids_nested_double_quotes_in_subexpressions(path):
    """`"... $(if($x){"y"}) ..."` parses in 5.1 but is needlessly
    fragile, and it sat at one of the reported error positions. Build
    the fragment in a variable instead."""
    src = path.read_text(encoding="ascii")
    offenders = [
        i for i, line in enumerate(src.splitlines(), 1)
        if '$(if(' in line.replace(" ", "")
    ]
    assert not offenders, (
        f"{path.name}: nested quoting inside $(...) on line(s) "
        f"{offenders}"
    )
