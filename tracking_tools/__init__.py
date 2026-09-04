"""Tracking tools for live microscope drift correction.

Deliberately empty of imports: every subpackage (microscope_interface,
tracker, tracking_runner, ...) pulls in different optional
dependencies — pythonnet for Zeiss MTB, pymmcore-plus for
Micro-Manager, torch for some trackers — and importing them here would
make the whole package unimportable whenever any one of them is
missing.

This file exists so `tracking_tools` is a REGULAR package rather than
an implicit namespace package. Every subpackage already had an
__init__.py while this one did not, which left the top level resolving
only when the repo root happened to be on sys.path at the moment of
import. That is fragile under `panel serve`, which puts the served
script's own directory on sys.path instead of the repo root.
"""
