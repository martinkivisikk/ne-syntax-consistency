"""Dependency tree visualiser for EstNLTK `Text` objects.

Copied from ~/projects/estnltk-syntax-vis (`src/estnltk_syntax_vis`,
commit 7cdbb0c); only this docstring differs. Change it there and copy it over again; for sentence records
of this project use `ne_syntax_consistency.viz`.
"""

from .model import Box, Row, SentenceModel, window
from .style import Style
from .svg import render_svg
from .visualiser import DependencyTreeVisualiser

__all__ = [
    "Box",
    "DependencyTreeVisualiser",
    "Row",
    "SentenceModel",
    "Style",
    "render_svg",
    "window",
]
