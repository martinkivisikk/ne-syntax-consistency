"""Colours, fonts and sizes of the diagrams."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    """Drawing options. Sizes are in pixels.

    The SVG has its own background (`background`), so the default colours also
    work in dark notebook themes.
    """

    font_family: str = "Arial, Helvetica, 'Liberation Sans', sans-serif"
    word_font_size: float = 14
    label_font_size: float = 11  # deprel labels
    row_font_size: float = 11

    background: str = "white"
    text: str = "#222"
    muted: str = "#6b6b6b"
    arc: str = "#8a8a8a"
    box_fill: str = "#e7eef7"
    highlight_fill: str = "#ffe3c2"
    highlight_arc: str = "#d9480f"
    context_arc: str = "#1c7ed6"

    margin: float = 10
    word_padding: float = 16  # horizontal space added to each word's text width
    leg_gap: float = 8  # distance between arc legs on one word
    level_height: float = 22  # vertical distance between arc levels
    label_padding: float = 10  # minimal free bar on each side of a deprel label
    row_height: float = 15
    gap_width: float = 28  # width of a "…" column for hidden words
    arrow_size: float = 5
    stroke_width: float = 1
    highlight_stroke_width: float = 2.2
