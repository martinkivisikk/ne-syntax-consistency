"""Dependency trees of sentence records as SVG arc diagrams.

An adapter from the Parquet sentence records (see data.SENTENCE_SCHEMA) to the
`syntax_vis` model; the drawing itself is done by `syntax_vis`.
"""

from ne_syntax_consistency.gspan import _case
from ne_syntax_consistency.syntax_vis import (
    Box,
    Row,
    SentenceModel,
    Style,
    render_svg,
    window,
)


def record_model(sentence: dict, span=()) -> SentenceModel:
    """`syntax_vis` model of a sentence record.

    Rows under the words: POS and case (`morph_analysis`); every NE is a box
    labelled with its type; `span` (0-based word indices) is highlighted.
    """
    return SentenceModel(
        words=tuple(sentence["words"]),
        # Stanza heads are 1-based (0 = root)
        heads=tuple(h - 1 if h else None for h in sentence["head"]),
        deprels=tuple(sentence["deprel"]),
        rows=(
            Row("pos", tuple(sentence["pos"])),
            Row("case", tuple(_case(f) or "-" for f in sentence["form"])),
        ),
        boxes=tuple(
            Box(min(idx), max(idx) + 1, tag)
            for tag, idx in zip(sentence["ne_tag"], sentence["ne_words"])
        ),
        highlight=frozenset(span),
    )


def record_svg(
    sentence: dict, span=(), window_size: int | None = None, style: Style | None = None
) -> str:
    """SVG of a sentence record with `span` highlighted.

    `window_size`: keep only the span, that many words around each span word and
    the heads of the span's roots (0 = just the span and its heads; None = the
    whole sentence).
    """
    model = record_model(sentence, span)
    if window_size is not None:
        model = window(model, window_size)
    return render_svg(model, style)
