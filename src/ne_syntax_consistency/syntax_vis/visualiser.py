"""`DependencyTreeVisualiser`: dependency trees of EstNLTK `Text` objects."""

from collections.abc import Sequence

from estnltk import Text
from IPython.display import display_html

from .extract import ExtractionConfig, Highlight, RowSpec, extract
from .model import window as apply_window
from .style import Style
from .svg import render_svg


class DependencyTreeVisualiser:
    """Draws the dependency trees of a `Text` as SVG arc diagrams.

    Configure once, then apply to many texts::

        vis = DependencyTreeVisualiser(
            syntax_layer="stanza_syntax",
            rows=[("morph_analysis", "partofspeech"), ("morph_analysis", "form")],
            span_layer="ner",
            span_label="nertag",
        )
        vis(text)                     # display all sentences in a notebook
        vis(text, highlight=text.ner[0], window=0)  # a fragment around an NE
        svg = vis.svg(text, sentence=0)

    Parameters
    ----------
    syntax_layer
        Layer with the attributes `id`, `head` (0 = root) and `deprel`; its spans
        are the words of the diagram.
    sentence_layer
        Layer of sentences; one diagram per sentence.
    rows
        Rows drawn under the words, each `(layer, attribute)`, `(label, function)`
        or `function`. Functions get the word's span of the syntax layer.
    span_layer, span_label
        Optional layer whose spans (e.g. named entities) are drawn as boxes over
        their words, labelled with the attribute `span_label`.
    ambiguous
        For ambiguous layers (e.g. `morph_analysis`): show the `"first"`
        annotation's value or `"all"` distinct values joined with `|`.
    missing
        Shown when a row has no span or value for a word.
    style
        Colours and sizes, see `Style`.
    """

    def __init__(
        self,
        syntax_layer: str = "stanza_syntax",
        rows: Sequence[RowSpec] = (),
        span_layer: str | None = None,
        span_label: str | None = None,
        sentence_layer: str = "sentences",
        ambiguous: str = "first",
        missing: str = "-",
        style: Style | None = None,
    ):
        if ambiguous not in ("first", "all"):
            raise ValueError(f"ambiguous must be 'first' or 'all', got {ambiguous!r}")
        self.config = ExtractionConfig(
            syntax_layer=syntax_layer,
            sentence_layer=sentence_layer,
            rows=tuple(rows),
            span_layer=span_layer,
            span_label=span_label,
            ambiguous=ambiguous,
            missing=missing,
        )
        self.style = style or Style()

    def svgs(
        self,
        text: Text,
        sentence: int | None = None,
        highlight: Highlight = None,
        window: int | None = None,
    ) -> list[str]:
        """SVG strings of all sentences, or of the one selected by `sentence` or
        `highlight`.

        `highlight` is a span (e.g. `text.ner[0]`), several spans (e.g.
        `text.words[3:5]` or a list of word spans) or word indices within
        `sentence`. `window` keeps only the highlight, `window` words on each side
        of every highlighted word and the heads the highlight attaches to;
        `window=0` draws just the fragment.
        """
        svgs = []
        for _, model in extract(text, self.config, sentence, highlight):
            if window is not None:
                model = apply_window(model, window)
            svgs.append(render_svg(model, self.style))
        return svgs

    def svg(
        self,
        text: Text,
        sentence: int | None = None,
        highlight: Highlight = None,
        window: int | None = None,
    ) -> str:
        """SVG string of one sentence (see `svgs` for the arguments).

        The sentence is `sentence`, the one containing `highlight`, or the only
        sentence of a one-sentence text.
        """
        if sentence is None and highlight is None:
            n = (
                len(text[self.config.sentence_layer])
                if (self.config.sentence_layer in text.layers)
                else 0
            )
            if n != 1:
                raise ValueError(
                    f"the Text has {n} sentences; give `sentence` or `highlight`"
                )
            sentence = 0
        return self.svgs(text, sentence, highlight, window)[0]

    def html_output(
        self,
        text: Text,
        sentence: int | None = None,
        highlight: Highlight = None,
        window: int | None = None,
    ) -> str:
        """HTML with the diagrams (see `svgs` for the arguments).

        Each diagram is in its own horizontally scrollable `div`.
        """
        return "\n".join(
            f'<div style="overflow-x:auto;margin-bottom:12px">{svg}</div>'
            for svg in self.svgs(text, sentence, highlight, window)
        )

    def __call__(
        self,
        text: Text,
        sentence: int | None = None,
        highlight: Highlight = None,
        window: int | None = None,
    ) -> None:
        """Display the diagrams in a notebook (see `svgs` for the arguments)."""
        display_html(self.html_output(text, sentence, highlight, window), raw=True)
