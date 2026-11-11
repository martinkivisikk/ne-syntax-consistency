"""Extraction of sentence models from EstNLTK `Text` objects.

Words are the spans of the syntax layer. Other layers (rows, boxes, highlight)
are matched to them by text position, so they only need to cover the same words,
not be the same layer object.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from estnltk import Text

from .model import Box, Row, SentenceModel

SYNTAX_ATTRIBUTES = ("id", "head", "deprel")

RowSpec = tuple[str, str] | tuple[str, Callable[[Any], Any]] | Callable[[Any], Any]
"""A row under the words: `(layer, attribute)`, `(label, function)` or `function`.

Functions get the word's span of the syntax layer and return the value to show.
"""

Highlight = Any  # a span, an iterable of spans, or an iterable of word indices


@dataclass(frozen=True)
class ExtractionConfig:
    syntax_layer: str = "stanza_syntax"
    sentence_layer: str = "sentences"
    rows: tuple[RowSpec, ...] = ()
    span_layer: str | None = None
    span_label: str | None = None
    ambiguous: str = "first"  # "first" or "all" annotations of ambiguous layers
    missing: str = "-"  # shown for a missing span or value


def _layer(text: Text, name: str, what: str):
    if name not in text.layers:
        available = ", ".join(sorted(text.layers)) or "none"
        raise ValueError(
            f"{what} layer {name!r} is missing from the Text (layers: {available})"
        )
    return text[name]


def _positions(span) -> list[tuple[int, int]]:
    """Text positions covered by a span: its parts for enveloping spans."""
    parts = getattr(span, "spans", None)
    if parts is not None:
        return [p for part in parts for p in _positions(part)]
    return [(span.start, span.end)]


def _is_span(obj) -> bool:
    return hasattr(obj, "start") and hasattr(obj, "end") and hasattr(obj, "base_span")


class _SentenceIndex:
    """Positions of the sentences and syntax words of a `Text`."""

    def __init__(self, text: Text, config: ExtractionConfig):
        syntax = _layer(text, config.syntax_layer, "Syntax")
        missing = [a for a in SYNTAX_ATTRIBUTES if a not in syntax.attributes]
        if missing:
            raise ValueError(
                f"syntax layer {config.syntax_layer!r} lacks the attribute(s) "
                f"{', '.join(missing)}"
            )
        self.sentences = list(_layer(text, config.sentence_layer, "Sentence"))
        self.sentence_starts = [s.start for s in self.sentences]
        self.words = list(syntax)
        self.word_starts = [w.start for w in self.words]

    def words_of(self, k: int) -> list:
        sentence = self.sentences[k]
        lo = bisect_left(self.word_starts, sentence.start)
        hi = bisect_left(self.word_starts, sentence.end)
        return self.words[lo:hi]

    def sentence_of(self, start: int, end: int) -> int:
        """Index of the sentence that contains the text range `start .. end`."""
        k = bisect_right(self.sentence_starts, start) - 1
        if k < 0 or end > self.sentences[k].end:
            raise ValueError(
                f"text range {start}..{end} does not lie inside a single sentence"
            )
        return k


def _value(span, attribute: str, config: ExtractionConfig) -> str:
    if span is None:
        return config.missing
    values = []
    for annotation in span.annotations:
        value = annotation[attribute]
        if value is not None and str(value) not in values:
            values.append(str(value))
    if not values:
        return config.missing
    return values[0] if config.ambiguous == "first" else "|".join(values)


def _row(text: Text, spec: RowSpec, words: list, config: ExtractionConfig) -> Row:
    if callable(spec):
        label, function = getattr(spec, "__name__", ""), spec
    elif isinstance(spec, tuple) and len(spec) == 2 and callable(spec[1]):
        label, function = spec
    elif isinstance(spec, tuple) and len(spec) == 2:
        layer_name, attribute = spec
        layer = _layer(text, layer_name, "Row")
        if attribute not in layer.attributes:
            raise ValueError(f"layer {layer_name!r} has no attribute {attribute!r}")
        values = [_value(layer.get(w.base_span), attribute, config) for w in words]
        return Row(attribute, tuple(values))
    else:
        raise ValueError(
            f"a row must be (layer, attribute), (label, function) or a function, "
            f"got {spec!r}"
        )
    values = []
    for word in words:
        value = function(word)
        values.append(config.missing if value is None else str(value))
    return Row(label, tuple(values))


def _boxes(text: Text, words: list, config: ExtractionConfig) -> tuple[Box, ...]:
    if config.span_layer is None:
        return ()
    layer = _layer(text, config.span_layer, "Span")
    if config.span_label is not None and config.span_label not in layer.attributes:
        raise ValueError(
            f"layer {config.span_layer!r} has no attribute {config.span_label!r}"
        )
    if not words:
        return ()
    first, last = words[0].start, words[-1].end
    boxes = []
    for span in layer:
        if span.end <= first or span.start >= last:
            continue
        inside = [
            i
            for i, w in enumerate(words)
            if span.start <= w.start and w.end <= span.end
        ]
        if not inside:
            continue
        label = (
            "" if config.span_label is None else _value(span, config.span_label, config)
        )
        boxes.append(Box(inside[0], inside[-1] + 1, label))
    return tuple(boxes)


def _model(text: Text, index: _SentenceIndex, k: int, config: ExtractionConfig):
    words = index.words_of(k)
    position = {}
    for i, word in enumerate(words):
        word_id = word.annotations[0]["id"]
        if word_id in position:
            raise ValueError(f"sentence {k} has two words with id {word_id!r}")
        position[word_id] = i
    heads: list[int | None] = []
    for word in words:
        head = word.annotations[0]["head"]
        if head == 0:
            heads.append(None)
        elif head in position:
            heads.append(position[head])
        else:
            raise ValueError(
                f"head {head!r} of word {word.text!r} (sentence {k}) is not the id "
                f"of a word in the same sentence"
            )
    return SentenceModel(
        words=tuple(w.text for w in words),
        heads=tuple(heads),
        deprels=tuple(str(w.annotations[0]["deprel"]) for w in words),
        rows=tuple(_row(text, spec, words, config) for spec in config.rows),
        boxes=_boxes(text, words, config),
    )


def _highlight_words(
    highlight: Highlight, index: _SentenceIndex, sentence: int | None
) -> tuple[int, frozenset[int]]:
    """Resolve a highlight to `(sentence, word indices in that sentence)`."""
    items = [highlight] if _is_span(highlight) else list(highlight)
    if not items:
        raise ValueError("highlight is empty")
    if all(isinstance(i, int) for i in items):
        if sentence is None:
            raise ValueError("give `sentence` when the highlight is word indices")
        n = len(index.words_of(sentence))
        if any(not 0 <= i < n for i in items):
            raise ValueError(f"highlight indices must be in 0..{n - 1}")
        return sentence, frozenset(items)
    if not all(_is_span(i) for i in items):
        raise ValueError("highlight must be a span, spans or word indices")

    positions = [p for item in items for p in _positions(item)]
    found = index.sentence_of(
        min(s for s, _ in positions), max(e for _, e in positions)
    )
    if sentence is not None and sentence != found:
        raise ValueError(f"the highlight is in sentence {found}, not in {sentence}")
    words = index.words_of(found)
    chosen = frozenset(
        i
        for i, w in enumerate(words)
        if any(s <= w.start and w.end <= e for s, e in positions)
    )
    if not chosen:
        raise ValueError("the highlight covers no word of the syntax layer")
    return found, chosen


def extract(
    text: Text,
    config: ExtractionConfig,
    sentence: int | None = None,
    highlight: Highlight = None,
) -> list[tuple[int, SentenceModel]]:
    """Sentence models of `text` as `(sentence index, model)` pairs.

    Without `sentence` and `highlight` all sentences are returned; otherwise
    only the given sentence or the one that contains the highlight.
    """
    index = _SentenceIndex(text, config)
    n = len(index.sentences)
    if sentence is not None and not -n <= sentence < n:
        raise ValueError(f"sentence {sentence} is out of range (the Text has {n})")
    if sentence is not None:
        sentence %= n
    chosen: frozenset[int] = frozenset()
    if highlight is not None:
        sentence, chosen = _highlight_words(highlight, index, sentence)
    indices: Sequence[int] = range(n) if sentence is None else [sentence]
    models = []
    for k in indices:
        model = _model(text, index, k, config)
        if chosen:
            model = replace(model, highlight=chosen)
        models.append((k, model))
    return models
