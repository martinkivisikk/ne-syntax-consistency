"""Horizontal layout of a sentence diagram: columns, arc levels and leg positions.

Every arc is drawn as a horizontal bar at its nesting level with two vertical
legs down to the head and the dependent. Arcs that end at the same word get
separate legs, ordered so that legs and bars do not cross in projective trees.
"""

import unicodedata
from dataclasses import dataclass

from .model import SentenceModel
from .style import Style

# Advance widths of Arial / Liberation Sans in 1/1000 em (printable ASCII)
_ARIAL_CHARS = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)
_ARIAL_WIDTHS = """
278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 556 556 556 556
556 556 556 556 556 556 278 278 584 584 584 556 1015 667 667 722 722 667 611 778
722 278 500 667 556 833 722 778 667 778 722 667 611 722 667 944 667 667 611 278
278 278 469 556 333 556 556 500 556 556 278 556 556 222 222 500 222 833 556 556
556 556 333 500 278 556 500 722 500 500 500 334 260 334 584
"""
_ARIAL = dict(zip(_ARIAL_CHARS, map(int, _ARIAL_WIDTHS.split()), strict=True))


def text_width(text: str, font_size: float, bold: bool = False) -> float:
    """Approximate rendered width of `text` in a sans-serif font."""
    units = 0
    for char in text:
        base = unicodedata.normalize("NFD", char)[0]  # õ -> o, š -> s
        units += _ARIAL.get(base, 600)
    return units / 1000 * font_size * (1.07 if bold else 1.0)


@dataclass(frozen=True)
class Arc:
    """A dependency arc between word indices `head` and `dependent`."""

    head: int
    dependent: int
    deprel: str
    level: int
    x_head: float  # x of the leg at the head
    x_dependent: float  # x of the leg at the dependent (arrow end)
    x_label: float  # centre of the deprel label on the bar

    @property
    def left(self) -> int:
        return min(self.head, self.dependent)

    @property
    def right(self) -> int:
        return max(self.head, self.dependent)


@dataclass(frozen=True)
class Column:
    """Horizontal extent of one word, or of a "…" gap for hidden words."""

    x: float  # left edge
    width: float
    word: int | None  # word index, `None` for a gap

    @property
    def centre(self) -> float:
        return self.x + self.width / 2


@dataclass(frozen=True)
class Layout:
    columns: tuple[Column, ...]  # left to right, gaps included
    column_of: dict[int, Column]  # visible word -> its column
    arcs: tuple[Arc, ...]  # in dependent order, roots excluded
    root_x: dict[int, float]  # root word -> x of its root arrow
    max_level: int
    width: float  # width of the word area (without margins and row labels)


def arc_levels(arcs: list[tuple[int, int]]) -> list[int]:
    """Nesting level of each arc, given as `(head, dependent)` word index pairs.

    Shorter arcs are placed first. An arc's level is one above every already
    placed arc it overlaps horizontally (contains or crosses); arcs that only
    share an end word can share a level, as their legs are apart.
    """
    order = sorted(range(len(arcs)), key=lambda k: abs(arcs[k][0] - arcs[k][1]))
    levels = [0] * len(arcs)
    placed: list[tuple[int, int, int]] = []  # (left, right, level)
    for k in order:
        lo, hi = sorted(arcs[k])
        below = [lvl for (a, b, lvl) in placed if a < hi and lo < b]
        levels[k] = 1 + max(below, default=0)
        placed.append((lo, hi, levels[k]))
    return levels


def leg_offsets(
    n_columns: int, arcs: list[tuple[int, int]], levels: list[int], roots: set[int]
) -> tuple[dict[tuple[int, int], float], dict[int, float], list[int]]:
    """Leg positions on each word, as multiples of the leg gap from the centre.

    Arcs and roots are given in column positions. Returns `(offsets,
    root_offsets, leg_counts)`: `offsets[(k, column)]` for the end of arc `k` at
    `column`, `root_offsets[column]` for root arrows and the number of legs in
    each column. Left to right, a word's legs are: arcs going left from
    the lowest to the highest, the root arrow, arcs going right from the highest
    to the lowest. Higher arcs thus stand nearer the centre and their bars pass
    over the legs of lower arcs instead of crossing them.
    """
    ends: list[list[tuple[tuple, object]]] = [[] for _ in range(n_columns)]
    for k, (h, d) in enumerate(arcs):
        for word, other in ((h, d), (d, h)):
            distance = abs(word - other)
            if other < word:
                key = (0, levels[k], distance)
            else:
                key = (2, -levels[k], -distance)
            ends[word].append((key, k))
    for word in roots:
        ends[word].append(((1,), None))

    offsets: dict[tuple[int, int], float] = {}
    root_offsets: dict[int, float] = {}
    counts = []
    for word, word_ends in enumerate(ends):
        word_ends.sort(key=lambda end: end[0])
        n = len(word_ends)
        counts.append(n)
        for position, (_, k) in enumerate(word_ends):
            offset = position - (n - 1) / 2
            if k is None:
                root_offsets[word] = offset
            else:
                offsets[(k, word)] = offset
    return offsets, root_offsets, counts


def _label_x(x1: float, x2: float, crossing: list[float], need: float) -> float:
    """Centre of the longest bar segment between `x1` and `x2` not crossed by a leg.

    Falls back to the bar centre when that segment is shorter than `need`.
    """
    lo, hi = sorted((x1, x2))
    points = [lo, *sorted(x for x in crossing if lo < x < hi), hi]
    a, b = max(zip(points, points[1:], strict=False), key=lambda seg: seg[1] - seg[0])
    return (a + b) / 2 if b - a >= need else (lo + hi) / 2


def layout(sentence: SentenceModel, style: Style) -> Layout:
    """Compute columns, arc levels and leg x positions for `sentence`.

    Works on column positions: the visible words in order, with a gap column
    wherever words are hidden. Arcs are drawn only between visible words.
    """
    n_words = len(sentence.words)
    visible = sentence.visible_words
    slots: list[int | None] = []  # word index per column, None for a gap
    for k, i in enumerate(visible):
        if i > (visible[k - 1] if k else -1) + 1:
            slots.append(None)
        slots.append(i)
    if visible and visible[-1] < n_words - 1:
        slots.append(None)
    pos = {word: p for p, word in enumerate(slots) if word is not None}
    n = len(slots)

    deps = [d for d in visible if (h := sentence.heads[d]) is not None and h in pos]
    pairs = [(pos[sentence.heads[d]], pos[d]) for d in deps]  # type: ignore[index]
    roots = {pos[d] for d in visible if sentence.heads[d] is None}
    levels = arc_levels(pairs)
    offsets, root_offsets, counts = leg_offsets(n, pairs, levels, roots)

    # Column widths: the widest of the word, its rows and its legs
    widths = []
    for p, i in enumerate(slots):
        if i is None:
            widths.append(style.gap_width)
            continue
        bold = i in sentence.highlight
        texts = [text_width(sentence.words[i], style.word_font_size, bold)]
        texts += [text_width(r.values[i], style.row_font_size) for r in sentence.rows]
        legs = (counts[p] - 1) * style.leg_gap
        widths.append(max(max(texts), legs) + style.word_padding)
    # Box labels are centred under their boxes; widen boxes that are too narrow
    for box in sentence.boxes:
        cols = [pos[i] for i in range(box.start, box.end) if i in pos]
        if not cols:
            continue
        need = text_width(box.label, style.row_font_size) + style.word_padding
        have = sum(widths[cols[0] : cols[-1] + 1])
        if need > have:
            for p in cols:
                widths[p] += (need - have) / len(cols)

    # Extra space between columns `p` and `p + 1`, so deprel labels fit on bars
    gaps = [0.0] * max(n - 1, 0)

    def centres() -> list[float]:
        xs, x = [], 0.0
        for p, w in enumerate(widths):
            xs.append(x + w / 2)
            x += w + (gaps[p] if p < n - 1 else 0)
        return xs

    def leg_x(xs: list[float], k: int) -> tuple[float, float]:
        h, d = pairs[k]
        return (
            xs[h] + offsets[(k, h)] * style.leg_gap,
            xs[d] + offsets[(k, d)] * style.leg_gap,
        )

    def label_need(k: int) -> float:
        deprel = sentence.deprels[deps[k]]
        return text_width(deprel, style.label_font_size) + 2 * style.label_padding

    for k in sorted(range(len(pairs)), key=lambda k: abs(pairs[k][0] - pairs[k][1])):
        x_h, x_d = leg_x(centres(), k)
        missing = label_need(k) - abs(x_d - x_h)
        if missing > 0:
            lo, hi = sorted(pairs[k])
            for p in range(lo, hi):
                gaps[p] += missing / (hi - lo)

    xs = centres()
    columns = tuple(
        Column(x - w / 2, w, word) for x, w, word in zip(xs, widths, slots, strict=True)
    )
    root_x = {slots[r]: xs[r] + root_offsets[r] * style.leg_gap for r in roots}
    legs = [leg_x(xs, k) for k in range(len(pairs))]
    arcs = []
    for k, d in enumerate(deps):
        # Legs that cross this bar: legs of higher arcs and root arrows
        crossing = [
            x for j, leg in enumerate(legs) if levels[j] > levels[k] for x in leg
        ]
        crossing += root_x.values()
        arcs.append(
            Arc(
                head=sentence.heads[d],  # type: ignore[arg-type]
                dependent=d,
                deprel=sentence.deprels[d],
                level=levels[k],
                x_head=legs[k][0],
                x_dependent=legs[k][1],
                x_label=_label_x(*legs[k], crossing, label_need(k)),
            )
        )
    width = columns[-1].x + columns[-1].width if columns else 0.0
    column_of = {c.word: c for c in columns if c.word is not None}
    return Layout(
        columns, column_of, tuple(arcs), root_x, max(levels, default=0), width
    )
