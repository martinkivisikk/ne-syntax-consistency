"""Plain sentence model that the layout and SVG code draw from.

The model does not depend on EstNLTK: `extract.py` builds it from a `Text`,
tests and other callers can build it directly.
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Row:
    """An attribute row drawn under the words: a label and one value per word."""

    label: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class Box:
    """A background box over the words `start .. end - 1` with an optional label."""

    start: int
    end: int
    label: str = ""


@dataclass(frozen=True)
class SentenceModel:
    """One sentence: words, dependency tree, rows, boxes and highlight.

    `heads[i]` is the 0-based index of word `i`'s head, or `None` for a root.
    `highlight` holds 0-based word indices. `visible` limits the drawing to some
    words (`None` = all); hidden stretches are drawn as "…" and arcs to hidden
    words are left out.
    """

    words: tuple[str, ...]
    heads: tuple[int | None, ...]
    deprels: tuple[str, ...]
    rows: tuple[Row, ...] = ()
    boxes: tuple[Box, ...] = ()
    highlight: frozenset[int] = field(default_factory=frozenset)
    visible: frozenset[int] | None = None

    def __post_init__(self) -> None:
        n = len(self.words)
        if len(self.heads) != n or len(self.deprels) != n:
            raise ValueError(
                f"words, heads and deprels must have the same length, got "
                f"{n}, {len(self.heads)} and {len(self.deprels)}"
            )
        for i, head in enumerate(self.heads):
            if head is not None and not 0 <= head < n:
                raise ValueError(f"head {head} of word {i} is outside the sentence")
            if head == i:
                raise ValueError(f"word {i} is its own head")
        for row in self.rows:
            if len(row.values) != n:
                raise ValueError(
                    f"row {row.label!r} has {len(row.values)} values for {n} words"
                )
        for box in self.boxes:
            if not 0 <= box.start < box.end <= n:
                raise ValueError(f"box {box} is outside the sentence")
        if any(not 0 <= i < n for i in self.highlight):
            raise ValueError("highlight contains word indices outside the sentence")
        if self.visible is not None:
            if not self.visible:
                raise ValueError("visible must contain at least one word")
            if any(not 0 <= i < n for i in self.visible):
                raise ValueError("visible contains word indices outside the sentence")

    @property
    def visible_words(self) -> tuple[int, ...]:
        """Indices of the words to draw, in sentence order."""
        if self.visible is None:
            return tuple(range(len(self.words)))
        return tuple(sorted(self.visible))

    @property
    def highlight_roots(self) -> frozenset[int]:
        """Highlighted words whose head is outside the highlight (or a root)."""
        return frozenset(
            i for i in self.highlight if self.heads[i] not in self.highlight
        )


def window(sentence: SentenceModel, size: int) -> SentenceModel:
    """Show only the highlight, `size` words on each side of every highlighted word
    and the heads of the highlight's roots.

    With `size=0` this draws a fragment: its words, "…" for the words between
    them, and the head each fragment root attaches to. Without a highlight the
    sentence is returned unchanged.
    """
    if size < 0:
        raise ValueError(f"window size must be >= 0, got {size}")
    if not sentence.highlight:
        return sentence
    n = len(sentence.words)
    keep = {
        j
        for i in sentence.highlight
        for j in range(max(i - size, 0), min(i + size + 1, n))
    }
    keep |= {
        head
        for i in sentence.highlight_roots
        if (head := sentence.heads[i]) is not None
    }
    return replace(sentence, visible=frozenset(keep))
