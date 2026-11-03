"""Whole internal dependency trees of NE spans.

Each NE span gets exactly one pattern: the tree formed by its words and the
arcs between them. Counting these gives the support of complete patterns only
(no sub-patterns), unlike gSpan, which counts every connected subgraph.

Notation: a word is written as its label; each dependent is a bracket that
starts with its deprel, placed on the side of its head where it occurs in the
sentence. Spans with several roots (words whose head is outside the span) are
forests; their trees are joined with " + " in word order.

    Mart Laar                    PER [flat PER]
    Eesti Posti Tartu osakond    [nmod [nmod ORG] ORG] [nmod ORG] ORG
"""

import collections
from collections.abc import Iterator

import pandas as pd

from ne_syntax_consistency.gspan import _case

LABELS = ("type", "pos", "pos_case")


def _label(sentence: dict, i: int, tag: str, label: str) -> str:
    if label == "type":
        return tag
    pos = sentence["pos"][i]
    if label == "pos":
        return pos
    return f"{pos}:{_case(sentence['form'][i]) or '-'}"


def span_tree(sentence: dict, idx, tag: str, label: str = "type") -> dict:
    """The internal tree of one span (word indices `idx`) as a pattern string."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    span = sorted(idx)
    in_span = set(span)
    # Stanza ids are 1-based; convert heads to 0-based indices, root -> None
    heads = [h - 1 if h else None for h in sentence["head"]]
    deprels = sentence["deprel"]
    children = collections.defaultdict(list)
    roots = []
    for i in span:
        if heads[i] in in_span:
            children[heads[i]].append(i)
        else:
            roots.append(i)

    def render(i: int) -> str:
        left = [f"[{deprels[c]} {render(c)}]" for c in children[i] if c < i]
        right = [f"[{deprels[c]} {render(c)}]" for c in children[i] if c > i]
        return " ".join([*left, _label(sentence, i, tag, label), *right])

    words = sentence["words"]
    return {
        "pattern": " + ".join(render(r) for r in roots),
        "n_words": len(span),
        "n_roots": len(roots),
        # relation of each root to the word outside the span
        "ext_deprel": "+".join(deprels[r] for r in roots),
        "ne_text": " ".join(words[i] for i in span),
    }


def span_trees(sentence: dict, label: str = "type") -> Iterator[dict]:
    """One row per NE span in a sentence record (see data.SENTENCE_SCHEMA)."""
    for ne_index, (tag, idx) in enumerate(
        zip(sentence["ne_tag"], sentence["ne_words"])
    ):
        yield {
            "text_id": sentence["text_id"],
            "ne_index": ne_index,
            "nertag": tag,
            **span_tree(sentence, idx, tag, label),
        }


def support_table(spans: pd.DataFrame) -> pd.DataFrame:
    """Support per NE type and pattern from span_trees rows, most frequent first."""
    table = (
        spans.groupby(["nertag", "pattern"])
        .agg(
            support=("pattern", "size"),
            n_words=("n_words", "first"),
            n_roots=("n_roots", "first"),
            example=("ne_text", "first"),
        )
        .reset_index()
    )
    table["share_of_type"] = table["support"] / table.groupby("nertag")[
        "support"
    ].transform("sum")
    return table.sort_values(["nertag", "support"], ascending=[True, False])
