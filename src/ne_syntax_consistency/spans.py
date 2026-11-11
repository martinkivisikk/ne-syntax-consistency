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
    return _with_shares(table)


def merge_support_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine support_table results of consecutive chunks of spans.

    Gives the same table as support_table on all spans at once: supports are
    added up and n_words, n_roots and example come from the first chunk.
    """
    table = (
        pd.concat(tables)
        .groupby(["nertag", "pattern"], sort=False)
        .agg(
            support=("support", "sum"),
            n_words=("n_words", "first"),
            n_roots=("n_roots", "first"),
            example=("example", "first"),
        )
        .reset_index()
    )
    return _with_shares(table)


def _with_shares(table: pd.DataFrame) -> pd.DataFrame:
    table["share_of_type"] = table["support"] / table.groupby("nertag")[
        "support"
    ].transform("sum")
    # stable sort keeps ties in (nertag, pattern) order, as support_table does
    return table.sort_values(
        ["nertag", "support", "pattern"], ascending=[True, False, True]
    ).reset_index(drop=True)


def top_patterns(table: pd.DataFrame, min_support: int, top_n: int) -> pd.DataFrame:
    """The top_n most frequent patterns per NE type with support >= min_support."""
    top = (
        table[table["support"] >= min_support]
        .sort_values(["nertag", "support"], ascending=[True, False])
        .groupby("nertag", group_keys=False)
        .head(top_n)
    )
    return top.assign(rank=top.groupby("nertag").cumcount() + 1)


def sample_examples(
    spans: pd.DataFrame, top: pd.DataFrame, n: int, seed: int, distinct: bool = True
) -> pd.DataFrame:
    """Up to n random spans for each (nertag, pattern) in top.

    With distinct=True, each span text (ne_text) is drawn at most once per
    pattern, so one very frequent name (e.g. "/ EMÜ" from EU directive numbers)
    cannot fill a pattern's examples; spans are shuffled and the first span of
    each text is kept, so texts are drawn in proportion to their frequency but
    without replacement (frequent names appear once, next to rare ones).
    ne_text_count is the number of the pattern's spans with that text, to
    weight labels back to span level. With distinct=False, spans are drawn uniformly. The same
    arguments give the same examples (the export script and notebook 04 agree).
    """
    examples = []
    for p in top.itertuples():
        m = spans[(spans["nertag"] == p.nertag) & (spans["pattern"] == p.pattern)]
        counts = m["ne_text"].value_counts()
        if distinct:
            drawn = (
                m.sample(frac=1, random_state=seed).drop_duplicates("ne_text").head(n)
            )
        else:
            drawn = m.sample(n=min(n, len(m)), random_state=seed)
        examples.append(
            drawn.assign(
                rank=p.rank,
                support=p.support,
                share_of_type=p.share_of_type,
                ne_text_count=drawn["ne_text"].map(counts).to_numpy(),
            )
        )
    return pd.concat(examples, ignore_index=True)
