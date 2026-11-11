"""Export random examples of the top span-tree patterns to CSV for labelling.

    uv run python scripts/export_examples.py --label type --name full_nochat

Reads results/span_trees/span_trees_<label>_<name>{.csv,_rare.csv,_spans.parquet}
(from scripts/span_trees.py), takes the --top most frequent patterns per NE type
with support >= --min-support, draws --n spans per pattern at random, at most
one per distinct span text (--uniform: plain uniform draw; see
ne_syntax_consistency.spans.sample_examples, the same draw as notebook 04 with
the same seed) and loads only those sentences from data/<sentences>.
ne_text_count is the number of the pattern's spans with that text, for weighting
labels back to span level.

Writes results/span_trees/examples_<label>_<name>.csv, one row per example,
ordered by type, pattern rank and draw, with empty columns for labelling:
ne (ok / type / boundary / not_ne / unclear: is the span a correct NE with the
correct type?), ne_fix (the correct type or span) and note. The parse is not
labelled. The span is marked [[...]] in `sentence`.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ne_syntax_consistency.spans import LABELS, sample_examples, top_patterns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "span_trees"
COLUMNS = ["text_id", "words", "head", "deprel", "ne_tag", "ne_words"]


def example_row(s: dict, ne_index: int) -> dict:
    """Readable context of one span: marked sentence, arcs, attachment, other NEs."""
    idx = sorted(s["ne_words"][ne_index])
    in_span = set(idx)
    words = s["words"]
    # Stanza heads are 1-based (0 = root); convert to 0-based, root -> None
    heads = [h - 1 if h else None for h in s["head"]]
    marked = list(words)
    marked[idx[0]] = "[[" + marked[idx[0]]
    marked[idx[-1]] = marked[idx[-1]] + "]]"
    return {
        # each span word: word/deprel>its head word
        "span_arcs": " ".join(
            f"{words[i]}/{s['deprel'][i]}>"
            + (words[heads[i]] if heads[i] is not None else "ROOT")
            for i in idx
        ),
        "root_heads": "; ".join(
            f"{words[i]} -{s['deprel'][i]}-> "
            + (words[heads[i]] if heads[i] is not None else "ROOT")
            for i in idx
            if heads[i] not in in_span
        ),
        "other_nes": "; ".join(
            " ".join(words[i] for i in sorted(w)) + "=" + t
            for j, (t, w) in enumerate(zip(s["ne_tag"], s["ne_words"]))
            if j != ne_index
        ),
        "sentence": " ".join(marked),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", choices=LABELS, default="type")
    parser.add_argument("--name", default="full_nochat", help="span_trees.py --name")
    parser.add_argument(
        "--sentences", default="full", help="sentence Parquet file or dir in data/"
    )
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--top", type=int, default=10, help="patterns per NE type")
    parser.add_argument("--n", type=int, default=4, help="examples per pattern")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--uniform", action="store_true", help="allow repeated span texts"
    )
    args = parser.parse_args()

    stem = RESULTS / f"span_trees_{args.label}_{args.name}"
    table = pd.concat(
        [pd.read_csv(f"{stem}.csv"), pd.read_csv(f"{stem}_rare.csv")],
        ignore_index=True,
    )
    spans = pq.read_table(
        f"{stem}_spans.parquet",
        columns=["text_id", "ne_index", "nertag", "pattern", "ne_text", "ext_deprel"],
        read_dictionary=["nertag", "pattern"],
    ).to_pandas()
    top = top_patterns(table, args.min_support, args.top)
    examples = sample_examples(spans, top, args.n, args.seed, distinct=not args.uniform)

    sentences = {
        s["text_id"]: s
        for s in pq.read_table(
            ROOT / "data" / args.sentences,
            columns=COLUMNS,
            filters=[("text_id", "in", sorted(set(examples["text_id"].tolist())))],
        ).to_pylist()
    }
    context = pd.DataFrame(
        example_row(sentences[r.text_id], r.ne_index) for r in examples.itertuples()
    )
    out = pd.concat([examples.reset_index(drop=True), context], axis=1)
    out["nertag"] = out["nertag"].astype(str)
    out["pattern"] = out["pattern"].astype(str)
    out = out.assign(ne="", ne_fix="", note="")[
        [
            "nertag",
            "rank",
            "pattern",
            "support",
            "share_of_type",
            "text_id",
            "ne_index",
            "ne_text",
            "ne_text_count",
            "ext_deprel",
            "root_heads",
            "span_arcs",
            "other_nes",
            "sentence",
            "ne",
            "ne_fix",
            "note",
        ]
    ]
    out = out.sort_values(["nertag", "rank"], kind="stable")

    path = RESULTS / f"examples_{args.label}_{args.name}.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; move it first so labels are not overwritten")
    out.to_csv(path, index=False)
    print(f"{len(out)} examples ({out['pattern'].nunique()} patterns) -> {path}")
    print(out.groupby("nertag").size().to_string())


if __name__ == "__main__":
    main()
