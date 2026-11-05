"""Export multi-root NE spans of a sample to CSV for labelling.

    uv run python scripts/export_multiroot.py data/sample_200000_s1.parquet
    uv run python scripts/export_multiroot.py data/sample_200000_s1.parquet --per-type 10

A span is multi-root when more than one of its words has its head outside the
span (same definition as n_roots in notebooks/02_ne_inspection.py and
spans.span_tree). Writes results/multiroot_<sample>.csv with all of them, or
with --per-type N a random sample of N spans per NE type (--seed) to
results/multiroot_<sample>_n<N>_s<seed>.csv. Each row has the span in context
([[...]] in `sentence`), its patterns as in notebooks/04_span_trees.py, and
empty columns to fill in:

    label       ok           NE and syntax both correct, the span is legitimately
                             not one subtree
                ne_error     the NE annotation is wrong
                parse_error  the NE is right, the dependency analysis is wrong
                both         NE and syntax both wrong
                unclear      cannot decide
    error_type  for ne_error/both:
                merged       two or more NEs annotated as one span
                too_long     span includes words that are not part of the name
                truncated    part of the name is outside the span
                wrong_type   the boundaries are right but the type is wrong
                not_ne       not a named entity at all
                (for parse_error, a short description, e.g. "flat missing")
    note        free text
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ne_syntax_consistency.spans import span_tree

RESULTS = Path(__file__).resolve().parents[1] / "results"


def multiroot_rows(s: dict):
    # Stanza heads are 1-based (0 = root); convert to 0-based, root -> None
    heads = [h - 1 if h else None for h in s["head"]]
    for ne_index, (tag, idx) in enumerate(zip(s["ne_tag"], s["ne_words"])):
        ids = sorted(idx)
        roots = [i for i in ids if heads[i] not in idx]
        if len(roots) < 2:
            continue
        words = list(s["words"])
        words[ids[0]] = "[[" + words[ids[0]]
        words[ids[-1]] = words[ids[-1]] + "]]"
        yield {
            "text_id": s["text_id"],
            "ne_index": ne_index,
            "nertag": tag,
            "ne_text": " ".join(s["words"][i] for i in ids),
            "n_words": len(ids),
            "n_roots": len(roots),
            "pattern_type": span_tree(s, idx, tag, "type")["pattern"],
            "pattern_pos_case": span_tree(s, idx, tag, "pos_case")["pattern"],
            "root_deprels": "+".join(s["deprel"][i] for i in roots),
            # each root -> the word it attaches to outside the span
            "root_heads": "; ".join(
                f"{s['words'][i]} -{s['deprel'][i]}-> "
                + (s["words"][heads[i]] if heads[i] is not None else "ROOT")
                for i in roots
            ),
            "span_deprels": " ".join(
                f"{s['words'][i]}/{s['deprel'][i]}>{s['head'][i]}" for i in ids
            ),
            "other_nes": "; ".join(
                " ".join(s["words"][i] for i in sorted(w)) + "=" + t
                for j, (t, w) in enumerate(zip(s["ne_tag"], s["ne_words"]))
                if j != ne_index
            ),
            "sentence": " ".join(words),
            "label": "",
            "error_type": "",
            "note": "",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", type=Path)
    parser.add_argument(
        "--per-type", type=int, help="random sample of this many spans per NE type"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    columns = [
        "text_id",
        "text",
        "words",
        "pos",
        "form",
        "head",
        "deprel",
        "ne_tag",
        "ne_words",
    ]
    sentences = pq.read_table(args.sample, columns=columns).to_pylist()
    df = pd.DataFrame([r for s in sentences for r in multiroot_rows(s)])
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"multiroot_{args.sample.stem}.csv"
    if args.per_type:
        df = (
            df.groupby("nertag", group_keys=False)
            .sample(n=args.per_type, random_state=args.seed)
            .reset_index(drop=True)
        )
        path = path.with_stem(f"{path.stem}_n{args.per_type}_s{args.seed}")
    df.to_csv(path, index=False)
    print(f"{len(df)} multi-root spans -> {path}")
    print(df["nertag"].value_counts().to_string())


if __name__ == "__main__":
    main()
