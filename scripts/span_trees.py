"""Support of whole internal NE span trees (complete patterns only).

    uv run python scripts/span_trees.py data/sample_200000_s1.parquet --label type
    uv run python scripts/span_trees.py data/ud_edt_train.parquet --label pos_case

Every NE span contributes exactly one pattern: the tree of its words and the
arcs between them (see ne_syntax_consistency.spans), so a pattern's support is
the number of spans whose whole internal tree is that pattern; sub-patterns are
not counted. Writes results/span_trees/span_trees_<label>_<sample>.csv with support per
NE type and pattern, and the share of that type's spans (share_of_type), most
frequent first, for patterns with support >= --min-support; the rarer patterns
go to .._rare.csv. Shares are of all the type's spans, rare ones included.
Also writes .._spans.parquet with one row per span.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ne_syntax_consistency.spans import LABELS, span_trees, support_table

RESULTS = Path(__file__).resolve().parents[1] / "results" / "span_trees"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", type=Path, help="sentence Parquet (sample or UD)")
    parser.add_argument("--label", choices=LABELS, default="type")
    parser.add_argument(
        "--min-support", type=int, default=10, help="rarer patterns go to _rare.csv"
    )
    parser.add_argument("--top", type=int, default=10, help="patterns printed per type")
    args = parser.parse_args()

    sentences = pq.read_table(
        args.sample,
        columns=[
            "text_id",
            "words",
            "pos",
            "form",
            "head",
            "deprel",
            "ne_tag",
            "ne_words",
        ],
    ).to_pylist()
    spans = pd.DataFrame(row for s in sentences for row in span_trees(s, args.label))
    table = support_table(spans)

    stem = f"span_trees_{args.label}_{args.sample.stem}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    frequent = table["support"] >= args.min_support
    table[~frequent].to_csv(RESULTS / f"{stem}_rare.csv", index=False)
    table = table[frequent]
    table.to_csv(RESULTS / f"{stem}.csv", index=False)
    spans.to_parquet(RESULTS / f"{stem}_spans.parquet", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        for tag, group in table.groupby("nertag"):
            print(
                f"\n{tag}: {len(group)} patterns with support >= {args.min_support}"
                f", covering {group['share_of_type'].sum():.1%} of spans"
            )
            print(
                group.head(args.top)[
                    ["support", "share_of_type", "pattern", "example"]
                ].to_string(index=False, float_format="{:.3f}".format)
            )
    print(f"\nwrote {RESULTS / stem}.csv and {stem}_rare.csv")


if __name__ == "__main__":
    main()
