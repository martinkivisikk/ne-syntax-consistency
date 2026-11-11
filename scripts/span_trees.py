"""Support of whole internal NE span trees (complete patterns only).

    uv run python scripts/span_trees.py data/sample_200000_s1.parquet --label type
    uv run python scripts/span_trees.py data/ud_edt_train.parquet --label pos_case
    uv run python scripts/span_trees.py data/full --label type
    uv run python scripts/span_trees.py data/full --max-id 19256634 --name full_nochat
    uv run python scripts/span_trees.py data/full --min-id 19256634 --name full_chat

The input is one sentence Parquet file or a directory of them (such as the full
corpus export in data/full); files are read one at a time, so the full corpus
does not have to fit in memory. --min-id/--max-id keep sentences with
min_id <= text_id < max_id; in data/full, text_id 19256634 onward is the chat
room section. --name replaces <sample> in the output names (default: the input
file or directory name).

Every NE span contributes exactly one pattern: the tree of its words and the
arcs between them (see ne_syntax_consistency.spans), so a pattern's support is
the number of spans whose whole internal tree is that pattern; sub-patterns are
not counted. Writes results/span_trees/span_trees_<label>_<sample>.csv with
support per NE type and pattern, and the share of that type's spans
(share_of_type), most frequent first, for patterns with support >=
--min-support; the rarer patterns go to .._rare.csv. Shares are of all the
type's spans, rare ones included. Also writes .._spans.parquet with one row per
span.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ne_syntax_consistency.spans import LABELS, span_trees, support_table

RESULTS = Path(__file__).resolve().parents[1] / "results" / "span_trees"
COLUMNS = ["text_id", "words", "pos", "form", "head", "deprel", "ne_tag", "ne_words"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sample", type=Path, help="sentence Parquet file or a directory of them"
    )
    parser.add_argument("--label", choices=LABELS, default="type")
    parser.add_argument(
        "--min-support", type=int, default=10, help="rarer patterns go to _rare.csv"
    )
    parser.add_argument("--top", type=int, default=10, help="patterns printed per type")
    parser.add_argument("--min-id", type=int, help="keep text_id >= this")
    parser.add_argument("--max-id", type=int, help="keep text_id < this")
    parser.add_argument("--name", help="output name instead of the input's stem")
    args = parser.parse_args()

    files = (
        sorted(args.sample.glob("*.parquet")) if args.sample.is_dir() else [args.sample]
    )
    stem = f"span_trees_{args.label}_{args.name or args.sample.stem}"
    filters = [
        f
        for f in [
            ("text_id", ">=", args.min_id) if args.min_id is not None else None,
            ("text_id", "<", args.max_id) if args.max_id is not None else None,
        ]
        if f
    ] or None
    RESULTS.mkdir(parents=True, exist_ok=True)
    spans_path = RESULTS / f"{stem}_spans.parquet"

    # support per file, summed at the end; spans are streamed to Parquet
    partial = []
    writer = None
    t0 = time.perf_counter()
    try:
        for n, path in enumerate(files, 1):
            sentences = pq.read_table(
                path, columns=COLUMNS, filters=filters
            ).to_pylist()
            spans = pd.DataFrame(
                row for s in sentences for row in span_trees(s, args.label)
            )
            if spans.empty:
                continue
            partial.append(support_table(spans).drop(columns="share_of_type"))
            table = pa.Table.from_pandas(spans, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(spans_path, table.schema)
            writer.write_table(table)
            if len(files) > 1:
                print(
                    f"{time.perf_counter() - t0:6.0f} s  {n}/{len(files)} files",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    table = (
        pd.concat(partial, ignore_index=True)
        .groupby(["nertag", "pattern"], sort=False)
        .agg(
            support=("support", "sum"),
            n_words=("n_words", "first"),
            n_roots=("n_roots", "first"),
            example=("example", "first"),
        )
        .reset_index()
    )
    table["share_of_type"] = table["support"] / table.groupby("nertag")[
        "support"
    ].transform("sum")
    table = table.sort_values(["nertag", "support"], ascending=[True, False])

    frequent = table["support"] >= args.min_support
    table[~frequent].to_csv(RESULTS / f"{stem}_rare.csv", index=False)
    table = table[frequent]
    table.to_csv(RESULTS / f"{stem}.csv", index=False)

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
    print(f"\nwrote {RESULTS / stem}.csv, {stem}_rare.csv and {stem}_spans.parquet")


if __name__ == "__main__":
    main()
