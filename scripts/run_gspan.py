"""Mine frequent NE ego-graph patterns with gSpan on an exported sentence sample.

    uv run python scripts/export_sample.py --sentences 200000 --seed 1
    uv run python scripts/run_gspan.py data/sample_200000_s1.parquet --min-support 0.01

Writes results/gspan_<level>_<sample>[_pertype].csv (patterns) and
.._graphs.parquet (one row per ego graph, indexed by the pattern graph_ids).
`share` is support divided by the number of ego graphs in the pattern's group
(all NE spans, or with --per-type the spans of one NE type).
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ne_syntax_consistency.gspan import LEVELS, ego_graphs, mine

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", type=Path, help="Parquet file from export_sample.py")
    parser.add_argument("--level", choices=LEVELS, default="pos_case")
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.01,
        help="share of ego graphs (<1) or a count (>=1)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=7,
        help="including relation nodes; 7 = three arcs",
    )
    parser.add_argument(
        "--max-dependents",
        type=int,
        default=2,
        help="dependents kept per head on each side of the span",
    )
    parser.add_argument(
        "--per-type",
        action="store_true",
        help="mine each NE type separately; a share in --min-support is then "
        "a share of that type's spans",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    egos = [
        ego
        for batch in pq.ParquetFile(args.sample).iter_batches()
        for sentence in batch.to_pylist()
        for ego in ego_graphs(sentence, args.level, args.max_dependents)
    ]
    sizes = pd.Series([len(e.labels) for e in egos])
    print(
        f"{len(egos)} ego graphs in {time.perf_counter() - t0:.0f} s; "
        f"nodes median {sizes.median():.0f}, p99 {sizes.quantile(0.99):.0f}, "
        f"max {sizes.max()}",
        flush=True,
    )

    # Every reported pattern contains the focal type (*PER etc.), so mining each
    # type's ego graphs on its own is the same as a per-type support threshold
    groups = (
        {
            t: [i for i, e in enumerate(egos) if e.nertag == t]
            for t in sorted({e.nertag for e in egos})
        }
        if args.per_type
        else {"all": list(range(len(egos)))}
    )
    results = []
    t0 = time.perf_counter()
    for group, ids in groups.items():
        min_support = (
            round(args.min_support * len(ids))
            if args.min_support < 1
            else int(args.min_support)
        )
        patterns = mine(
            [egos[i] for i in ids],
            min_support=min_support,
            max_vertices=args.max_vertices,
        )
        # graph_ids index into the group; map them back to the full ego graph list
        patterns["graph_ids"] = [
            [ids[g] for g in gids] for gids in patterns["graph_ids"]
        ]
        patterns.insert(0, "group", group)
        patterns.insert(2, "share", patterns["support"] / len(ids))
        print(
            f"{group}: {len(ids)} ego graphs, min support {min_support}, "
            f"{len(patterns)} patterns",
            flush=True,
        )
        results.append(patterns)
    patterns = pd.concat(results, ignore_index=True)
    print(f"mine {time.perf_counter() - t0:.0f} s", flush=True)

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"gspan_{args.level}_{args.sample.stem}"
    if args.per_type:
        stem = stem.with_name(stem.name + "_pertype")
    patterns.to_csv(f"{stem}.csv", index=False)
    graphs = pd.DataFrame(
        {
            "text_id": e.text_id,
            "ne_index": e.ne_index,
            "nertag": e.nertag,
            "ne_text": e.ne_text,
            "n_nodes": len(e.labels),
        }
        for e in egos
    )
    graphs.to_parquet(f"{stem}_graphs.parquet")
    print(f"wrote {stem}.csv and {stem}_graphs.parquet", flush=True)


if __name__ == "__main__":
    main()
