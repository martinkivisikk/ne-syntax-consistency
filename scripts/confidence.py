"""Confidence of per-type gSpan patterns as rules `context -> NE type`.

    uv run python scripts/run_gspan.py data/sample_200000_s1.parquet --per-type
    uv run python scripts/confidence.py data/sample_200000_s1.parquet

A typed pattern such as `O:V:- -obl-> *LOC:H:in` has the context
`O:V:- -obl-> *NE:H:in` (focal type masked). Its confidence is
support(typed) / support(context), counted over all NE spans: the share of NE
spans in that context that have this type. Lift divides by the type's overall
share. Context supports come from one gSpan run on masked ego graphs, at the
lowest typed support, so every needed context is found (a context's support is
at least that of each of its typed patterns).

Use the same --level/--max-dependents/--max-vertices as the run_gspan.py run.
Writes results/gspan_<level>_<sample>_pertype_confidence.csv and
.._exceptions.csv: for complete rules with confidence >= --min-confidence, every
NE span in the rule's context whose type differs from the rule's (one row per
rule and span; a span can break several rules).
"""

import argparse
import ast
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ne_syntax_consistency.gspan import (
    LEVELS,
    contains,
    ego_graphs,
    mask_focal_type,
    mine,
    pattern_graph,
    pattern_key,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", type=Path, help="Parquet file from export_sample.py")
    parser.add_argument("--level", choices=LEVELS, default="pos_case")
    parser.add_argument("--max-vertices", type=int, default=7)
    parser.add_argument("--max-dependents", type=int, default=2)
    parser.add_argument(
        "--prune-margin",
        type=float,
        default=0.01,
        help="drop a rule if a sub-pattern's confidence is within this margin",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="rules whose exceptions are written out (complete patterns only)",
    )
    args = parser.parse_args()

    stem = RESULTS / f"gspan_{args.level}_{args.sample.stem}_pertype"
    typed = pd.read_csv(f"{stem}.csv")
    egos = [
        ego
        for batch in pq.ParquetFile(args.sample).iter_batches()
        for sentence in batch.to_pylist()
        for ego in ego_graphs(sentence, args.level, args.max_dependents)
    ]
    graphs = pd.read_parquet(f"{stem}_graphs.parquet")
    if len(graphs) != len(egos) or (graphs["nertag"] != [e.nertag for e in egos]).any():
        raise SystemExit(
            "ego graphs differ from the run_gspan.py run; check the options"
        )

    min_support = int(typed["support"].min())
    print(
        f"mining {len(egos)} masked ego graphs, min support {min_support} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    contexts = mine(
        [mask_focal_type(e) for e in egos],
        min_support=min_support,
        max_vertices=args.max_vertices,
    )
    print(
        f"{len(contexts)} contexts; mine {time.perf_counter() - t0:.0f} s", flush=True
    )

    # Type distribution of NE spans in each context
    nertag = graphs["nertag"].to_numpy()
    types = sorted(graphs["nertag"].unique())
    contexts["key"] = [pattern_key(c) for c in contexts["dfs_code"]]
    counts = (
        pd.DataFrame(
            [pd.Series(nertag[ids]).value_counts() for ids in contexts["graph_ids"]],
            columns=types,
        )
        .fillna(0)
        .astype(int)
        .reset_index(drop=True)
    )
    contexts = pd.concat(
        [
            contexts[["key", "pattern", "support", "graph_ids"]],
            counts.add_prefix("n_"),
        ],
        axis=1,
    ).rename(
        columns={
            "pattern": "context",
            "support": "context_support",
            "graph_ids": "context_graph_ids",
        }
    )

    typed["key"] = [pattern_key(c, mask=True) for c in typed["dfs_code"]]
    out = typed.drop(columns=["share"]).merge(
        contexts, on="key", how="left", validate="many_to_one"
    )
    missing = out["context_support"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} patterns have no matching context")

    prior = graphs["nertag"].value_counts(normalize=True)
    out["confidence"] = out["support"] / out["context_support"]
    out["lift"] = out["confidence"] / out["group"].map(prior)
    out["graph_ids"] = out["graph_ids"].map(ast.literal_eval)
    cols = ["group", "pattern", "context", "complete", "n_vertices", "support"]
    cols += ["context_support", "confidence", "lift"] + [f"n_{t}" for t in types]
    out = out[cols + ["dfs_code", "graph_ids", "context_graph_ids"]].sort_values(
        ["confidence", "support"], ascending=False, ignore_index=True
    )
    out.drop(columns="context_graph_ids").to_csv(f"{stem}_confidence.csv", index=False)

    # Rules: complete patterns above the confidence threshold, without
    # punctuation arcs, minus those a sub-pattern already explains (plan.md,
    # "redundancy pruning")
    rules = out[
        out["complete"]
        & (out["confidence"] >= args.min_confidence)
        & ~out["pattern"].str.contains("-punct->")
    ]
    graphs_of = {i: pattern_graph(code) for i, code in rules["dfs_code"].items()}
    redundant = {
        b.Index
        for b in rules.itertuples()
        for a in rules.itertuples()
        if a.Index != b.Index
        and a.group == b.group
        and graphs_of[a.Index].number_of_nodes() < graphs_of[b.Index].number_of_nodes()
        and b.confidence - a.confidence < args.prune_margin
        and contains(graphs_of[b.Index], graphs_of[a.Index])
    }
    rules = rules.drop(index=list(redundant))
    rules.drop(columns=["graph_ids", "context_graph_ids"]).to_csv(
        f"{stem}_rules.csv", index=False
    )

    # Exceptions: NE spans in a rule's context whose type is not the rule's type
    exceptions = pd.DataFrame(
        [
            {
                "expected": rule.group,
                "pattern": rule.pattern,
                "confidence": rule.confidence,
                "graph_id": gid,
            }
            for rule in rules.itertuples()
            for gid in rule.context_graph_ids
            if nertag[gid] != rule.group
        ]
    )
    exceptions = exceptions.join(graphs, on="graph_id")
    exceptions.to_csv(f"{stem}_exceptions.csv", index=False)
    print(
        f"wrote {stem}_confidence.csv, _rules.csv and _exceptions.csv "
        f"({len(rules)} rules, {len(exceptions)} exceptions, "
        f"{exceptions['graph_id'].nunique()} distinct spans)",
        flush=True,
    )


if __name__ == "__main__":
    main()
