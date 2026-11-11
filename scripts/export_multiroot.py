"""Export all multi-root NE spans of a sample to CSV for labelling.

    uv run python scripts/export_multiroot.py data/sample_200000_s1.parquet

A span is multi-root when more than one of its words has its head outside the
span (same definition as n_roots in notebooks/02_ne_inspection.py). Writes
results/multiroot/multiroot_<sample>.csv with empty label, error_type and note columns,
in the format of results/multiroot/multiroot_labels_20k.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RESULTS = Path(__file__).resolve().parents[1] / "results" / "multiroot"


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
    args = parser.parse_args()

    columns = ["text_id", "text", "words", "head", "deprel", "ne_tag", "ne_words"]
    sentences = pq.read_table(args.sample, columns=columns).to_pylist()
    df = pd.DataFrame([r for s in sentences for r in multiroot_rows(s)])
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"multiroot_{args.sample.stem}.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} multi-root spans -> {path}")
    print(df["nertag"].value_counts().to_string())


if __name__ == "__main__":
    main()
