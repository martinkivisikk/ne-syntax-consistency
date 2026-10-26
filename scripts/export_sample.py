"""Export a random sentence sample from the database to Parquet.

    uv run python scripts/export_sample.py --sentences 200000 --seed 1

Writes data/sample_<sentences>_s<seed>.parquet (one row per sentence, see
ne_syntax_consistency.data.SENTENCE_SCHEMA). Mining scripts read this file, so
the database is only queried once per sample.
"""

import argparse
import logging
import time
from pathlib import Path

from ne_syntax_consistency.data import COLLECTION, export_sample

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sentences", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)

    DATA.mkdir(exist_ok=True)
    path = DATA / f"sample_{args.sentences}_s{args.seed}.parquet"
    t0 = time.perf_counter()
    print(
        f"sampling about {args.sentences} sentences (seed {args.seed}) ...", flush=True
    )
    for written, skipped in export_sample(
        path, args.sentences, args.seed, args.collection
    ):
        print(
            f"{time.perf_counter() - t0:6.0f} s  {written} written, {skipped} skipped",
            flush=True,
        )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
