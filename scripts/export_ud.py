"""Export the UD Estonian treebanks (EDT, EWT) to Parquet.

    uv run python scripts/export_ud.py

Reads data/UD_Estonian-{EDT,EWT}/et_*-ud-{train,dev,test}.conllu and writes
data/ud_<treebank>_<split>.parquet (one row per sentence, see
ne_syntax_consistency.data.UD_SCHEMA). The files have the same columns as the
Koondkorpus samples, so run_gspan.py and confidence.py run on them unchanged.
"""

import argparse
import logging
import time
from pathlib import Path

from ne_syntax_consistency.data import export_ud

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--treebanks", nargs="+", default=["EDT", "EWT"])
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)

    for tb in args.treebanks:
        for split in args.splits:
            conllu = DATA / f"UD_Estonian-{tb}" / f"et_{tb.lower()}-ud-{split}.conllu"
            path = DATA / f"ud_{tb.lower()}_{split}.parquet"
            t0 = time.perf_counter()
            n = export_ud(conllu, path)
            print(f"{path.name}: {n} sentences ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
