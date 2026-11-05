"""Export every sentence with an NE from the whole collection, in resumable parts.

    uv run python scripts/export_full.py
    uv run python scripts/export_full.py --stop 5000000   # only ids < 5M for now
    uv run python scripts/export_full.py --summary        # progress so far

Walks the sentence ids in parts of --part-size ids and writes each part to
data/full/part_<start>_<stop>.parquet (SENTENCE_SCHEMA, sentences with at least
one NE only). A part file appears only once the part is complete, so the script
can be stopped at any time (Ctrl-C) and rerun with the same arguments: finished
parts are skipped and only the interrupted part is redone. Uses one database
connection; if it drops, the part is retried after a pause.

Every --verify-every'th part is also checked against the NE layers of all
sentences in the range (see data.export_range), which is slower. Each part
records its statistics (sentences, skipped, sentences missing the NE layer)
in the file's metadata; --summary adds them up.
"""

import argparse
import contextlib
import json
import logging
import time
from pathlib import Path

import psycopg2
import pyarrow.parquet as pq
from estnltk.storage.postgres import collection_table_identifier
from psycopg2.sql import SQL

from ne_syntax_consistency.data import COLLECTION, export_range
from ne_syntax_consistency.db import get_storage

FULL = Path(__file__).resolve().parents[1] / "data" / "full"


def part_path(out: Path, start: int, stop: int) -> Path:
    return out / f"part_{start:08d}_{stop:08d}.parquet"


def summary(out: Path) -> None:
    parts = sorted(out.glob("part_*.parquet"))
    stats = [json.loads(pq.read_schema(p).metadata[b"export"]) for p in parts]
    totals = {
        key: sum(s[key] for s in stats)
        for key in ("sentences", "skipped", "missing_ne_layer", "seconds")
    }
    print(
        f"{len(parts)} parts in {out}, {sum(s['verified'] for s in stats)} verified\n"
        f"ids covered: {sum(s['stop'] - s['start'] for s in stats)}\n"
        f"sentences with NEs written: {totals['sentences']}\n"
        f"skipped (layers not aligned with words): {totals['skipped']}\n"
        f"sentences without a stored NE layer: {totals['missing_ne_layer']}\n"
        f"export time: {totals['seconds'] / 3600:.1f} h"
    )


def max_id(storage, collection: str) -> int:
    table = collection_table_identifier(storage, collection)
    with storage.conn.cursor() as cur:
        cur.execute(SQL("SELECT max(id) FROM {}").format(table))
        return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--out", type=Path, default=FULL)
    parser.add_argument("--part-size", type=int, default=100_000)
    parser.add_argument("--start", type=int, default=0, help="first id (inclusive)")
    parser.add_argument("--stop", type=int, help="last id (exclusive); default all")
    parser.add_argument("--verify-every", type=int, default=20, help="0 = never verify")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)

    if args.summary:
        summary(args.out)
        return
    args.out.mkdir(parents=True, exist_ok=True)

    storage = get_storage()
    try:
        stop = (
            args.stop if args.stop is not None else max_id(storage, args.collection) + 1
        )
        # part boundaries are multiples of --part-size so reruns line up
        first = args.start - args.start % args.part_size
        bounds = [
            (lo, min(lo + args.part_size, stop))
            for lo in range(first, stop, args.part_size)
        ]
        todo = [b for b in bounds if not part_path(args.out, *b).exists()]
        print(
            f"ids {first}-{stop}: {len(bounds)} parts, {len(bounds) - len(todo)} done,"
            f" {len(todo)} to go",
            flush=True,
        )
        t0 = time.perf_counter()
        for k, (lo, hi) in enumerate(todo, 1):
            verify = (
                args.verify_every > 0
                and (lo // args.part_size) % args.verify_every == 0
            )
            for attempt in range(args.retries + 1):
                try:
                    stats = export_range(
                        storage[args.collection],
                        part_path(args.out, lo, hi),
                        lo,
                        hi,
                        verify=verify,
                    )
                    break
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    if attempt == args.retries:
                        raise
                    wait = 30 * (attempt + 1)
                    print(f"  database error ({e!r}), retrying in {wait} s", flush=True)
                    with contextlib.suppress(psycopg2.Error):
                        storage.close()
                    time.sleep(wait)
                    storage = get_storage()
            elapsed = time.perf_counter() - t0
            eta = elapsed / k * (len(todo) - k)
            print(
                f"[{k}/{len(todo)}] ids {lo}-{hi}: {stats['sentences']} sentences"
                f" ({stats['skipped']} skipped, {stats['missing_ne_layer']} without"
                f" NE layer{', verified' if verify else ''}) in {stats['seconds']:.0f} s"
                f" | elapsed {elapsed / 3600:.1f} h, left ~{eta / 3600:.1f} h",
                flush=True,
            )
    finally:
        storage.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
