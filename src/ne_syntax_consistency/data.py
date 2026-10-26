from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from estnltk import Text

from ne_syntax_consistency.db import get_storage

COLLECTION = "koondkorpus_sentences"
LAYERS = ["sentences", "morph_analysis", "v171_named_entities", "v172_stanza_syntax"]

# One row per sentence; token lists are aligned with `words`
SENTENCE_SCHEMA = pa.schema(
    [
        ("text_id", pa.int64()),
        ("text", pa.string()),
        ("words", pa.list_(pa.string())),
        # first morph_analysis analysis
        ("pos", pa.list_(pa.string())),
        ("form", pa.list_(pa.string())),
        ("lemma", pa.list_(pa.string())),
        # v172_stanza_syntax: head is the 1-based Stanza id, 0 = root
        ("head", pa.list_(pa.int32())),
        ("deprel", pa.list_(pa.string())),
        # v171_named_entities: one entry per NE, word indices are 0-based
        ("ne_tag", pa.list_(pa.string())),
        ("ne_words", pa.list_(pa.list_(pa.int32()))),
    ]
)


def load_sample(
    n: int, layers: list[str] = LAYERS, collection: str = COLLECTION
) -> list[tuple[int, Text]]:
    storage = get_storage()
    try:
        return list(storage[collection].select(layers=layers).head(n))
    finally:
        storage.close()


def sentence_record(text_id: int, text: Text) -> dict | None:
    words = text["words"]
    morph = text["morph_analysis"]
    syntax = text["v172_stanza_syntax"]
    if len(morph) != len(words) or len(syntax) != len(words):
        return None
    start2i = {w.start: i for i, w in enumerate(words)}
    analyses = [m.annotations[0] for m in morph]
    tokens = [s.annotations[0] for s in syntax]
    nes = text["v171_named_entities"]
    return {
        "text_id": text_id,
        "text": text.text,
        "words": [w.text for w in words],
        "pos": [a["partofspeech"] for a in analyses],
        "form": [a["form"] or "" for a in analyses],
        "lemma": [a["lemma"] for a in analyses],
        "head": [t["head"] for t in tokens],
        "deprel": [t["deprel"] for t in tokens],
        "ne_tag": [ne.annotations[0]["nertag"] for ne in nes],
        "ne_words": [[start2i[w.start] for w in ne.spans] for ne in nes],
    }


def export_sample(
    path: Path,
    n: int,
    seed: int,
    collection: str = COLLECTION,
    chunk_size: int = 10000,
) -> Iterator[tuple[int, int]]:
    """Write a random sample of about n sentences to Parquet, in chunks.

    Yields (sentences written, sentences skipped) after each chunk so callers can
    report progress. The file is written to `path` only once complete.
    """
    tmp = path.with_suffix(".partial")
    storage = get_storage()
    written = skipped = 0
    try:
        sample = (
            storage[collection]
            .select(layers=LAYERS + ["words"])
            .sample(n, amount_type="SIZE", seed=seed)
        )
        with pq.ParquetWriter(tmp, SENTENCE_SCHEMA) as writer:
            rows = []
            for text_id, text in sample:
                record = sentence_record(text_id, text)
                if record is None:
                    skipped += 1
                    continue
                rows.append(record)
                if len(rows) == chunk_size:
                    writer.write_table(pa.Table.from_pylist(rows, SENTENCE_SCHEMA))
                    written += len(rows)
                    rows = []
                    yield written, skipped
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, SENTENCE_SCHEMA))
                written += len(rows)
                yield written, skipped
    finally:
        storage.close()
    tmp.rename(path)
