import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from estnltk import Layer, Text
from estnltk.converters.conll.conll_importer import conll_to_texts_list
from estnltk.storage.postgres import PgCollection, collection_table_identifier
from estnltk.storage.postgres.queries.layer_query import LayerQuery
from estnltk.storage.postgres.queries.missing_layer_query import MissingLayerQuery
from estnltk.storage.postgres.queries.slice_query import SliceQuery
from estnltk.taggers import VabamorfTagger
from psycopg2.sql import SQL

from ne_syntax_consistency.db import get_storage

COLLECTION = "koondkorpus_sentences"
NE_LAYER = "v171_named_entities"
LAYERS = ["sentences", "morph_analysis", NE_LAYER, "v172_stanza_syntax"]
# All nertag values of NE_LAYER (checked on the 200k sample; export_range can
# verify that no sentence with an NE is missed)
NE_TAGS = ("PER", "LOC", "ORG")

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


# UD treebanks: SENTENCE_SCHEMA with gold morphology and syntax, plus the UD ids,
# UPOS and Vabamorf's first analysis of the same words for comparison
UD_SCHEMA = SENTENCE_SCHEMA.append(pa.field("sent_id", pa.string()))
for _name, _type in [
    ("doc_id", pa.string()),
    ("upos", pa.list_(pa.string())),
    ("vm_pos", pa.list_(pa.string())),
    ("vm_form", pa.list_(pa.string())),
]:
    UD_SCHEMA = UD_SCHEMA.append(pa.field(_name, _type))

# UD Case feature -> Vabamorf case code (see gspan.CASES)
UD_CASES = {
    "Nom": "n",
    "Gen": "g",
    "Par": "p",
    "Ill": "ill",
    "Ine": "in",
    "Ela": "el",
    "All": "all",
    "Ade": "ad",
    "Abl": "abl",
    "Tra": "tr",
    "Ter": "ter",
    "Ess": "es",
    "Abe": "ab",
    "Com": "kom",
    "Add": "adt",
}
UD_NUMBER = {"Sing": "sg", "Plur": "pl"}

# MISC NE=B-Per / I-Per; lenient, as a few tags are broken (BäOrg, B_Gep, i-Prod,
# B-OrgSpaceAfter=No, bare Org). A tag without B/I continues a span of its type.
UD_NE_TYPES = ("Per", "Loc", "Org", "Gep", "Eve", "Prod", "Muu", "Unk")
_NE_TAG = re.compile(rf"^(?:([BI])[-_.ä]?)?({'|'.join(UD_NE_TYPES)})", re.IGNORECASE)


def ud_ne_tag(value: str | None) -> tuple[str, str] | None:
    """Parse a MISC NE value into (B or I, upper-case type), or None."""
    m = _NE_TAG.match(value or "")
    if m is None:
        return None
    return (m.group(1) or "I").upper(), m.group(2).upper()


def ud_ne_spans(tags: list[tuple[str, str] | None]) -> list[tuple[str, list[int]]]:
    """BIO tags -> (type, word indices); an I- without a matching open span starts one."""
    spans = []
    open_type = None
    for i, tag in enumerate(tags):
        if tag is None:
            open_type = None
            continue
        bio, ne_type = tag
        if bio == "B" or ne_type != open_type:
            spans.append((ne_type, []))
            open_type = ne_type
        spans[-1][1].append(i)
    return spans


def _ud_form(feats: dict | None) -> str:
    feats = feats or {}
    return " ".join(
        x
        for x in (UD_NUMBER.get(feats.get("Number")), UD_CASES.get(feats.get("Case")))
        if x
    )


def ud_records(path: Path) -> Iterator[dict]:
    """One UD_SCHEMA record per sentence of a CoNLL-U file.

    pos: XPOS (Vabamorf-style) with H for UPOS PROPN; form: number and case from
    the UD features as Vabamorf codes (e.g. "sg g"); head/deprel: gold UD.
    vm_pos/vm_form: first analysis of VabamorfTagger on the same words.
    text_id is the 0-based sentence index in the file.
    """
    sent_ids = [
        line.split("=", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# sent_id")
    ]
    vabamorf = VabamorfTagger()
    k = 0
    for text in conll_to_texts_list(str(path)):
        # VabamorfTagger needs a compound_tokens layer; it can stay empty
        text.add_layer(
            Layer(
                "compound_tokens", text_object=text, attributes=["type", "normalized"]
            )
        )
        vabamorf.tag(text)
        words = text["words"]
        syntax = text["conll_syntax"]
        morph = text["morph_analysis"]
        start2i = {w.start: i for i, w in enumerate(words)}
        for sentence in text["sentences"]:
            ids = [start2i[w.start] for w in sentence]
            tokens = [syntax[i].annotations[0] for i in ids]
            analyses = [morph[i].annotations[0] for i in ids]
            nes = ud_ne_spans(
                [ud_ne_tag(t["misc"] and t["misc"].get("NE")) for t in tokens]
            )
            yield {
                "text_id": k,
                "text": sentence.enclosing_text,
                "words": [words[i].text for i in ids],
                "pos": [
                    "H"
                    if t["upostag"] == "PROPN" and t["xpostag"] != "Y"
                    # goeswith parts (split numbers like "000") have no XPOS
                    else t["xpostag"] or "X"
                    for t in tokens
                ],
                "form": [_ud_form(t["feats"]) for t in tokens],
                "lemma": [t["lemma"] for t in tokens],
                "head": [t["head"] for t in tokens],
                "deprel": [t["deprel"] for t in tokens],
                "ne_tag": [ne_type for ne_type, _ in nes],
                "ne_words": [idx for _, idx in nes],
                "sent_id": sent_ids[k],
                "doc_id": text.meta.get("file_prefix"),
                "upos": [t["upostag"] for t in tokens],
                "vm_pos": [a["partofspeech"] for a in analyses],
                "vm_form": [a["form"] or "" for a in analyses],
            }
            k += 1
    if k != len(sent_ids):
        raise ValueError(f"{path}: {k} sentences read, {len(sent_ids)} sent_id lines")


def export_ud(conllu: Path, path: Path) -> int:
    """Write ud_records(conllu) to Parquet; return the number of sentences."""
    table = pa.Table.from_pylist(list(ud_records(conllu)), UD_SCHEMA)
    pq.write_table(table, path)
    return table.num_rows


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


def has_ne_query(tags: tuple[str, ...] = NE_TAGS) -> LayerQuery:
    """estnltk query for sentences with at least one NE of the given tags."""
    query = LayerQuery(NE_LAYER, nertag=tags[0])
    for tag in tags[1:]:
        query = query | LayerQuery(NE_LAYER, nertag=tag)
    return query


def count_missing_ne_layer(collection: PgCollection, start: int, stop: int) -> int:
    """Number of sentences with ids in [start, stop) that have no NE layer stored."""
    condition = SliceQuery(start, stop) & MissingLayerQuery(NE_LAYER)
    table = collection_table_identifier(collection.storage, collection.name)
    query = SQL("SELECT count(*) FROM {table} WHERE {condition}").format(
        table=table, condition=condition.eval(collection)
    )
    with collection.storage.conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchone()[0]


def export_range(
    collection: PgCollection,
    path: Path,
    start: int,
    stop: int,
    verify: bool = False,
    itersize: int = 200,
) -> dict:
    """Write the sentences with NEs whose ids are in [start, stop) to Parquet.

    Selects them with has_ne_query(). With verify=True, also loads the NE layer
    of every sentence in the range and checks that exactly the sentences with
    a non-empty NE layer were selected (raises ValueError otherwise).
    The file is written to `path` only once complete, with the returned
    statistics in its schema metadata under b"export".
    """
    t0 = time.perf_counter()
    ids = SliceQuery(start, stop)
    rows = []
    selected = set()
    skipped = 0
    for text_id, text in collection.select(
        ids & has_ne_query(), layers=LAYERS + ["words"], itersize=itersize
    ):
        selected.add(text_id)
        record = sentence_record(text_id, text)
        if record is None:
            skipped += 1
        else:
            rows.append(record)
    if verify:
        expected = {
            text_id
            for text_id, text in collection.select(
                ids, layers=[NE_LAYER], itersize=itersize
            )
            if len(text[NE_LAYER])
        }
        if expected != selected:
            raise ValueError(
                f"ids {start}-{stop}: has_ne_query selected {len(selected)} sentences,"
                f" {len(expected)} have NEs; missed e.g."
                f" {sorted(expected - selected)[:10]},"
                f" extra e.g. {sorted(selected - expected)[:10]}"
            )
    stats = {
        "start": start,
        "stop": stop,
        "sentences": len(rows),
        "skipped": skipped,
        "missing_ne_layer": count_missing_ne_layer(collection, start, stop),
        "verified": verify,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    table = pa.Table.from_pylist(rows, SENTENCE_SCHEMA).replace_schema_metadata(
        {b"export": json.dumps(stats).encode()}
    )
    tmp = path.with_suffix(".partial")
    try:
        pq.write_table(table, tmp)
        tmp.rename(path)
    finally:
        tmp.unlink(missing_ok=True)
    return stats
