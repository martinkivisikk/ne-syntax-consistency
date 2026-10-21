from estnltk import Text

from ne_syntax_consistency.db import get_storage

COLLECTION = "koondkorpus_sentences"
LAYERS = ["sentences", "morph_analysis", "v171_named_entities", "v172_stanza_syntax"]


def load_sample(
    n: int, layers: list[str] = LAYERS, collection: str = COLLECTION
) -> list[tuple[int, Text]]:
    storage = get_storage()
    try:
        return list(storage[collection].select(layers=layers).head(n))
    finally:
        storage.close()
