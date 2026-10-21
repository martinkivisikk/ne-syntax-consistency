import os
from pathlib import Path

from dotenv import load_dotenv
from estnltk.storage.postgres import PostgresStorage

# Project root .env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

REQUIRED_VARS = ["KK_HOST", "KK_PORT", "KK_DBNAME", "KK_USER", "KK_PASSWORD"]


def get_storage() -> PostgresStorage:
    """Connect to the Koondkorpus database using credentials from .env."""
    load_dotenv(ENV_FILE)
    missing = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing:
        raise RuntimeError(f"Missing database settings: {', '.join(missing)} (see .env.example)")
    return PostgresStorage(
        host=os.environ["KK_HOST"],
        port=int(os.environ["KK_PORT"]),
        dbname=os.environ["KK_DBNAME"],
        user=os.environ["KK_USER"],
        password=os.environ["KK_PASSWORD"],
        schema=os.environ.get("KK_SCHEMA", "koondkorpus"),
        role=os.environ.get("KK_ROLE", "koondkorpus_read"),
    )
