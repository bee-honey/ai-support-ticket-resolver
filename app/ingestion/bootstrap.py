"""One-time data bootstrap for a fresh deployment.

Streamlit Community Cloud (and similar platforms) give the app an ephemeral
filesystem: chroma_db/ is empty on every fresh container -- it's gitignored,
so it's never part of the repo clone, and anything written to disk at runtime
is wiped on the next restart (sleep/wake, reboot, or a new deploy). There's
also no shell access there to run `scripts/ingest.py` by hand.

This checks whether the vector store's collection is empty and, if so,
ingests the checked-in production dataset -- so a freshly deployed or
freshly woken app is self-sufficient. Local development is unaffected:
chroma_db/ persists across runs there, so this is a no-op once it's already
populated.
"""

from __future__ import annotations

from app.ingestion.mapping import CSVFieldMapping
from app.ingestion.pipeline import IngestionPipeline, IngestionSummary
from app.vectorstore.chroma_store import VectorStoreService

# Matches the source + mapping used to build the currently-ingested production
# dataset -- see README.md "Adjusting for the Real Mesos CSV".
DEFAULT_SOURCE = "data/mesos_scoped.csv"
DEFAULT_MAPPING = "config/mesos_mapping.json"


def ensure_ingested(
    vector_store: VectorStoreService,
    source: str = DEFAULT_SOURCE,
    mapping_path: str = DEFAULT_MAPPING,
) -> IngestionSummary | None:
    """Ingest `source` into `vector_store` only if its collection is currently empty.

    Returns the ingestion summary, or None if the collection already had data
    (the common case in local dev, where chroma_db/ persists between runs).
    """
    if vector_store.collection_info()["count"] > 0:
        return None
    mapping = CSVFieldMapping.from_json(mapping_path)
    pipeline = IngestionPipeline(vector_store=vector_store, csv_mapping=mapping)
    return pipeline.ingest(source)
