"""Resolver Support -- Streamlit entrypoint.

    streamlit run ui/resolver_support.py

(Not named `app.py`: Streamlit puts this folder on sys.path, so a `ui/app.py`
would shadow the `app/` package and break `import app.config`.)

Owns everything shared across pages (page config, logo, header, navigation,
and the one-time data bootstrap below); each page lives in `ui/views/`.
Pages themselves contain NO Chroma/OpenAI calls of their own -- the chat page
only talks to `RAGService`, the Evals page only to `evals/`. That boundary
lets Phase 2 point the UI at a FastAPI backend without restructuring the
pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vectorstore.chroma_store import VectorStoreService  # noqa: E402
from app.ingestion.bootstrap import ensure_ingested  # noqa: E402
from ui.branding import ICON_PATH, PRODUCT_NAME, render_header  # noqa: E402

st.set_page_config(page_title=PRODUCT_NAME, page_icon=str(ICON_PATH), layout="wide")
st.logo(str(ICON_PATH), size="large")
render_header()


@st.cache_resource
def _bootstrap_data() -> None:
    """Ingest the checked-in dataset if the vector store is empty (once per process).

    Deployments with an ephemeral filesystem (e.g. Streamlit Community Cloud)
    start every fresh container with an empty chroma_db/ and no shell access to
    run scripts/ingest.py -- this makes a freshly booted app self-sufficient
    instead of silently answering "not enough evidence" to everything. A
    failure here (e.g. a bad API key) is shown as a banner, not a crash --
    `st.cache_resource` replays whatever this function rendered on every
    subsequent page load, so the message stays visible until it's fixed and
    the app is rebooted, without retrying (and re-erroring) on every rerun.
    """
    try:
        vector_store = VectorStoreService()
        if vector_store.collection_info()["count"] == 0:
            with st.spinner("Setting up the knowledge base for the first time (this can take a few minutes)…"):
                ensure_ingested(vector_store)
    except Exception as exc:
        st.error(f"Could not set up the knowledge base automatically: {exc}")


_bootstrap_data()

navigation = st.navigation(
    [
        st.Page("views/chat.py", title="Resolver", icon=":material/support_agent:", default=True),
        st.Page("views/evals.py", title="Evals", icon=":material/analytics:"),
    ]
)
navigation.run()
