"""Resolver Support -- Streamlit entrypoint.

    streamlit run ui/app.py

Owns everything shared across pages (page config, logo, header, navigation);
each page lives in `ui/views/`. Pages contain NO Chroma/OpenAI calls of their
own -- the chat page only talks to `RAGService`, the Evals page only to
`evals/`. That boundary lets Phase 2 point the UI at a FastAPI backend without
restructuring the pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.branding import ICON_PATH, PRODUCT_NAME, render_header  # noqa: E402

st.set_page_config(page_title=PRODUCT_NAME, page_icon=str(ICON_PATH), layout="wide")
st.logo(str(ICON_PATH), size="large")
render_header()

navigation = st.navigation(
    [
        st.Page("views/chat.py", title="Resolver", icon=":material/support_agent:", default=True),
        st.Page("views/evals.py", title="Evals", icon=":material/analytics:"),
    ]
)
navigation.run()
