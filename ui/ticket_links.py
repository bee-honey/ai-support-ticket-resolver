"""Shared helper so every page links to a ticket the same way.

A click sets `?ticket=<id>` and navigates to the Tickets page, which reads
that query param on load (see `views/tickets.py`) and jumps straight to that
ticket's detail view -- so a citation in the chatbot's answer, or a retrieved
ticket ID in an Evals trace, can be verified by eye in one click instead of
trusted blind.
"""

from __future__ import annotations

import streamlit as st

TICKETS_PAGE = "views/tickets.py"


def ticket_page_link(ticket_id: str, label: str | None = None) -> None:
    st.page_link(
        TICKETS_PAGE,
        label=label or ticket_id,
        icon=":material/confirmation_number:",
        query_params={"ticket": ticket_id},
    )
