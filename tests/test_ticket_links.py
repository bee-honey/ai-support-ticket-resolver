"""ticket_page_link(): AppTest can't deeply inspect st.page_link (shows as UnknownElement),
so its exact arguments are verified directly against a mocked streamlit.page_link instead."""

from __future__ import annotations

from unittest.mock import patch

from ui.ticket_links import TICKETS_PAGE, ticket_page_link


def test_links_to_the_tickets_page_with_the_ticket_id_as_a_query_param():
    with patch("ui.ticket_links.st.page_link") as page_link:
        ticket_page_link("MESOS-1873")
    page_link.assert_called_once_with(
        TICKETS_PAGE, label="MESOS-1873", icon=":material/confirmation_number:", query_params={"ticket": "MESOS-1873"}
    )


def test_custom_label_is_used_but_the_query_param_still_carries_the_real_id():
    with patch("ui.ticket_links.st.page_link") as page_link:
        ticket_page_link("MESOS-1873", label="MESOS-1873 — Fetcher doesn't extract .tar files")
    _, kwargs = page_link.call_args
    assert kwargs["label"] == "MESOS-1873 — Fetcher doesn't extract .tar files"
    assert kwargs["query_params"] == {"ticket": "MESOS-1873"}
