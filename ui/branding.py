"""Product name, logo and page header shared by every page."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

PRODUCT_NAME = "Resolver Support"
TAGLINE = "AI-assisted ticket resolution, grounded in your support history"
ICON_PATH = Path(__file__).resolve().parent / "assets" / "icon.svg"


def _icon_data_uri() -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(ICON_PATH.read_bytes()).decode()


def render_header() -> None:
    """Logo + product name + tagline. Colours inherit from the active theme."""
    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;padding:2px 0 14px;'
        'margin-bottom:6px;border-bottom:1px solid rgba(128,128,128,0.25)">'
        f'<img src="{_icon_data_uri()}" width="48" height="48" alt="{PRODUCT_NAME} logo" />'
        '<div><div style="font-size:1.75rem;font-weight:700;line-height:1.15;letter-spacing:-0.01em">'
        f'Resolver <span style="font-weight:400;opacity:0.7">Support</span></div>'
        f'<div style="font-size:0.9rem;opacity:0.65;margin-top:2px">{TAGLINE}</div></div></div>',
        unsafe_allow_html=True,
    )
