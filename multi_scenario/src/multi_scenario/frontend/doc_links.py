"""F8.2.F — central helper for building deep-links to the mkdocs site.

The mkdocs site (built by F10.1) is the canonical narrative documentation;
Streamlit pages link into it from metric tiles, preflight checks, scenario
field labels, and the new ``📖 Docs`` sidebar entry. ``doc_link(slug)`` is
the single source-of-truth for "where is the docs site hosted right now?"
so a future deployment URL change is one variable, not a hundred.

Resolution order:
1. ``MULTI_SCENARIO_DOCS_URL`` env var — a full URL like
   ``https://example.github.io/coopvmas`` (no trailing slash). This is what
   the deployed site (F10.1's GitHub Pages job) sets in the production
   Streamlit container.
2. Fallback to ``http://127.0.0.1:8000`` — the default that ``mkdocs serve``
   exposes during local dev. So a developer running ``mkdocs serve`` in one
   terminal + ``streamlit run`` in another gets working deep-links for free.

``doc_link("scenarios/discovery#m1-success-rate")`` →
``http://127.0.0.1:8000/scenarios/discovery#m1-success-rate`` (or the
deployed equivalent when the env var is set).
"""

import os

_DOCS_URL_ENV = "MULTI_SCENARIO_DOCS_URL"
_DEFAULT_LOCAL_URL = "http://127.0.0.1:8000"


def docs_base_url() -> str:
    """Resolve the docs site's base URL — no trailing slash."""
    raw = os.environ.get(_DOCS_URL_ENV) or _DEFAULT_LOCAL_URL
    return raw.rstrip("/")


def doc_link(slug: str) -> str:
    """Build a full URL into the mkdocs site for ``slug`` (e.g. ``"path#anchor"``)."""
    base = docs_base_url()
    slug = slug.lstrip("/")
    return f"{base}/{slug}"


def doc_icon_link(slug: str, *, label: str = "Learn more") -> str:
    """Markdown for an inline ``📖 [Learn more →](<url>)`` deep-link snippet."""
    return f"📖 [{label} →]({doc_link(slug)})"
