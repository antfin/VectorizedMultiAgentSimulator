"""F8.2.F — In-app documentation hub.

Embeds the mkdocs site (built by F10.1) inside Streamlit so a student
running the dashboard never has to leave to read an explanation. URL
resolution lives in :mod:`multi_scenario.frontend.doc_links` so a future
deployment-URL change is one variable, not a hundred.

When ``MULTI_SCENARIO_DOCS_URL`` is set, the page iframe-embeds that URL.
With the env var unset, it falls back to ``http://127.0.0.1:8000`` — the
default ``mkdocs serve`` port — so a dev running both servers gets a
working in-app docs view with zero config.

If the docs site isn't reachable, the page surfaces a helpful info panel
explaining how to start ``mkdocs serve`` locally (rather than rendering
an iframe pointing at nothing).
"""

# pylint: disable=wrong-import-position,invalid-name

import streamlit as st

from multi_scenario.frontend.doc_links import docs_base_url
from multi_scenario.frontend.theme import apply_theme

apply_theme(title="Documentation", subtitle="In-app reference & concepts")

base = docs_base_url()
st.markdown(f"#### 📖 Docs site &nbsp; <small>`{base}`</small>", unsafe_allow_html=True)
st.caption(
    "Set the ``MULTI_SCENARIO_DOCS_URL`` env var to point at the deployed "
    "mkdocs site (F10.1's GitHub Pages target); without it this page falls "
    "back to ``http://127.0.0.1:8000`` for local ``mkdocs serve``."
)

# Reachability hint — st.components.v1.iframe doesn't fail loudly on a
# 404 / unreachable host, so we offer guidance up-front when the
# default-local URL is in play.
if base == "http://127.0.0.1:8000":
    st.info(
        "**Local dev mode** — make sure ``mkdocs serve`` is running in another "
        "terminal so the embed below resolves. Start it with:\n\n"
        "```bash\nmkdocs serve\n```\n\n"
        "Then refresh this page."
    )

# Iframe-embed. Height roughly fills the visible viewport; ``scrolling=True``
# means navigation inside the docs site doesn't jail-break out of the embed.
st.components.v1.iframe(base, height=1200, scrolling=True)

st.divider()
st.markdown("Prefer to open the docs in a new tab? " f"[Open ``{base}`` ↗]({base})")
