"""F8.2.F — doc_links helper contract.

Single source of truth for "where is the docs site hosted?" — used by
metric tooltips, the Docs sidebar page, and any future inline
"📖 Learn more" deep-link. A drift between this helper and the actual
Streamlit/mkdocs deployment URL would silently produce broken links;
these tests pin the resolution rules.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

from multi_scenario.frontend.doc_links import doc_icon_link, doc_link, docs_base_url


def test_default_url_is_local_mkdocs_serve(monkeypatch):
    """Without env var, default to ``mkdocs serve``'s :8000 — works for local dev."""
    monkeypatch.delenv("MULTI_SCENARIO_DOCS_URL", raising=False)
    assert docs_base_url() == "http://127.0.0.1:8000"


def test_env_var_overrides_default(monkeypatch):
    """``MULTI_SCENARIO_DOCS_URL`` wins; trailing slash gets normalised away."""
    monkeypatch.setenv("MULTI_SCENARIO_DOCS_URL", "https://example.github.io/coopvmas/")
    assert docs_base_url() == "https://example.github.io/coopvmas"


def test_doc_link_combines_base_and_slug(monkeypatch):
    monkeypatch.setenv("MULTI_SCENARIO_DOCS_URL", "https://docs.example.com")
    url = doc_link("scenarios/discovery#m1-success-rate")
    assert url == "https://docs.example.com/scenarios/discovery#m1-success-rate"


def test_doc_link_strips_leading_slash_from_slug(monkeypatch):
    """``/foo`` and ``foo`` should resolve identically — defensive against caller typos."""
    monkeypatch.setenv("MULTI_SCENARIO_DOCS_URL", "https://docs.example.com")
    assert doc_link("/foo") == doc_link("foo") == "https://docs.example.com/foo"


def test_doc_icon_link_returns_markdown(monkeypatch):
    monkeypatch.setenv("MULTI_SCENARIO_DOCS_URL", "https://docs.example.com")
    md = doc_icon_link("metrics#m1", label="Read M1 details")
    assert md.startswith("📖")
    assert "Read M1 details" in md
    assert "https://docs.example.com/metrics#m1" in md


def test_metric_tooltip_includes_learn_more_link(monkeypatch):
    """F8.2.E + F8.2.F integration: tooltip ends with the "Learn more" link."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.metrics_glossary import tooltip_text

    monkeypatch.setenv("MULTI_SCENARIO_DOCS_URL", "https://docs.example.com")
    text = tooltip_text("M1_success_rate")
    assert "📖" in text
    assert "Learn more" in text
    # The slug for M1 from the glossary should be present in the URL.
    assert "https://docs.example.com/results-analysis/metrics#m1-success-rate" in text
