# Installation

```bash
git clone <repo>
cd multi_scenario  # or coopvmas after F10.6 extraction
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
pytest -q  # sanity smoke; expect ~940 passes
```

## Optional extras

```bash
pip install -e '.[playwright]'   # real-browser Submit page tests
playwright install chromium      # ~150 MiB download

pip install -e '.[docs]'         # mkdocs-material to preview this site
mkdocs serve                     # local doc preview at http://127.0.0.1:8000
```

## OVH setup

If you'll submit to OVH AI Training: [Operations → OVH setup](../operations/ovh_setup.md).

## API key for LERO

LERO needs an LLM API key in your env or project `.env`:

```bash
echo 'OPENAI_API_KEY=sk-…' >> .env  # or ANTHROPIC_API_KEY
```

The Streamlit Submit page's preflight will warn if it's missing.
