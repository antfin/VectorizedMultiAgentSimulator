# Submitting experiments

**The Streamlit Submit page is the canonical entry point** for running
experiments (both local and OVH). The CLI exists as a developer
convenience; both fronts route through the same dispatch code.

## Via Streamlit (preferred)

```bash
streamlit run src/multi_scenario/frontend/streamlit_app.py
# Open http://localhost:8501, click "Submit" in the sidebar.
```

5-step workflow:

1. **Pick** — cascading picker walks `experiments/<scenario>/<folder>/configs/*.yaml`.
2. **Inspect & edit** — pre-filled form with widgets per scenario/algorithm parameter. For LERO YAMLs, additional **LERO** + **LLM** widgets appear (see [Submit page](../frontend/submit_page.md)).
3. **Save** — only shown when you've edited fields; forces "save as new" so the original YAML stays clean.
4. **Preflight** — three LED-rolled-up cards (Configuration / System / Storage) run real probes (cuda available, OVH bucket reachable, API key present for LERO, etc.).
5. **Submit** — gated on preflight green. After click: status panel with auto-poll for OVH jobs; on DONE, results auto-pull to the local run-dir.

## Via CLI

```bash
# Local run, full lifecycle in this process.
python -m multi_scenario.cli run experiments/discovery/baseline/configs/baseline.yaml

# OVH submission; auto-uploads code on hash drift, returns job_id.
python -m multi_scenario.cli run --runner ovh experiments/discovery/lero/configs/lero_s3b_local.yaml

# Same, but emit a parseable JSON record on the last line — useful for
# scripted / chat-driven pipelines.
python -m multi_scenario.cli run --json experiments/.../config.yaml | jq -r .run_id

# OVH submit + wait-for-DONE + auto-pull results.
python -m multi_scenario.cli sweep --follow --runner ovh experiments/.../config.yaml
```

Full CLI reference: [CLI → `multi-scenario run`](../cli/run.md).

## What lands on disk

For both fronts, results land under `<storage.path>/<run_id>__<timestamp>/`:

```text
input/{config.json, provenance.json}
output/
  metrics.json          # M1–M9 aggregate
  report.json           # run-end manifest
  eval_episodes.json    # per-episode raw eval data
  benchmarl/<run>/      # BenchMARL training output + final checkpoint
  videos/               # before/after-training mp4s (if not headless)
  lero/                 # LERO-specific: only present for LERO runs
    final_summary.json
    evolution_history.json
    evolution_doc.md    # human-readable narrative
    prompts/iter_<N>/   # rendered prompts + per-candidate responses
logs/run.log
run_state.json
```

See [Reproducibility → LERO S3b-local reproduction](../reproducibility/lero_s3b_local_reproduction.md) for what a published comparison looks like.
