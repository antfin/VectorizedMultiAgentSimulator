# Frontend (Streamlit)

The Streamlit app under `src/multi_scenario/frontend/` exposes 4 main pages:

| Page | What you do there |
|---|---|
| [Submit](submit_page.md) | Pick a YAML, edit knobs, run preflight, fire local or OVH |
| [Experiments](experiments_page.md) | Browse all completed runs across the experiments root |
| [Run detail](run_detail_page.md) | Drill into one run — metrics, scalars, videos, LERO trace |
| [Comparison](comparison_page.md) | Cross-run plots, leaderboards, sweep aggregation |

Launch:

```bash
streamlit run src/multi_scenario/frontend/streamlit_app.py
# http://localhost:8501
```

## Sidebar — experiments root

The sidebar's "Experiments root" picker scopes every page to one
directory. Default = `experiments/` under the repo. Set via
`MULTI_SCENARIO_EXPERIMENTS_ROOT` env var, or switch interactively.

## Test coverage

- AppTest end-to-end suite at `tests/integration/frontend/test_submit_page_e2e.py` (16 tests).
- Dispatch matrix at `tests/integration/dispatch_matrix/test_dispatch_matrix.py` (9 tests).
- Helper smoke tests at `tests/integration/frontend/test_submit_helpers_smoke.py` (7 tests).
- Opt-in Playwright tests at `tests/integration/playwright/test_submit_browser.py` (3 tests, skip when `[playwright]` extra not installed).

See [Operations → Testing](../operations/testing.md).
