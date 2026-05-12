# Testing

Layered test strategy. ~940 tests across the suite; full regression
runs in ~80s on CPU.

## Layers

| Layer | Where | Wall | Purpose |
|---|---|---|---|
| Unit (domain) | `tests/unit/domain/` | <1s | Pure-Python types, codegen, whitelist, hex compliance |
| Unit (application) | `tests/unit/application/` | <1s each | Use-case logic with mocked ports |
| Unit (adapters) | `tests/unit/adapters/` | <1s each | Adapter-specific (evolution_doc renderer, etc.) |
| Integration (frontend) | `tests/integration/frontend/` | ~30s | Streamlit AppTest — 49 tests covering Submit / Experiments / Run detail / Comparison |
| Integration (dispatch matrix) | `tests/integration/dispatch_matrix/` | ~6s | (ER1, LERO) × (local, OVH-mocked) — 11 tests pinning every dispatch path |
| Integration (LERO) | `tests/integration/lero/` | ~20s slow | Orchestrator + BenchMARL adapters + trace writer + evolution doc + smoke E2E |
| Reproducibility | `tests/reproducibility/` | ~10s | Config-parity vs published configs; same-seed determinism |
| Smoke (slow) | `tests/integration/smoke/` | ~30s | Discovery MAPPO end-to-end with video recording (skipped on headless) |
| Playwright (opt-in) | `tests/integration/playwright/` | ~10s each | Real-Chromium browser tests; skipped without `[playwright]` extra |

## Running

```bash
pytest -q                           # all but slow + playwright (default)
pytest -q --ignore=tests/integration/smoke   # excludes the video tests
pytest -m slow -q                   # slow tests only
pytest -m playwright -q             # browser tests (needs playwright install chromium)
```

## Dispatch matrix specifically

`tests/integration/dispatch_matrix/test_dispatch_matrix.py` pins the
two contracts every submission must satisfy:

1. **CLI ↔ Streamlit cfg parity** — picking a YAML through the Submit
   page produces a cfg dict byte-equal to `ExperimentConfig.from_yaml(yaml)`.
   Catches regressions in widget rendering / form assembly.
2. **End-to-end submission** — `submit_to_local` produces the full
   run-dir; OVH-mocked verifies the cfg + secret_env shipped to
   `OvhRunner`.

When you change anything in `application/submission.py`,
`frontend/forms.py`, or the LERO/LLM widget code, expect this suite to
catch any drift.

## Helpers

- `tests/integration/dispatch_matrix/_helpers.py` — YAML factories
  (`er1_smoke_cfg`, `lero_smoke_cfg`, `ovh_smoke_cfg`,
  `write_smoke_yaml`) + run-dir assertions
  (`assert_er1_run_dir_complete`, `assert_lero_run_dir_complete`).
- `tests/integration/frontend/_submit_helpers.py` — reusable Submit
  page drivers (`drive_pick`, `drive_run_preflight`, `drive_submit`)
  and assertions (`assert_form_clean`, `assert_form_dirty`,
  `assert_preflight_pass/fail`, `assert_lero_widgets_rendered`).

## Playwright (opt-in)

Real browser coverage for visual / JS-side regressions. Install:

```bash
pip install -e '.[playwright]'
playwright install chromium  # ~150 MiB
pytest -m playwright
```

3 starter tests at `tests/integration/playwright/test_submit_browser.py`
demonstrate the pattern (data-testid selectors, session-scoped
Streamlit fixture). Add more as visual regressions arise.

## CI

GitHub Actions workflows at `.github/workflows/`:

- `tests-linux.yml` / `tests-mac.yml` / `tests-windows.yml` — pytest matrix.
- `pre_commit.yml` — lint + format on every push.
