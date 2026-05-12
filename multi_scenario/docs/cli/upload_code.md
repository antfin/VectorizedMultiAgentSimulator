# `multi-scenario upload-code`

Push the local source tree to the OVH code bucket so OVH containers
`pip install -e /workspace/code` against the latest version.

```bash
multi-scenario upload-code [--ovh-config configs/ovh.yaml] [--dry-run]
```

## Auto-upload on `multi-scenario run --runner ovh`

You usually don't call this directly — `multi-scenario run --runner ovh`
auto-uploads when the local code hash drifts from the bucket's
`.code_hash` blob. Explicit call is useful when:

- You want to see what files would upload (`--dry-run`).
- You're staging code for a sweep where multiple jobs will share the same upload.
- You hit an OVH dispatch failure and want to verify the bucket contents.

## What gets uploaded

Whitelist-based — see `DEFAULT_INCLUDE_DIRS` / `DEFAULT_EXCLUDE_PATTERNS`
in `adapters/storage/code_uploader.py`:

- **Included**: `src/multi_scenario/`, `experiments/` (configs only), `configs/`, `pyproject.toml`, `README.md`.
- **Excluded**: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `*/results/`, `*/output/`, `*/logs/`, run-folder timestamp dirs.

A `.code_hash` blob is written alongside; Streamlit's preflight uses
it to surface "your local code is newer than the bucket" warnings.
