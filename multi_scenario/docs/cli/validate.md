# `multi-scenario validate`

Validate a YAML against the `ExperimentConfig` schema without running.

```bash
multi-scenario validate <yaml_path>
```

Catches:

- Schema violations (missing required keys, wrong types).
- Cross-field invariants (`minibatch_size <= frames_per_batch`, `cfg.lero requires cfg.llm`, etc.).
- Unknown scenario / algorithm names.

Exit code `0` on valid, `2` on schema failure. Used by CI and the
Submit page's preflight under "Config schema valid".
