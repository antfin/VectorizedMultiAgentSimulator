# Scenarios

Four VMAS scenarios supported, all under `adapters/scenarios/`:

| Scenario | Task | Default n / t / k | Status |
|---|---|---|---|
| [Discovery](discovery.md) | Rendezvous coverage (k agents per target) | 4 / 4 / 2 | F8.4 reference, LERO target |
| [Navigation](navigation.md) | Goal navigation with obstacles | 4 / 4 / — | F2.4 adapter only; campaign deferred to F11.2 |
| [Transport](transport.md) | Cooperative object transport | 4 / 1 / — | F2.4 adapter only; campaign deferred to F11.3 |
| [Flocking](flocking.md) | Boids-style alignment | 4 / — / — | F2.4 adapter only; campaign deferred to F11.4 |

Adapter contract: implement `Scenario` Protocol (`make_env`,
`success_predicate`, `coverage_progress`, `utilization_predicate`,
`has_comm`, `default_params`). Adding a new VMAS scenario = one new
file under `adapters/scenarios/`.
