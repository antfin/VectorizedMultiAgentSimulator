# Reproducibility

Pinned science results — each is backed by a regression test that
asserts the number on disk matches the published claim.

| Result | Doc | Test |
|---|---|---|
| LERO S3b-local M1 = 0.795 (rendezvous_comm reproduction) + Phase 5b M1 = 0.570 (ER1 apples-to-apples) | [LERO S3b-local](lero_s3b_local_reproduction.md) | manual re-eval (see doc) |
| Same config + same seed produces byte-equal metrics | [Determinism test](determinism_test.md) | `tests/reproducibility/test_same_seed_byte_equal.py` |
| ER1 vs rendezvous_comm threshold logic | — | `tests/reproducibility/test_compare_to_reference.py` |
| LERO YAML field parity vs rendezvous_comm's `s3b_local.yaml` | — | `tests/reproducibility/test_lero_s3b_local_config_parity.py` |
| ER1 YAML field parity | — | `tests/reproducibility/test_er1_config_parity.py` |

> Citation source for rendezvous_comm history (referenced from the
> LERO comparison doc): [`docs/_drafts/rendezvous_comm_history.md`](../_drafts/rendezvous_comm_history.md).
> This is scaffolding — gets deleted at F10.8 once coopvmas has
> reproduced the relevant numbers.
