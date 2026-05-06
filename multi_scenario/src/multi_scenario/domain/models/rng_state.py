"""RngState — captured state of the RNGs used during a run.

Pure-data model. The actual seeding / state-capture functions live in adapters
because they import torch / numpy / random; those land later (Phase 2+ when
``ExperimentService`` first wires up determinism).
"""

from pydantic import BaseModel, Field

from ._common import STRICT


class RngState(BaseModel):
    """Captured RNG state used during a run; persisted across resume cycles.

    ``captures`` is a ``{rng_name: encoded_state}`` dict that adapter functions
    populate (e.g. ``"python.random"``, ``"numpy"``, ``"torch.cpu"``,
    ``"torch.cuda"``). Encoding format (base64-of-pickled, hex, …) is the
    adapter's choice; the model stays format-agnostic.
    """

    model_config = STRICT

    seed: int
    captures: dict[str, str] = Field(default_factory=dict)
