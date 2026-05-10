"""VideoRecorder — rolls out a TorchRL/BenchMARL policy with rendering and writes MP4.

Reuses the BenchMARL ``Experiment``'s ``test_env`` (a ``TransformedEnv`` wrapping
a ``VmasEnv``) and ``policy`` directly — no separate VMAS env, no state-dict
reconstruction. Frames come from the underlying VMAS ``Environment.render``
(``test_env.base_env._env``); MP4 encoding via ``imageio[ffmpeg]``.

Recording is driven by ``BenchmarlBaseAdapter.train`` at two points: just after
the experiment is constructed (random-init policy → ``before_training.mp4``)
and just after ``experiment.run`` returns (trained policy → ``after_training.mp4``).
"""

import logging
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type


# F8.2.C — per-recorder safety cap. Today's 2-video flow (before + after)
# never approaches this. The cap is forward-looking: if a future feature
# enables many recordings (per-eval video, multi-seed comparison montages,
# etc.), we don't want a misconfigured run to fill the disk. 10 was chosen
# as "enough for any realistic per-run debugging surface, small enough to
# stay under ~50 MB total at typical resolutions".
_MAX_VIDEOS_PER_RUN = 10

_log = logging.getLogger(__name__)


class VideoRecorder:
    """Rolls out one episode through ``test_env`` + ``policy`` and writes MP4."""

    # One public method is the whole point of a recorder; the call needs five
    # legitimately distinct args (env / policy / horizon / path / fps) so
    # too-few-public-methods + too-many-arguments + too-many-positional both
    # apply. Bundling them into a config dataclass would be pure ceremony.
    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments

    def __init__(self) -> None:
        # F8.2.C: per-instance counter; the cap is per-recorder so a regen
        # invocation starts fresh (different recorder instance) — only way
        # to exceed it is to wire a multi-video mode against ONE recorder.
        self._n_recorded = 0

    def record(
        self,
        test_env: Any,
        policy: Any,
        max_steps: int,
        output_path: Path,
        fps: int = 15,
    ) -> None:
        """Run one rendered episode; write frames to ``output_path`` as MP4.

        Single-env (``env_index=0``) capture; ``num_envs > 1`` test envs still
        produce one video for the first env's trajectory.

        F8.2.C: after ``_MAX_VIDEOS_PER_RUN`` successful records on the same
        instance, further calls log a warning and no-op (no MP4 written).
        Today's 2-video flow never trips this; the cap protects against
        future per-eval-video or comparison-montage features.
        """
        if self._n_recorded >= _MAX_VIDEOS_PER_RUN:
            _log.warning(
                "video cap reached (%d) — skipping further recordings",
                _MAX_VIDEOS_PER_RUN,
            )
            return
        # The recorder is intentionally tied to the BenchMARL/VMAS env shape;
        # the descent through ``base_env._env`` is documented in F2.11.
        # pylint: disable=protected-access
        vmas_env = test_env.base_env._env

        frames: list[Any] = []
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = test_env.reset()
            for _ in range(max_steps):
                td = policy(td)
                td = test_env.step(td)
                frame = vmas_env.render(mode="rgb_array", env_index=0)
                if frame is not None:
                    frames.append(frame)
                # Step → next state via TorchRL's "next" key; advance for the next iteration.
                td = td["next"]
                # Break only when env_index=0 (the one we render) is done.
                # Pre-fix: ``td["done"].any()`` triggered when ANY of the
                # ``num_envs`` parallel envs reached done — for ER1
                # (num_envs=600) at least one always finished within a few
                # steps, clipping the video to a couple of seconds even when
                # env 0 was mid-episode. This bug was caught after ER1
                # produced 2-second videos despite max_steps=200.
                if "done" in td.keys() and _env0_done(td["done"]):
                    break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")
        self._n_recorded += 1


def _env0_done(done: Any) -> bool:
    """True iff env_index=0 is done.

    BenchMARL's ``done`` tensor shape varies by setup:
    - single-env (``num_envs=1``): ``[1]`` or ``[1, n_agents]`` or ``[]``.
    - multi-env (``num_envs=N>1``): ``[N]`` or ``[N, n_agents]``.

    We render env_index=0 only (per the recorder's design), so we want the
    loop to terminate when env 0 is done — independent of the other envs.
    Pre-fix this used ``done.any()`` over the whole tensor; with
    ``num_envs=600`` (ER1) at least one env always finishes within a few
    steps, clipping the video to a couple of seconds even when env 0 was
    mid-episode (Smoke 2026-05-10 lesson).
    """
    if not torch.is_tensor(done):
        # Defensive — TorchRL always returns tensors here, but if a future
        # version returns a Python bool / list, fall through to "done means
        # done" semantics so we don't accidentally never-stop.
        return bool(done)
    if done.dim() == 0:
        return bool(done.item())
    # First leading dim is the env axis; collapse any per-agent / per-axis
    # tail dims with .any() within env 0's slice.
    return bool(done[0].any().item())
