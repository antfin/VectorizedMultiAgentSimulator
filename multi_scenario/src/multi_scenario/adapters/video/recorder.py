"""VideoRecorder — rolls out a TorchRL/BenchMARL policy with rendering and writes MP4.

Reuses the BenchMARL ``Experiment``'s ``test_env`` (a ``TransformedEnv`` wrapping
a ``VmasEnv``) and ``policy`` directly — no separate VMAS env, no state-dict
reconstruction. Frames come from the underlying VMAS ``Environment.render``
(``test_env.base_env._env``); MP4 encoding via ``imageio[ffmpeg]``.

Recording is driven by ``BenchmarlBaseAdapter.train`` at two points: just after
the experiment is constructed (random-init policy → ``before_training.mp4``)
and just after ``experiment.run`` returns (trained policy → ``after_training.mp4``).
"""

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type


class VideoRecorder:
    """Rolls out one episode through ``test_env`` + ``policy`` and writes MP4."""

    # One public method is the whole point of a recorder; the call needs five
    # legitimately distinct args (env / policy / horizon / path / fps) so
    # too-few-public-methods + too-many-arguments + too-many-positional both
    # apply. Bundling them into a config dataclass would be pure ceremony.
    # pylint: disable=too-few-public-methods,too-many-arguments,too-many-positional-arguments

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
        """
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
                if "done" in td.keys() and td["done"].any():
                    break

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264")
