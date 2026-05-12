"""Application-layer use cases for submitting an :class:`ExperimentConfig`.

Two entry points (CLI's ``multi-scenario run`` and the Streamlit Submit page)
were each building OvhRunner + computing the run dir + assembling the S3
prefix in their own glue functions. F7.7.A6 collapses that into one function
per dispatch target so behavioural differences can't accidentally appear:

- :func:`submit_to_ovh` — orchestration for the OVH path.
- :func:`submit_to_local` — orchestration for the local path.

Caller-specific concerns (error wrapping, logger choice, what to do with the
result) stay in the caller; only the orchestration lives here. Hex-clean:
this module imports from ``adapters/`` and ``domain/`` only — never from
``cli/`` or ``frontend/``.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.adapters.runners.ovh import OvhRunner
from multi_scenario.adapters.runners.ovh_cli import OvhClient
from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
from multi_scenario.domain.models import ExperimentConfig, OvhJobConfig, RunId


class _Logger(Protocol):
    """Subset of the logger Protocol :class:`OvhRunner` needs."""

    # pylint: disable=missing-function-docstring,too-few-public-methods
    def info(self, msg: str) -> None:
        ...

    def debug(self, msg: str) -> None:
        ...

    def warning(self, msg: str) -> None:
        ...

    def error(self, msg: str) -> None:
        ...


@dataclass(frozen=True)
class OvhSubmission:
    """Result of one OVH submission — everything callers surface to the user."""

    job_id: str
    run_id: RunId
    run_dir: Path
    s3_prefix: str  # ``<bucket>@<region>/<run_dir.name>`` (per-run; Stage 1 lesson)
    dashboard_url: str  # link to the OVH manager UI


@dataclass(frozen=True)
class LocalSubmission:
    """Result of one local synchronous run."""

    run_id: str
    run_dir: Path
    experiment_result: Any  # multi_scenario.domain.models.ExperimentResult


def build_run_dir(cfg: ExperimentConfig, *, mkdir: bool = False) -> tuple[RunId, Path]:
    """Compute ``<storage.path>/<run_id>__<UTC-timestamp>/`` for ``cfg``.

    Shared between local + OVH dispatch so the run-dir convention is
    single-source.

    ``mkdir`` defaults to **False**: pure-data, no I/O. Callers that need
    the dir to exist (local: FileLogger writes ``logs/run.log`` immediately)
    pass ``mkdir=True``. OVH callers leave it False — the path is just a
    *label* used to build the S3 prefix; the OVH container creates its own
    workspace inside the bucket mount, never touches the local filesystem.

    On macOS / read-only hosts, mkdir-ing a container mount path
    (e.g. ``/workspace/results``) raises ``OSError [Errno 30]``. The
    Streamlit Submit page hit this on the first OVH click after F7.7.A6
    (2026-05-10) — that's why the default flipped to False.

    OVH-container short-circuit (``MULTI_SCENARIO_USE_STORAGE_ROOT_AS_RUN_DIR``):
    when set (OvhRunner injects it before ``ovhai job run``), the function
    returns ``(run_id, storage_root)`` directly — no nested timestamp segment.
    This is required because the OVH-side S3 prefix is *already* the per-run
    folder (``<bucket>/<host_run_dir_name>/``); without this short-circuit
    the container creates a SECOND timestamped subdir inside the prefix,
    yielding nested ``<host_dir>/<container_dir>/input/...`` after pullback —
    Run Detail can't find it. The first OVH smoke after Stage 1 hit this
    (job d0237edc-…, 2026-05-10) — that's why this branch exists.
    """
    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    storage_root = (
        Path(cfg.runtime.storage.path)
        if cfg.runtime is not None
        else Path("experiments")
    )
    if os.environ.get("MULTI_SCENARIO_USE_STORAGE_ROOT_AS_RUN_DIR"):
        # Container reuses the host's per-run S3 prefix as its run-dir.
        run_dir = storage_root
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = storage_root / run_id.folder_name(timestamp)
    if mkdir:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def submit_to_ovh(
    cfg: ExperimentConfig,
    *,
    ovh_cfg: OvhJobConfig,
    yaml_path_in_repo: str,
    run_dir: Path,
    logger: _Logger,
    client: OvhClient | None = None,
    secrets: FernetSecretsAdapter | None = None,
) -> OvhSubmission:
    """Construct OvhRunner, submit, return the user-visible bundle.

    ``client`` and ``secrets`` are injectable for tests. Defaults instantiate
    fresh real adapters — same behaviour the previous duplicated glue had.

    F8.4 Phase 2.5: when ``cfg.lero is not None``, collect LLM API keys
    from the submitter's local environment (``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY`` / ``OVH_API_KEY``) and ship them via the
    Fernet-encrypted ``secret_env`` channel. The in-container
    ``experiment_service._run_lero`` decrypts them into ``os.environ``
    before constructing the LLM client (see :mod:`secrets_priming`).

    Doesn't catch exceptions: ``OvhRunner.submit`` raises
    :class:`OvhCliError` (via :class:`OvhClient`) on failure; callers wrap
    according to their UX (CLI re-raises, Streamlit writes to session_state).
    """
    secret_env, secret_passphrase = _collect_lero_secret_env(cfg)
    runner = OvhRunner(
        ovh_config=ovh_cfg,
        client=client or OvhClient(),
        secrets=secrets or FernetSecretsAdapter(),
        logger=logger,
        yaml_path_in_repo=yaml_path_in_repo,
        secret_env=secret_env,
        secret_passphrase=secret_passphrase,
    )
    job_id = runner.submit(cfg, run_dir)
    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    # ``s3_prefix`` mirrors what OvhRunner actually used for the volume URI
    # (``<bucket>/<run_dir.name>`` since Stage 1) — so users + auto-sync look
    # in the same place. Pre-Stage 1 it was ``<bucket>/<run_id>`` and re-runs
    # of same exp_id+seed clobbered each other.
    return OvhSubmission(
        job_id=job_id,
        run_id=run_id,
        run_dir=run_dir,
        s3_prefix=f"{ovh_cfg.bucket_results}@{ovh_cfg.region}/{run_dir.name}",
        dashboard_url=(
            f"https://www.ovh.com/manager/#/dedicated/aiTraining/jobs/{job_id}"
        ),
    )


#: LLM-provider API keys that LERO needs in-container. Order matters
#: for the F9.8 preflight check, but here we just collect any that
#: the submitter exported locally; LiteLLM picks the right one per
#: model prefix.
_LERO_API_KEY_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OVH_API_KEY",
)


def _collect_lero_secret_env(
    cfg: ExperimentConfig,
) -> tuple[dict[str, str] | None, str | None]:
    """Build the Fernet-encrypted secret_env when ``cfg.lero`` is set.

    Returns ``(secret_env, passphrase)`` where:
    - ``secret_env`` is the dict of LLM API keys present in the
      submitter's local environment, OR ``None`` when none are set
      OR ``cfg.lero is None`` (non-LERO submissions don't need keys).
    - ``passphrase`` is a freshly-generated 32-byte urlsafe token
      that decrypts the blob in-container, OR ``None`` matching the
      env-dict.

    The passphrase is held only in this process and in the OVH job's
    ``--env`` ciphertext — never written to disk locally, never
    persisted in the OVH dashboard's job-args (it's just an env var
    on the running container).
    """
    # pylint: disable=import-outside-toplevel
    import os
    import secrets as _secrets_module

    if cfg.lero is None:
        return None, None
    # Best-effort ``.env`` autoload before reading ``os.environ`` — mirrors
    # the convenience of :func:`litellm_adapter._load_env_once` but at
    # submit-time, so a user who keeps ``OPENAI_API_KEY`` in
    # ``multi_scenario/.env`` (or the repo-root ``.env``) doesn't have
    # to remember to ``export`` it before ``multi-scenario run``.
    # ``override=False`` means a shell ``export`` still wins, so this
    # never overrides an intentional value.
    try:
        from dotenv import load_dotenv

        here = Path.cwd()
        for parent in (here, *here.parents):
            candidate = parent / ".env"
            if candidate.is_file():
                load_dotenv(candidate, override=False)
                break
    except ImportError:
        pass
    collected = {k: v for k in _LERO_API_KEY_ENV_VARS if (v := os.environ.get(k))}
    if not collected:
        # No keys to ship — leave the OvhRunner without secret_env.
        # The in-container LERO will then fail at LiteLLM call time
        # with a clear 401, which is a better signal than silently
        # shipping nothing and getting a confusing crash.
        return None, None
    passphrase = _secrets_module.token_urlsafe(32)
    return collected, passphrase


def submit_to_local(
    cfg: ExperimentConfig,
    *,
    run_dir: Path,
    logger: _Logger,
) -> LocalSubmission:
    """Construct LocalRunner, run synchronously, return the bundle.

    Pre-flight: ``LocalRunner._assert_cuda_available()`` runs inside
    ``LocalRunner.run`` when ``device=cuda``; we don't double-check here.

    Doesn't catch exceptions: training crashes propagate as RuntimeError /
    BenchMARL-internal types; callers wrap according to their UX.
    """
    runner = LocalRunner(logger=logger)
    result = runner.run(cfg, run_dir=run_dir)
    return LocalSubmission(
        run_id=result.run_id,
        run_dir=run_dir,
        experiment_result=result,
    )
