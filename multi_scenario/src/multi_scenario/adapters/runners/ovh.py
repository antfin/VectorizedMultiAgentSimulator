"""OvhRunner — submits an experiment to OVH AI Training and waits for completion.

Implements the :class:`Runner` port (F1.11). ``supports_resume = False`` per
F5.7's capability flag — OVH-resume is intentionally out of scope. The
``multi-scenario resume`` CLI checks this and refuses with a helpful message.

**Scope today (F6.2):** framework wiring + submission + polling. Result
loading from S3 is the user's responsibility until F6.3 lands the
``S3StorageAdapter``; for now ``run()`` reads ``run_dir/output/metrics.json``
directly, assuming someone has synced the bucket back. Code upload to
``bucket_code`` is F6.4's job.
"""

import time
from pathlib import Path

from multi_scenario.adapters.runners.ovh_cli import OvhClient, OvhCliError
from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    OvhJobConfig,
)
from multi_scenario.domain.ports import Logger


class OvhJobError(RuntimeError):
    """Raised when an OVH job fails to terminate in DONE state."""


class OvhRunner:
    """Runner adapter that submits to OVH AI Training and blocks until done."""

    # Lots of orthogonal collaborators (cli + secrets + ovh-cfg + logger +
    # storage + secret env + passphrase + sleep); bundling them into a single
    # config object would obscure the DI lines and make tests harder to read.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    name: str = "ovh"
    # Per F5.7 capability flag: OVH-resume is intentionally out of scope.
    supports_resume: bool = False

    def __init__(
        self,
        ovh_config: OvhJobConfig,
        client: OvhClient,
        secrets: FernetSecretsAdapter,
        logger: Logger,
        storage: LocalStorageAdapter | None = None,
        s3_storage: S3StorageAdapter | None = None,
        secret_env: dict[str, str] | None = None,
        secret_passphrase: str | None = None,
        sleep: callable = time.sleep,
    ) -> None:
        self._ovh_config = ovh_config
        self._client = client
        self._secrets = secrets
        self._logger = logger
        self._storage = storage or LocalStorageAdapter()
        # When supplied, F6.3 sync_to_local pulls the run-folder back from S3
        # before we read metrics.json. Without it, the user must hand-sync.
        self._s3_storage = s3_storage
        # Optional encrypted secrets to ship as env vars on the OVH job.
        self._secret_env = secret_env or {}
        self._secret_passphrase = secret_passphrase
        self._sleep = sleep  # injectable for fast tests

    def run(
        self,
        cfg: ExperimentConfig,
        run_dir: Path,
        resume_from: Path | None = None,
    ) -> ExperimentResult:
        """Submit the run to OVH, poll until terminal state, load + return the result."""
        if resume_from is not None:
            raise OvhJobError(
                "OvhRunner does not support resume (supports_resume=False); "
                "rerun the job from scratch instead."
            )

        args = self._build_submit_args(cfg, run_dir)
        self._logger.info(f"submitting OVH job for {cfg.experiment.id}")
        job_id = self._client.submit(args)
        self._logger.info(f"OVH job submitted: id={job_id}")

        info = self._poll_until_terminal(job_id)
        if info.state.upper() != "DONE":
            tail = _safe_logs_tail(self._client, job_id)
            raise OvhJobError(f"OVH job {job_id} ended in state={info.state}; last logs:\n{tail}")
        self._logger.info(f"OVH job {job_id} DONE; loading result from {run_dir}")
        # F6.3: when an S3 storage adapter is wired, pull the whole run-folder
        # back from S3 before reading metrics.json locally. Without it, the
        # user must hand-sync (status quo until F6.3 wired into the call site).
        if self._s3_storage is not None:
            self._logger.info(f"syncing {run_dir.name} from S3 → {run_dir}")
            self._s3_storage.sync_to_local(run_dir, run_dir)
        return self._storage.load_result(run_dir)

    def _build_submit_args(self, cfg: ExperimentConfig, run_dir: Path) -> list[str]:
        """Compose the ``ovhai job run`` argument list for ``cfg``.

        Per-experiment S3 prefix isolation (gotcha #1 from project memory):
        each job mounts its own ``s3://<bucket_results>/<run_id>`` prefix so
        parallel jobs can't overwrite each other during the OVH FINALIZING
        sync. We intentionally drop the trailing slash to avoid the
        silent-prefix-loss bug documented in the rendezvous_comm port.
        """
        cfg_oc = self._ovh_config
        run_id = f"{cfg.experiment.id}_s{cfg.experiment.seed}"
        # No trailing slash; per-experiment isolation.
        results_uri = f"{cfg_oc.bucket_results}@{cfg_oc.region}/{run_id}:{cfg_oc.mount_results}"
        code_uri = f"{cfg_oc.bucket_code}@{cfg_oc.region}:{cfg_oc.mount_code}:RO"

        args = [
            "--name",
            f"multi-scenario-{run_id}",
            "--gpu",
            cfg_oc.gpu_type,
            "--gpus",
            str(cfg_oc.n_gpu),
            "--volume",
            code_uri,
            "--volume",
            results_uri,
            cfg_oc.image,
            "--",
            *cfg_oc.default_runner.split(),
            str(run_dir),
            *cfg_oc.default_extra_cli.split(),
        ]
        # Encrypted secret env vars (F6.1) — ship as one --env per key.
        if self._secret_env and self._secret_passphrase:
            ship = self._secrets.encrypt_for_env(self._secret_env, self._secret_passphrase)
            for k, v in ship.items():
                args = ["--env", f"{k}={v}", *args]
        return args

    def _poll_until_terminal(self, job_id: str):
        """Poll ``ovhai job get`` until terminal state or timeout."""
        deadline = time.time() + self._ovh_config.timeout_sec
        while True:
            info = self._client.get(job_id)
            if info.is_terminal:
                return info
            if time.time() >= deadline:
                raise OvhJobError(
                    f"OVH job {job_id} did not reach a terminal state within "
                    f"{self._ovh_config.timeout_sec}s; last state={info.state}"
                )
            self._sleep(self._ovh_config.poll_interval_sec)


def _safe_logs_tail(client: OvhClient, job_id: str, tail: int = 50) -> str:
    """Best-effort ``ovhai job logs`` for error reporting; returns '' on failure."""
    try:
        return client.logs(job_id, tail=tail)
    except OvhCliError:
        return ""
