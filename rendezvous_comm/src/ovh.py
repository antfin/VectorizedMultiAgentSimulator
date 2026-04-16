"""OVH Public Cloud CLI wrapper for AI Training jobs.

Wraps `ovhai` CLI commands via subprocess. Requires `ovhai` to be installed
and authenticated (`ovhai login`).

Configuration is loaded from configs/ovh.yaml (edit that file to change
defaults for buckets, region, GPU model, etc.). Sensitive data (tokens,
passwords) should NEVER go in the config — use `ovhai login`.
"""
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_log = logging.getLogger("rendezvous.ovh")

# ── Load OVH config from YAML ───────────────────────────────────

_OVH_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "ovh.yaml"


def load_ovh_config() -> Dict[str, Any]:
    """Load OVH configuration from configs/ovh.yaml.

    Returns the parsed dict, or sensible defaults if file is missing.
    """
    if _OVH_CONFIG_PATH.exists():
        with open(_OVH_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    _log.warning(f"OVH config not found: {_OVH_CONFIG_PATH}")
    return {}


def _cfg() -> Dict[str, Any]:
    """Cached config access."""
    if not hasattr(_cfg, "_cache"):
        _cfg._cache = load_ovh_config()
    return _cfg._cache


def reload_ovh_config():
    """Force reload of OVH config (useful after editing ovh.yaml)."""
    if hasattr(_cfg, "_cache"):
        del _cfg._cache


def _storage_cfg() -> Dict[str, str]:
    return _cfg().get("storage", {})


def _training_cfg() -> Dict[str, Any]:
    return _cfg().get("training", {})


def _mounts_cfg() -> Dict[str, str]:
    return _cfg().get("mounts", {})


# ── Derived defaults from config ─────────────────────────────────

def _gpu_models_from_config() -> Dict[str, Dict]:
    """Build GPU_MODELS dict from config, with hardcoded fallback."""
    fallback = {
        "V100S": {"vram_gb": 32, "eur_per_hr": 2.10, "flavor": "ai1-1-gpu"},
    }
    return _cfg().get("gpu_models", fallback)


GPU_MODELS = _gpu_models_from_config()


def default_region() -> str:
    return _storage_cfg().get("region", "GRA")


def default_bucket_code() -> str:
    return _storage_cfg().get("bucket_code", "rendezvous-code")


def default_bucket_results() -> str:
    return _storage_cfg().get("bucket_results", "rendezvous-results")


def default_gpu() -> str:
    return _training_cfg().get("default_gpu", "V100S")


def default_n_gpu() -> int:
    return _training_cfg().get("n_gpu", 1)


def default_image() -> str:
    return _training_cfg().get(
        "image",
        "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    )


def default_mount_code() -> str:
    return _mounts_cfg().get("code", "/workspace/code")


def default_mount_results() -> str:
    return _mounts_cfg().get("results", "/workspace/results")


@dataclass
class JobInfo:
    """Parsed OVH AI Training job."""

    id: str
    name: str
    status: str
    created_at: str = ""
    gpu_type: str = ""
    duration_seconds: int = 0
    url: str = ""


def check_cli_available() -> bool:
    """Check if ovhai CLI is installed and authenticated."""
    try:
        r = subprocess.run(
            ["ovhai", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


def _run_ovhai(args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run an ovhai command, returning the CompletedProcess."""
    cmd = ["ovhai"] + args
    _log.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )


def submit_training_job(
    config_yaml: str,
    gpu_type: Optional[str] = None,
    bucket_code: Optional[str] = None,
    bucket_results: Optional[str] = None,
    region: Optional[str] = None,
    image: Optional[str] = None,
    n_gpu: Optional[int] = None,
    job_name: Optional[str] = None,
    llm_env: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Submit an OVH AI Training job.

    All parameters default to values from configs/ovh.yaml.

    For LERO experiments that need LLM access, pass API keys via llm_env:
        llm_env={"OPENAI_API_KEY": "sk-..."}
    Keys are passed as --env flags to the container. They are NOT stored
    in git, the Docker image, or the code bucket. They are visible in
    OVH job metadata (ovhai job get) to project members only.

    Tip: read keys from local .env at submit time:
        from dotenv import dotenv_values
        llm_env = dotenv_values("rendezvous_comm/.env")
        submit_training_job(..., llm_env=llm_env)

    Args:
        config_yaml: path to config YAML relative to code root
            (e.g., "rendezvous_comm/configs/er1/demo.yaml")
        gpu_type: GPU model (V100S, CPU)
        bucket_code: S3 bucket with uploaded code
        bucket_results: S3 bucket for results
        region: OVH region (GRA, BHS, etc.)
        image: Docker image
        n_gpu: number of GPUs
        job_name: optional job name

    Returns:
        Job ID string, or None on failure.
    """
    # Apply defaults from config
    gpu_type = gpu_type or default_gpu()
    bucket_code = bucket_code or default_bucket_code()
    bucket_results = bucket_results or default_bucket_results()
    region = region or default_region()
    image = image or default_image()
    n_gpu = n_gpu if n_gpu is not None else default_n_gpu()
    mount_code = default_mount_code()
    mount_results = default_mount_results()

    if gpu_type not in GPU_MODELS:
        _log.error(
            f"Unknown GPU type: {gpu_type}. "
            f"Available: {list(GPU_MODELS.keys())}"
        )
        return None

    if job_name is None:
        # Derive from config path
        parts = Path(config_yaml).stem
        job_name = f"rendezvous_{parts}"

    # Extract exp_id from config to use as bucket prefix.
    # Each job gets its own prefix in the results bucket so that
    # parallel jobs don't overwrite each other during FINALIZING.
    # OVH syncs the entire local volume back to the bucket prefix
    # when the job ends — without isolation, the last job to
    # finalize wins and earlier results are lost.
    try:
        with open(config_yaml) as _f:
            _raw = yaml.safe_load(_f)
        exp_id = _raw.get("exp_id", "")
    except Exception:
        exp_id = ""
    # OVH volume syntax: `container@alias[/prefix]:mount:perm` —
    # the prefix is optional and must NOT have a trailing slash.
    # Without this, OVH parses the trailing slash as part of the
    # prefix string, fails silently, and falls back to bucket root,
    # causing parallel jobs to overwrite each other's results.
    results_volume = (
        f"{bucket_results}@{region}/{exp_id}:{mount_results}:rwd"
        if exp_id
        else f"{bucket_results}@{region}:{mount_results}:rwd"
    )
    code_volume = f"{bucket_code}@{region}:{mount_code}:ro"

    train_cmd = (
        f"export HOME=/tmp && "
        f"pip install "
        f"vmas benchmarl tensordict torchrl "
        f"pyyaml pandas scipy imageio matplotlib litellm && "
        f"cd {mount_code}/rendezvous_comm && "
        f"python train.py "
        f"{mount_code}/{config_yaml} "
        f"--device cuda"
    )

    # Map GPU model name to OVH flavor ID
    flavor = GPU_MODELS.get(gpu_type, {}).get("flavor", gpu_type)

    args = [
        "job", "run", image,
        "--name", job_name,
        "--flavor", flavor,
        "--gpu", str(n_gpu),
        "--volume", code_volume,
        "--volume", results_volume,
        "--env", f"RESULTS_DIR={mount_results}",
        "--env", f"CHECKPOINTS_DIR={mount_results}/checkpoints",
    ]
    # LLM API keys for LERO experiments — encrypted before submission
    # so they appear as opaque blobs in `ovhai job get`.
    if llm_env:
        from .secrets_util import encrypt_env
        import secrets
        passphrase = secrets.token_urlsafe(24)
        encrypted = encrypt_env(llm_env, passphrase)
        for key, value in encrypted.items():
            args.extend(["--env", f"{key}={value}"])
        _log.info(
            "LLM keys encrypted (%d keys). "
            "Passphrase is ephemeral — lost after this session.",
            len(llm_env),
        )
    args.extend([
        "--output", "json",
        "--", "bash", "-c", train_cmd,
    ])

    r = _run_ovhai(args, timeout=30)
    if r.returncode != 0:
        _log.error(f"Job submission failed: {r.stderr}")
        return None

    try:
        data = json.loads(r.stdout)
        job_id = data.get("id", "")
        _log.info(f"Job submitted: {job_id} ({job_name})")
        return job_id
    except json.JSONDecodeError:
        # Try to extract ID from text output
        _log.warning(f"Could not parse JSON response: {r.stdout}")
        return r.stdout.strip()


def list_jobs(status_filter: Optional[str] = None) -> List[JobInfo]:
    """List OVH AI Training jobs.

    Args:
        status_filter: optional status filter (RUNNING, DONE, ERROR, etc.)
    """
    args = ["job", "list", "--output", "json"]
    r = _run_ovhai(args, timeout=30)
    if r.returncode != 0:
        _log.error(f"Failed to list jobs: {r.stderr}")
        return []

    try:
        jobs_data = json.loads(r.stdout)
    except json.JSONDecodeError:
        _log.error(f"Invalid JSON from job list: {r.stdout[:200]}")
        return []

    jobs = []
    items = jobs_data if isinstance(jobs_data, list) else jobs_data.get("items", [])
    for j in items:
        status = j.get("status", {})
        status_str = status.get("state", "") if isinstance(status, dict) else str(status)
        info = JobInfo(
            id=j.get("id", ""),
            name=j.get("name", j.get("id", "")[:12]),
            status=status_str,
            created_at=j.get("createdAt", ""),
            gpu_type=j.get("resources", {}).get("gpuModel", ""),
            duration_seconds=status.get("duration", 0) if isinstance(status, dict) else 0,
        )
        if status_filter is None or info.status == status_filter:
            jobs.append(info)

    return jobs


def get_job(job_id: str) -> Optional[JobInfo]:
    """Get details of a specific job."""
    r = _run_ovhai(["job", "get", job_id, "--output", "json"], timeout=15)
    if r.returncode != 0:
        return None
    try:
        j = json.loads(r.stdout)
        status = j.get("status", {})
        status_str = status.get("state", "") if isinstance(status, dict) else str(status)
        return JobInfo(
            id=j.get("id", ""),
            name=j.get("name", ""),
            status=status_str,
            created_at=j.get("createdAt", ""),
            gpu_type=j.get("resources", {}).get("gpuModel", ""),
            duration_seconds=status.get("duration", 0) if isinstance(status, dict) else 0,
            url=j.get("status", {}).get("url", ""),
        )
    except json.JSONDecodeError:
        return None


def get_job_logs(job_id: str, tail: int = 100) -> str:
    """Get recent log output from a job."""
    r = _run_ovhai(
        ["job", "logs", job_id, f"--tail={tail}"], timeout=15,
    )
    return r.stdout if r.returncode == 0 else f"Error: {r.stderr}"


def stop_job(job_id: str) -> bool:
    """Stop a running job."""
    r = _run_ovhai(["job", "stop", job_id], timeout=15)
    return r.returncode == 0


def list_buckets() -> List[str]:
    """List available S3 buckets."""
    r = _run_ovhai(["bucket", "list", "--output", "json"], timeout=15)
    if r.returncode != 0:
        return []
    try:
        data = json.loads(r.stdout)
        items = data if isinstance(data, list) else data.get("items", [])
        return [b.get("name", "") for b in items if b.get("name")]
    except json.JSONDecodeError:
        return []


def upload_code(local_dir: str, bucket: Optional[str] = None, region: Optional[str] = None) -> bool:
    """Upload code directory to an OVH bucket.

    Strips the local absolute path so files appear at the bucket root
    (e.g., rendezvous_comm/src/...) instead of /Users/.../rendezvous_comm/...

    Excludes results/, __pycache__/, and other non-code directories
    by uploading only the relevant subdirectories.
    """
    bucket = bucket or default_bucket_code()
    region = region or default_region()
    local = Path(local_dir)
    parent = str(local.parent) + "/"

    # Upload only code-relevant subdirectories and top-level files
    _CODE_DIRS = ["src", "configs", "notebooks", "tests"]
    _CODE_FILES = ["train.py", "setup.py", "setup.cfg",
                   "pyproject.toml", "requirements.txt"]

    success = True
    # Upload subdirectories
    for subdir in _CODE_DIRS:
        p = local / subdir
        if not p.exists():
            continue
        r = _run_ovhai(
            [
                "bucket", "object", "upload",
                f"{bucket}@{region}",
                "--remove-prefix", parent,
                str(p),
            ],
            timeout=300,
        )
        if r.returncode != 0:
            _log.error(f"Upload {subdir} failed: {r.stderr}")
            success = False

    # Upload top-level files
    for fname in _CODE_FILES:
        p = local / fname
        if not p.exists():
            continue
        r = _run_ovhai(
            [
                "bucket", "object", "upload",
                f"{bucket}@{region}",
                "--remove-prefix", parent,
                str(p),
            ],
            timeout=60,
        )
        if r.returncode != 0:
            _log.error(f"Upload {fname} failed: {r.stderr}")
            success = False

    if success:
        _log.info(f"Uploaded {local_dir} to {bucket}@{region}")
    return success


def download_results(
    bucket: Optional[str] = None,
    local_dir: str = "",
    prefix: str = "",
    region: Optional[str] = None,
) -> bool:
    """Download results from an OVH bucket.

    Files are downloaded relative to CWD. We use --output to prepend
    the local_dir path so files land in the right place.
    """
    bucket = bucket or default_bucket_results()
    region = region or default_region()

    # Ensure local_dir ends with / for --output prefix
    out_prefix = str(local_dir).rstrip("/") + "/"

    args = [
        "bucket", "object", "download",
        f"{bucket}@{region}",
        "--output", out_prefix,
        "--no-overwrite",
        "--workers", "8",
    ]
    if prefix:
        args.extend(["--prefix", prefix])

    r = _run_ovhai(args, timeout=600)
    if r.returncode == 0:
        _log.info(f"Downloaded {bucket}/{prefix} to {local_dir}")
    else:
        _log.error(f"Download failed: {r.stderr}")
    return r.returncode == 0


def estimate_cost(
    gpu_type: str,
    n_runs: int,
    est_minutes_per_run: float,
    storage_gb: float = 1.0,
) -> Dict[str, float]:
    """Estimate OVH cost for a sweep.

    Returns dict with gpu_cost_eur, storage_cost_eur, total_eur.
    """
    fallback = next(iter(GPU_MODELS.values()))
    gpu = GPU_MODELS.get(gpu_type, fallback)
    gpu_hours = n_runs * est_minutes_per_run / 60.0
    gpu_cost = gpu_hours * gpu["eur_per_hr"]
    storage_cost = storage_gb * 0.007  # Standard S3 per month
    return {
        "gpu_type": gpu_type,
        "gpu_hours": round(gpu_hours, 2),
        "gpu_cost_eur": round(gpu_cost, 2),
        "storage_cost_eur": round(storage_cost, 4),
        "total_eur": round(gpu_cost + storage_cost, 2),
    }
