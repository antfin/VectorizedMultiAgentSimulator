"""OVH Public Cloud CLI wrapper for AI Training jobs.

Wraps `ovhai` CLI commands via subprocess. Requires `ovhai` to be installed
and authenticated (`ovhai login`).

GPU pricing (EUR/hr, OVH Public Cloud / startup program):
    L4 (24GB):   ~0.75
    L40S (48GB): ~1.40
    H100 (80GB): ~3.10
"""
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger("rendezvous.ovh")

# GPU models available in OVH Public Cloud (startup program)
GPU_MODELS = {
    "L4": {"vram_gb": 24, "eur_per_hr": 0.75},
    "L40S": {"vram_gb": 48, "eur_per_hr": 1.40},
    "H100": {"vram_gb": 80, "eur_per_hr": 3.10},
}

DEFAULT_REGION = "GRA"
DEFAULT_IMAGE = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"


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
    gpu_type: str = "L4",
    bucket_code: str = "rendezvous-code",
    bucket_results: str = "rendezvous-results",
    region: str = DEFAULT_REGION,
    image: str = DEFAULT_IMAGE,
    n_gpu: int = 1,
    job_name: Optional[str] = None,
) -> Optional[str]:
    """Submit an OVH AI Training job.

    Args:
        config_yaml: path to config YAML relative to code root
            (e.g., "rendezvous_comm/configs/er1/demo.yaml")
        gpu_type: GPU model (L4, L40S, H100)
        bucket_code: S3 bucket with uploaded code
        bucket_results: S3 bucket for results
        region: OVH region (GRA, BHS, etc.)
        image: Docker image
        n_gpu: number of GPUs
        job_name: optional job name

    Returns:
        Job ID string, or None on failure.
    """
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

    train_cmd = (
        f"pip install -e /workspace/code/rendezvous_comm 2>/dev/null; "
        f"cd /workspace/code && "
        f"python -m rendezvous_comm.train "
        f"/workspace/code/{config_yaml} "
        f"--device cuda"
    )

    args = [
        "job", "run", image,
        "--name", job_name,
        "--gpu", str(n_gpu),
        "--gpu-model", gpu_type,
        "--volume", f"{bucket_code}@{region}/:/workspace/code:ro",
        "--volume", f"{bucket_results}@{region}/:/workspace/results:rwd",
        "--env", "RESULTS_DIR=/workspace/results",
        "--output", "json",
        "--", "bash", "-c", train_cmd,
    ]

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
        ["job", "logs", job_id, "--tail", str(tail)], timeout=15,
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


def upload_code(local_dir: str, bucket: str, region: str = DEFAULT_REGION) -> bool:
    """Upload code directory to an OVH bucket."""
    r = _run_ovhai(
        ["bucket", "object", "upload", f"{bucket}@{region}", local_dir],
        timeout=300,
    )
    if r.returncode == 0:
        _log.info(f"Uploaded {local_dir} to {bucket}@{region}")
    else:
        _log.error(f"Upload failed: {r.stderr}")
    return r.returncode == 0


def download_results(
    bucket: str,
    local_dir: str,
    prefix: str = "",
    region: str = DEFAULT_REGION,
) -> bool:
    """Download results from an OVH bucket."""
    args = [
        "bucket", "object", "download",
        f"{bucket}@{region}",
        "--output-dir", local_dir,
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
    gpu = GPU_MODELS.get(gpu_type, GPU_MODELS["L4"])
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
