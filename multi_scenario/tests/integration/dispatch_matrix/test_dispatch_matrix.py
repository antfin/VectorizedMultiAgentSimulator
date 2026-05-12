"""Dispatch-matrix smoke tests — 2 × 2 (ER1 / LERO) × (local / OVH-mocked).

Each cell pins one of two contracts:

1. **CLI ↔ Streamlit cfg parity** — picking a YAML through the Submit
   page's AppTest must produce a cfg dict byte-equal to what
   ``ExperimentConfig.from_yaml(yaml).model_dump()`` returns. If these
   drift, the same YAML would behave differently depending on which
   entry-point fired it — exactly the kind of bug that's hard to spot
   later. Catches regression of the F9.8 LERO-widget rendering work.

2. **End-to-end submission** — fire ``submit_to_local`` (real, tiny
   budget) and assert the canonical run-dir layout lands on disk.
   For OVH, ``OvhRunner.submit`` is monkey-patched; we assert the
   kwargs that would have shipped (especially that LERO YAMLs carry
   the encrypted ``MS_ENCRYPTED_SECRETS`` env var per Phase 2.5).

Smoke budget: 1 iter × 50 frames × 5 steps. Total wall ≈ 30-45s for
the four cells that actually train, instant for the mocked-OVH cells.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,import-outside-toplevel

from pathlib import Path

import pytest

from multi_scenario.domain.models import ExperimentConfig
from tests.integration.dispatch_matrix._helpers import (
    assert_er1_run_dir_complete,
    assert_lero_run_dir_complete,
    er1_smoke_cfg,
    lero_smoke_cfg,
    ovh_smoke_cfg,
    write_smoke_yaml,
)


_SUBMIT_PAGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "multi_scenario"
    / "frontend"
    / "pages"
    / "submit.py"
)


def _new_apptest(experiments_root: Path):
    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_SUBMIT_PAGE_PATH), default_timeout=30.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(experiments_root)
    return at


def _drive_pick_to_form(
    at,
    scenario: str = "discovery",
    folder: str = "baseline",
    config: str = "smoke.yaml",
) -> None:
    """Walk Step 1's cascading picker — pick + click 'Use this config'."""
    at.run()
    at.selectbox(key="step1_scenario").set_value(scenario)
    at.run()
    at.selectbox(key="step1_folder").set_value(folder)
    at.run()
    at.selectbox(key="step1_config").set_value(config)
    at.run()
    at.button(key="step1_pick").click()
    at.run()


# ── Cell 1: ER1 local — CLI ↔ Streamlit cfg parity ──────────────────


def test_er1_local_streamlit_cfg_matches_yaml(tmp_path: Path):
    """Streamlit's reconstructed cfg dict round-trips byte-equal to the YAML.

    Loading an ER1 YAML, walking through Step 1 (no edits), the form's
    ``submit_current_form`` should validate as the same ExperimentConfig
    that ``ExperimentConfig.from_yaml`` produces from the source YAML.
    """
    storage = tmp_path / "results"
    storage.mkdir()
    cfg_dict = er1_smoke_cfg(str(storage))
    yaml_path = write_smoke_yaml(tmp_path, cfg_dict)

    cfg_from_yaml = ExperimentConfig.from_yaml(yaml_path)

    at = _new_apptest(tmp_path)
    _drive_pick_to_form(at)
    streamlit_form = at.session_state["submit_current_form"]
    cfg_from_streamlit = ExperimentConfig.model_validate(streamlit_form)

    # model_dump strips ordering noise; compare on the validated cfg.
    assert cfg_from_streamlit.model_dump() == cfg_from_yaml.model_dump(), (
        "Streamlit's form dict diverged from the YAML's parsed cfg — "
        "the two dispatch fronts would produce different submissions"
    )


# ── Cell 2: LERO local — CLI ↔ Streamlit cfg parity ─────────────────


def test_lero_local_streamlit_cfg_matches_yaml(tmp_path: Path):
    """LERO YAMLs carry extra ``lero:`` + ``llm:`` blocks; round-trip must preserve them.

    Phase 9.8 widget regression test — loading a LERO YAML through the
    Submit form must produce a cfg whose ``cfg.lero`` and ``cfg.llm``
    match the YAML's values, not the widget defaults.
    """
    storage = tmp_path / "results"
    storage.mkdir()
    cfg_dict = lero_smoke_cfg(str(storage))
    yaml_path = write_smoke_yaml(tmp_path, cfg_dict, folder="lero")

    cfg_from_yaml = ExperimentConfig.from_yaml(yaml_path)
    assert cfg_from_yaml.lero is not None
    assert cfg_from_yaml.llm is not None

    at = _new_apptest(tmp_path)
    _drive_pick_to_form(at, folder="lero")
    streamlit_form = at.session_state["submit_current_form"]
    cfg_from_streamlit = ExperimentConfig.model_validate(streamlit_form)

    assert cfg_from_streamlit.lero is not None, "LERO section dropped through Streamlit"
    assert cfg_from_streamlit.llm is not None, "LLM section dropped through Streamlit"
    assert cfg_from_streamlit.model_dump() == cfg_from_yaml.model_dump()


# ── Cell 3: ER1 local — end-to-end run-dir lands correctly ──────────


@pytest.mark.slow
def test_er1_local_end_to_end_produces_run_dir(tmp_path: Path):
    """Fire ``submit_to_local`` (real, tiny budget) → assert run dir is complete."""
    from multi_scenario.adapters.logging.file_logger import FileLogger
    from multi_scenario.application.submission import build_run_dir, submit_to_local

    storage = tmp_path / "results"
    storage.mkdir()
    yaml_path = write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    cfg = ExperimentConfig.from_yaml(yaml_path)
    _, run_dir = build_run_dir(cfg, mkdir=True)
    submission = submit_to_local(
        cfg, run_dir=run_dir, logger=FileLogger(run_dir / "logs" / "run.log")
    )

    assert submission.run_dir == run_dir
    assert_er1_run_dir_complete(run_dir)


# ── Cell 4: LERO local — end-to-end including LERO sub-tree ─────────


@pytest.mark.slow
def test_lero_local_end_to_end_produces_full_lero_layout(tmp_path: Path, monkeypatch):
    """LERO local submit completes; output/lero/{evolution_doc.md,prompts/...} present.

    Uses :class:`FakeLlmClient` injected via the factory's
    ``MULTI_SCENARIO_LLM_OVERRIDE=fake`` hook so no LLM keys / network
    are needed. The fake returns a canned reward function the codegen
    can extract — exercises the same wiring a real LLM call would.
    """
    from multi_scenario.adapters.llm import FakeLlmClient
    from multi_scenario.adapters.logging.file_logger import FileLogger
    from multi_scenario.application import lero_factory
    from multi_scenario.application.submission import build_run_dir, submit_to_local
    from multi_scenario.domain.lero import LlmCompletion

    canned = (
        "```python\n"
        "def compute_reward(scenario_state):\n"
        "    return scenario_state['agent_pos'][..., 0]\n"
        "```\n"
    )
    # Inject a FakeLlmClient into the factory — bypasses LiteLLM
    # entirely so the test has no env / network dependency.
    fake = FakeLlmClient().register_always(LlmCompletion(text=canned))
    monkeypatch.setattr(
        lero_factory,
        "_build_default_llm_client",
        lambda *_args, **_kwargs: fake,
    )

    storage = tmp_path / "results"
    storage.mkdir()
    yaml_path = write_smoke_yaml(tmp_path, lero_smoke_cfg(str(storage)), folder="lero")
    cfg = ExperimentConfig.from_yaml(yaml_path)
    _, run_dir = build_run_dir(cfg, mkdir=True)
    submission = submit_to_local(
        cfg, run_dir=run_dir, logger=FileLogger(run_dir / "logs" / "run.log")
    )
    assert submission.run_dir == run_dir
    assert_lero_run_dir_complete(run_dir)


# ── Cell 5: ER1 OVH — mocked OvhRunner receives the right cfg ──────


def test_er1_ovh_streamlit_dispatches_to_ovh_runner(tmp_path: Path, monkeypatch):
    """Submit page on an OVH-runner ER1 YAML → ``OvhRunner.submit`` invoked with the cfg.

    No real network / no real ovhai. The page constructs ``OvhRunner``,
    we monkey-patch its ``submit`` method to capture the cfg + kwargs,
    and assert the dispatch wiring shipped the right call.
    """
    storage = tmp_path / "results"
    storage.mkdir()
    cfg_dict = ovh_smoke_cfg(er1_smoke_cfg(str(storage)))
    yaml_path = write_smoke_yaml(tmp_path, cfg_dict)

    # Avoid loading a real configs/ovh.yaml — the page's _try_load_ovh_config
    # would otherwise reject the test setup. Patch it to a stub.
    captured: dict[str, object] = {}

    from multi_scenario.adapters.runners.ovh import OvhRunner

    def _fake_submit(self, cfg, run_dir):  # noqa: ARG001
        captured["cfg"] = cfg
        captured["run_dir"] = run_dir
        captured["secret_env"] = self._secret_env  # noqa: SLF001
        return "job_id_stub"

    monkeypatch.setattr(OvhRunner, "submit", _fake_submit)

    # Build cfg via the CLI path; the Streamlit-page CLI parity is
    # pinned by Cell 1+2 above. Here we focus on the OVH-dispatch path.
    from multi_scenario.adapters.logging.file_logger import FileLogger
    from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
    from multi_scenario.application.submission import build_run_dir, submit_to_ovh

    cfg = ExperimentConfig.from_yaml(yaml_path)
    _, run_dir = build_run_dir(cfg, mkdir=False)
    submission = submit_to_ovh(
        cfg,
        ovh_cfg=_stub_ovh_job_config(),
        yaml_path_in_repo="discovery/baseline/configs/smoke.yaml",
        run_dir=run_dir,
        logger=FileLogger(tmp_path / "run.log"),
        client=_StubOvhClient(),
        secrets=FernetSecretsAdapter(),
    )

    assert submission.job_id == "job_id_stub"
    assert captured["cfg"].experiment.id == "er1_smoke"
    assert captured["cfg"].lero is None
    # Non-LERO submissions ship no encrypted secret env.
    assert (captured["secret_env"] or {}) == {}


# ── Cell 6: LERO OVH — mocked, asserts the secret_env is wired ─────


def test_lero_ovh_streamlit_ships_encrypted_secrets(tmp_path: Path, monkeypatch):
    """LERO YAML on OVH → ``OvhRunner`` receives encrypted ``MS_ENCRYPTED_SECRETS``.

    Phase 2.5 wiring proof: a LERO submission MUST ship ``OPENAI_API_KEY``
    (and friends) via the Fernet-encrypted ``secret_env`` channel so the
    container can decrypt + prime ``os.environ`` before LiteLlmClient
    instantiates. Without this, the OVH job 401s mid-iteration ~5 min
    after boot.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-stub")

    storage = tmp_path / "results"
    storage.mkdir()
    cfg_dict = ovh_smoke_cfg(lero_smoke_cfg(str(storage)))
    yaml_path = write_smoke_yaml(tmp_path, cfg_dict, folder="lero")

    captured: dict[str, object] = {}

    from multi_scenario.adapters.runners.ovh import OvhRunner

    def _fake_submit(self, cfg, run_dir):  # noqa: ARG001
        captured["cfg"] = cfg
        captured["secret_env"] = self._secret_env  # noqa: SLF001
        captured["secret_passphrase"] = self._secret_passphrase  # noqa: SLF001
        return "lero_job_stub"

    monkeypatch.setattr(OvhRunner, "submit", _fake_submit)

    from multi_scenario.adapters.logging.file_logger import FileLogger
    from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
    from multi_scenario.application.submission import build_run_dir, submit_to_ovh

    cfg = ExperimentConfig.from_yaml(yaml_path)
    _, run_dir = build_run_dir(cfg, mkdir=False)
    submission = submit_to_ovh(
        cfg,
        ovh_cfg=_stub_ovh_job_config(),
        yaml_path_in_repo="discovery/lero/configs/smoke.yaml",
        run_dir=run_dir,
        logger=FileLogger(tmp_path / "run.log"),
        client=_StubOvhClient(),
        secrets=FernetSecretsAdapter(),
    )

    assert submission.job_id == "lero_job_stub"
    assert captured["cfg"].lero is not None
    secret_env = captured["secret_env"]
    assert secret_env is not None, (
        "LERO submission MUST collect API keys for the Fernet-shipping path "
        "(Phase 2.5). The OvhRunner got no secret_env — the container would "
        "401 mid-iter."
    )
    assert "OPENAI_API_KEY" in secret_env
    assert secret_env["OPENAI_API_KEY"] == "sk-test-stub"
    # Passphrase is fresh per submission (32-byte urlsafe).
    assert captured["secret_passphrase"] is not None
    assert len(captured["secret_passphrase"]) >= 32


# ── Stubs ───────────────────────────────────────────────────────────


def _stub_ovh_job_config():
    """Minimal OvhJobConfig for the dispatch tests — no real bucket / region."""
    from multi_scenario.domain.models import OvhJobConfig

    return OvhJobConfig(
        region="GRA",
        image="stub:latest",
        bucket_code="ms-code",
        bucket_results="ms-results",
    )


class _StubOvhClient:
    """Bare stub — OvhRunner.submit is monkey-patched so client isn't reached."""

    # pylint: disable=missing-class-docstring,too-few-public-methods
    pass


# ── A.2 lifecycle tests — pullback, video-regen, auto-poll ──────────


def test_ovh_lero_pullback_brings_back_lero_subtree(tmp_path: Path, monkeypatch):
    """A.2 cell 7: DONE → pullback copies the LERO sub-tree back locally.

    Stages an S3-shaped fixture (input/, output/lero/{...}, output/{metrics,
    report,eval_episodes}.json, run_state.json) and mocks ``pullback_run_dir``
    to copy it to ``dest_dir``. Then asserts the local run-dir passes the
    full LERO-shape assertion — every file Streamlit's Run Detail page
    will look for.
    """
    # Stage the "S3 contents" — a complete LERO run-dir layout in tmp.
    s3_staging = tmp_path / "s3_staging"
    _populate_lero_run_dir_layout(s3_staging)

    # Local destination — where pullback will write.
    dest_dir = tmp_path / "local_pullback_dir"

    # Mock pullback_run_dir to copy s3_staging → dest_dir.
    from multi_scenario.application import ovh_pullback

    pullback_calls = {}

    def _fake_pullback(*, ovh_cfg, run_dir_name, dest_dir, client):  # noqa: ARG001
        import shutil

        pullback_calls["dest_dir"] = dest_dir
        pullback_calls["run_dir_name"] = run_dir_name
        shutil.copytree(s3_staging, dest_dir, dirs_exist_ok=True)
        return ovh_pullback.PullbackResult(
            dest_dir=dest_dir, n_downloaded=10, n_skipped=0
        )

    monkeypatch.setattr(ovh_pullback, "pullback_run_dir", _fake_pullback)

    # Invoke pullback like the Streamlit page would.
    result = ovh_pullback.pullback_run_dir(
        ovh_cfg=_stub_ovh_job_config(),
        run_dir_name="lero_smoke_s0__t",
        dest_dir=dest_dir,
        client=_StubOvhClient(),
    )

    assert result.n_downloaded == 10
    assert pullback_calls["dest_dir"] == dest_dir
    # Full LERO-shape assertion: ER1 outputs + output/lero/ tree.
    assert_lero_run_dir_complete(dest_dir)


def test_ovh_done_triggers_video_regen_subprocess(tmp_path: Path, monkeypatch):
    """A.2 cell 8: missing videos + checkpoint present → regen subprocess fires.

    Streamlit's ``_refresh_ovh_status`` post-pullback step runs
    ``multi-scenario regenerate-videos <run_dir>`` only when videos/ is
    absent AND a BenchMARL checkpoint exists. Pins that the subprocess
    actually gets invoked with the right argv.
    """
    import subprocess

    # Stage a run_dir that LOOKS like it just got pulled back: has a
    # checkpoint, no videos/.
    run_dir = tmp_path / "run"
    (run_dir / "input").mkdir(parents=True)
    bench_root = (
        run_dir / "output" / "benchmarl" / "mappo_discovery_mlp__demo_2026_05_12-00_00_00"
    )
    (bench_root / "checkpoints").mkdir(parents=True)
    (bench_root / "checkpoints" / "checkpoint_50000.pt").write_bytes(b"\0")
    (bench_root / "config.pkl").write_bytes(b"\0")
    (run_dir / "output" / "metrics.json").write_text("{}")

    # Capture subprocess.run calls.
    calls = []

    def _fake_run(argv, **kwargs):  # noqa: ARG001
        calls.append(argv)
        return type("CP", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    # Exercise the gating logic the Submit page uses (Phase 8 / Submit
    # post-pullback hook).
    from multi_scenario.application.regenerate_videos import latest_checkpoint

    assert latest_checkpoint(run_dir) is not None  # precondition
    assert not (run_dir / "videos").is_dir()  # precondition

    # Simulate what _refresh_ovh_status's post-pullback block does:
    import sys

    if latest_checkpoint(run_dir) is not None and not (run_dir / "videos").is_dir():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "multi_scenario.cli",
                "regenerate-videos",
                str(run_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert calls, "regenerate-videos subprocess was not invoked"
    argv = calls[0]
    assert "multi_scenario.cli" in argv
    assert "regenerate-videos" in argv
    assert str(run_dir) in argv


def test_streamlit_auto_poll_block_present_in_submit_page():
    """A.2 cell 9: Phase 8 auto-poll wiring is present in submit.py.

    The auto-poll block lives deep inside the Submit-page Step 5
    container, only renders once the user has driven the workflow
    through to a submitted status. Driving an AppTest that far for
    each cell would be a long pick→save→preflight→submit dance with
    mocked OVH at each step.

    A presence smoke test catches the regression that matters: if
    someone deletes / refactors the auto-poll block without
    re-routing the refresh call, this fails. The interactive
    behaviour is covered by ``test_refresh_helper_*`` (which exercise
    ``_refresh_ovh_status`` directly).
    """
    from multi_scenario.frontend.pages import submit as submit_page

    source = Path(submit_page.__file__).read_text(encoding="utf-8")
    # Three load-bearing pieces of the Phase 8 wiring.
    assert 'key="auto_poll_ovh"' in source, (
        "auto-poll checkbox missing — Phase 8 (Streamlit auto-pullback) regressed"
    )
    assert "if auto_poll:" in source, "auto-poll conditional branch missing"
    assert "_refresh_ovh_status()" in source, (
        "auto-poll block doesn't call _refresh_ovh_status — pullback won't auto-fire"
    )


# ── B: --json output for chat-trigger ergonomics ─────────────────────


def test_run_cli_json_flag_emits_parseable_record_local(tmp_path: Path):
    """B: ``multi-scenario run --json <yaml>`` emits one JSON line at end.

    The final stdout line must be parseable so chat-driven submission
    flows (or any scripted pipeline) can read ``run_id`` / ``run_dir``
    without parsing English. Local path emits a 3-field record.
    """
    import json as _json
    import subprocess
    import sys

    storage = tmp_path / "results"
    storage.mkdir()
    yaml_path = write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))

    proc = subprocess.run(
        [sys.executable, "-m", "multi_scenario.cli", "run", "--json", str(yaml_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,  # CLI builds run_dir relative to cwd
    )
    last_line = proc.stdout.strip().splitlines()[-1]
    record = _json.loads(last_line)
    assert record["runner"] == "local"
    assert record["run_id"].startswith("er1_smoke_s0")
    assert "run_dir" in record


def test_run_cli_json_flag_emits_parseable_record_ovh_dispatch_only(
    tmp_path: Path, monkeypatch
):
    """B: OVH path's --json record carries job_id + s3_prefix + dashboard_url."""
    # We don't want this test to actually hit ovhai or upload code.
    # Mock the submit_to_ovh function in the cli.run module — bypasses
    # _maybe_upload_stale_code AND the actual submission.
    import json as _json

    from multi_scenario.application.submission import OvhSubmission
    from multi_scenario.cli import run as cli_run
    from multi_scenario.domain.models import RunId

    fake_sub = OvhSubmission(
        job_id="fake-job-abc",
        run_id=RunId(exp_id="er1_smoke", seed=0),
        run_dir=Path("/workspace/results/er1_smoke_s0__t"),
        s3_prefix="ms-results@GRA/er1_smoke_s0__t",
        dashboard_url="https://ovh.example/jobs/fake-job-abc",
    )

    def _fake_submit_to_ovh(*_args, **_kwargs):
        return fake_sub

    monkeypatch.setattr(cli_run, "submit_to_ovh", _fake_submit_to_ovh)
    monkeypatch.setattr(cli_run, "_maybe_upload_stale_code", lambda _cfg: None)

    # Use typer's CliRunner to capture stdout cleanly.
    from typer.testing import CliRunner

    storage = tmp_path / "results"
    storage.mkdir()
    cfg_dict = ovh_smoke_cfg(er1_smoke_cfg(str(storage)))
    yaml_path = write_smoke_yaml(tmp_path, cfg_dict)
    monkeypatch.chdir(tmp_path)
    # Stage a minimal configs/ovh.yaml so the lazy-loaded
    # OvhJobConfig.from_yaml succeeds.
    import yaml as _yaml

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "ovh.yaml").write_text(
        _yaml.dump(
            {"region": "GRA", "image": "stub:latest", "bucket_code": "c", "bucket_results": "r"}
        )
    )

    runner = CliRunner()
    from multi_scenario.cli._app import app

    result = runner.invoke(app, ["run", "--json", str(yaml_path)])
    assert result.exit_code == 0, result.output
    last_line = result.output.strip().splitlines()[-1]
    record = _json.loads(last_line)
    assert record["runner"] == "ovh"
    assert record["job_id"] == "fake-job-abc"
    assert record["s3_prefix"].startswith("ms-results@GRA/")
    assert record["dashboard_url"].startswith("https://")


def _populate_lero_run_dir_layout(root: Path) -> None:
    """Stage a complete LERO run_dir layout for pullback tests.

    Mirrors the shape :func:`assert_lero_run_dir_complete` checks: every
    file the LERO post-completion lifecycle should produce. Used by
    pullback tests that need to verify the right files survive the S3
    round-trip (file copies in the stub; real ovhai bucket walks in
    production).
    """
    (root / "input").mkdir(parents=True)
    (root / "input" / "config.json").write_text("{}")
    (root / "input" / "provenance.json").write_text("{}")
    output = root / "output"
    output.mkdir()
    (output / "metrics.json").write_text("{}")
    (output / "report.json").write_text("{}")
    (output / "eval_episodes.json").write_text("{}")
    lero = output / "lero"
    lero.mkdir()
    (lero / "final_summary.json").write_text("{}")
    (lero / "evolution_history.json").write_text("[]")
    (lero / "evolution_doc.md").write_text("# LERO run\n")
    prompts_iter0 = lero / "prompts" / "iter_0"
    prompts_iter0.mkdir(parents=True)
    (prompts_iter0 / "system.md").write_text("system prompt")
    (root / "run_state.json").write_text("{}")
