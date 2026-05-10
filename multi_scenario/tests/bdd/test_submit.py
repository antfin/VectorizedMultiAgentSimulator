"""F7.7.C2 — pytest-bdd glue for ``submit.feature``.

Step definitions delegate to Streamlit's AppTest harness (same as the F7.7.C1
end-to-end tests) so the .feature file remains the single source of truth
for the user journey while implementation details stay in one place.
"""

# pylint: disable=missing-function-docstring,unused-argument,import-outside-toplevel

from pathlib import Path

from pytest_bdd import given, parsers, scenarios, then, when


scenarios("features/submit.feature")


# ── Background ──────────────────────────────────────────────────────


@given("an experiments directory with a valid discovery baseline config")
def _experiments_directory(experiments_root: Path, context: dict) -> None:
    context["experiments_root"] = experiments_root


# ── Givens ──────────────────────────────────────────────────────────


@given("the local submit target is selected")
def _local_target(context: dict) -> None:
    context["target"] = "local"


@given("configs/ovh.yaml is absent")
def _no_ovh_yaml(monkeypatch, tmp_path, context):
    """Run the page with cwd in a fresh tmpdir → ``configs/ovh.yaml`` is missing."""
    monkeypatch.chdir(tmp_path)
    context["target"] = "ovh"


# ── When: pick → drive Step 1 ───────────────────────────────────────


@when(parsers.parse("I pick {scenario} / {folder} / {config}"))
def _pick(
    submit_page_path: Path, context: dict, scenario: str, folder: str, config: str
) -> None:
    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(submit_page_path), default_timeout=15.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(context["experiments_root"])
    at.run()
    at.selectbox(key="step1_scenario").set_value(scenario)
    at.run()
    at.selectbox(key="step1_folder").set_value(folder)
    at.run()
    at.selectbox(key="step1_config").set_value(config)
    at.run()
    at.button(key="step1_pick").click()
    at.run()
    context["at"] = at


@when("I switch the submit target to ovh")
def _switch_to_ovh(context: dict) -> None:
    at = context["at"]
    at.radio(key="step2_submit_target").set_value("ovh")
    at.run()
    # F7.7.A4: switching the submit target now legitimately writes
    # ``runner.type=ovh`` into the YAML output → marks dirty → Step 3
    # gates Step 4 until saved. Save the variant so preflight can run.
    at.text_input(key="step3_new_name").set_value("seed0_ovh.yaml")
    at.run()
    at.button(key="step3_save").click()
    at.run()


@when("I run preflight")
def _run_preflight(context: dict) -> None:
    at = context["at"]
    at.button(key="step4_run").click()
    at.run()


@when("I click Submit with a stubbed LocalRunner")
def _click_submit_with_stub(context: dict, monkeypatch) -> None:
    from multi_scenario.frontend.preflight import CheckStatus

    at = context["at"]
    rows = at.session_state["submit_preflight"]
    if not all(r.status == CheckStatus.PASS for r in rows):
        import pytest as _pt

        failing = [r.name for r in rows if r.status != CheckStatus.PASS]
        _pt.skip(f"local preflight not green on this machine: {failing}")

    class _FakeResult:
        run_id = "demo_s0"

    def _fake_run(self, cfg, run_dir, resume_from=None):  # noqa: ARG001
        return _FakeResult()

    monkeypatch.setattr(
        "multi_scenario.adapters.runners.local.LocalRunner.run",
        _fake_run,
    )
    at.button(key="step5_submit").click()
    at.run()


# ── Thens ───────────────────────────────────────────────────────────


@then(parsers.parse('the submission status is "{expected}"'))
def _status_is(context: dict, expected: str) -> None:
    at = context["at"]
    status = at.session_state["submit_submission_status"]
    assert status["status"] == expected, status


@then(parsers.parse('the "{row_name}" row is FAIL'))
def _row_fail(context: dict, row_name: str) -> None:
    from multi_scenario.frontend.preflight import CheckStatus

    rows = {r.name: r for r in context["at"].session_state["submit_preflight"]}
    assert row_name in rows, f"unknown row: {row_name}; have {sorted(rows)}"
    assert rows[row_name].status == CheckStatus.FAIL


@then("every other OVH probe row stays IDLE")
def _other_rows_idle(context: dict) -> None:
    from multi_scenario.frontend.preflight import CheckStatus

    rows = {r.name: r for r in context["at"].session_state["submit_preflight"]}
    cloud_rows = (
        "OVH CLI installed",
        "Results bucket reachable",
        "Code matches OVH bucket",
        "Per-run prefix not occupied",
        "Submitted YAML present in bucket",
        "No active OVH job with this run_id",
        "Cost cap not exceeded",
    )
    for name in cloud_rows:
        assert (
            rows[name].status == CheckStatus.IDLE
        ), f"{name} should cascade to IDLE; got {rows[name].status}"
        assert "fix the OVH config row first" in rows[name].detail
