"""F9.2 — :class:`JinjaPromptRenderer` contract.

Smaller-grained than the byte-parity test — pins the renderer's
behaviour against synthesised templates so a Jinja config drift
(``StrictUndefined``, ``keep_trailing_newline``, etc.) fails here
loudly instead of silently changing rendered prompts in production.
"""

# pylint: disable=missing-function-docstring

from pathlib import Path

import jinja2
import pytest

from multi_scenario.adapters.prompts import JinjaPromptRenderer


@pytest.fixture
def renderer_with_template(tmp_path: Path):
    def _factory(content: str, *, template: str = "system", version: str = "vt"):
        version_dir = tmp_path / version
        version_dir.mkdir()
        (version_dir / f"{template}.j2").write_text(content, encoding="utf-8")
        return JinjaPromptRenderer(root=tmp_path), version, template

    return _factory


def test_render_substitutes_context_vars(renderer_with_template):
    renderer, v, t = renderer_with_template(
        "agents={{ n_agents }}, targets={{ n_targets }}"
    )
    out = renderer.render(
        version=v, template=t, context={"n_agents": 4, "n_targets": 4}
    )
    assert out == "agents=4, targets=4"


def test_render_strict_undefined_raises_on_missing_var(renderer_with_template):
    """Missing variables must fail loudly so a typo doesn't silently
    render an empty string into the prompt the LLM sees."""
    renderer, v, t = renderer_with_template("agents={{ n_agents }}")
    with pytest.raises(jinja2.UndefinedError):
        renderer.render(version=v, template=t, context={})


def test_render_preserves_trailing_newline(renderer_with_template):
    """``keep_trailing_newline=True`` matters for byte-parity vs the
    rendezvous_comm string.Template output."""
    renderer, v, t = renderer_with_template("hello\n")
    out = renderer.render(version=v, template=t, context={})
    assert out == "hello\n"


def test_render_does_not_html_escape(renderer_with_template):
    """LLM prompts are plain text — angle brackets, ampersands, quotes
    must pass through verbatim."""
    renderer, v, t = renderer_with_template("name=<{{ name }}> & 'OK'")
    out = renderer.render(version=v, template=t, context={"name": "AGENT_1"})
    assert out == "name=<AGENT_1> & 'OK'"


def test_missing_template_file_raises_file_not_found(renderer_with_template):
    renderer, v, _t = renderer_with_template("hi")
    with pytest.raises(FileNotFoundError, match="prompt template not found"):
        renderer.render(version=v, template="nonexistent", context={})


def test_default_root_resolves_to_adapters_prompts_dir():
    """Without ``root``, the renderer reads from
    ``adapters/prompts/`` so production code doesn't have to wire it up."""
    from multi_scenario.adapters.prompts import DEFAULT_PROMPTS_ROOT

    renderer = JinjaPromptRenderer()
    # Should be able to load the v2_fewshot_k2_local/system.j2 we ported.
    out = renderer.render(
        version="v2_fewshot_k2_local",
        template="system",
        context={
            "n_agents": 4,
            "n_targets": 4,
            "agents_per_target": 2,
            "covering_range": 0.35,
            "experiment_context": "test",
            "comm_description": "",
            "agent_lidar_description": "",
        },
    )
    # Sanity: the system prompt mentions 'reward engineer' or similar.
    assert "reward engineer" in out or "observation" in out
    # The default root is what __init__ exports.
    assert DEFAULT_PROMPTS_ROOT.is_dir()


def test_renderer_satisfies_protocol():
    """``JinjaPromptRenderer`` structurally satisfies the
    :class:`PromptRenderer` Protocol — has the ``render`` method with
    the expected signature.

    The Protocol isn't ``@runtime_checkable`` (we want it pure-structural,
    no isinstance overhead). Test by callability + signature instead.
    """
    import inspect

    from multi_scenario.domain.ports import PromptRenderer  # noqa: F401

    renderer = JinjaPromptRenderer()
    sig = inspect.signature(renderer.render)
    assert {"version", "template", "context"} <= set(sig.parameters)
