"""F9.4 — extract candidate code from LLM responses + AST validation.

Pure-domain. The orchestrator calls :func:`extract_candidates` on the
LLM's text output to harvest valid ``compute_reward`` /
``enhance_observation`` Python functions, dropping anything that fails
AST validation. The patcher (F9.5) then ``compile()`` s the validated
strings inside its own process boundary.

Byte-parity contract: for any list of LLM response strings, our
:func:`extract_candidates` returns the same ``CandidateCode`` list as
the rendezvous_comm reference (same ``reward_source`` / ``obs_source``
strings, same elision when validation fails). Pinned by
``tests/integration/lero/test_codegen_byte_parity.py``.

Hex-clean: imports stdlib + ``multi_scenario.domain.lero.candidate``
only. No torch, no LiteLLM. The ``ast`` module is stdlib so this is
free to import from any layer.
"""

import ast
import logging
import re

from multi_scenario.domain.lero.candidate import CandidateCode


_log = logging.getLogger(__name__)

#: Imports the LLM is allowed to use inside its generated functions.
#: Anything outside this set fails AST validation. Locked to match
#: rendezvous_comm/codegen.py::ALLOWED_IMPORTS — divergence here would
#: cause the byte-parity test to fail loudly.
ALLOWED_IMPORTS: frozenset[str] = frozenset({"torch", "math", "numpy"})

#: ```python ... ``` fenced block extractor. Multiline-DOTALL.
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_candidates(
    responses: list[str],
    *,
    evolve_reward: bool = True,
    evolve_observation: bool = True,
) -> list[CandidateCode]:
    """Extract validated (reward, obs) code pairs from LLM responses.

    For each response, look for ```python ... ``` blocks and pick:
    - one block defining ``compute_reward`` (or ``compute_reward_bonus``)
      when ``evolve_reward=True``
    - one block defining ``enhance_observation`` when
      ``evolve_observation=True``

    Each picked block must pass :func:`validate_function` (AST parse,
    expected function name, expected single argument ``scenario_state``,
    no imports outside :data:`ALLOWED_IMPORTS`).

    A response that yields no valid functions is silently dropped —
    this is the rendezvous_comm semantics. The orchestrator's
    "iteration produced N valid candidates" log message is what makes
    the dropped count visible.

    Args:
        responses: list of raw LLM response strings (one per sibling
            completion). Empty / None entries are skipped.
        evolve_reward: when False, reward blocks are ignored even if
            present in the response. The corresponding ``reward_source``
            field on the returned CandidateCode stays None.
        evolve_observation: same idea for observation blocks.

    Returns:
        One :class:`CandidateCode` per response that produced at least
        one valid function. Order matches the input ``responses`` order.
    """
    candidates: list[CandidateCode] = []
    for resp in responses:
        if not resp:
            _log.warning("empty/None LLM response, skipping")
            continue
        code_blocks = _CODE_BLOCK_RE.findall(resp)
        if not code_blocks:
            _log.warning("no ```python``` blocks found in LLM response")
            continue

        reward_src: str | None = None
        obs_src: str | None = None

        for block in code_blocks:
            block = block.strip()
            if "def compute_reward" in block and evolve_reward:
                # Accept both names: replace mode = compute_reward,
                # bonus mode = compute_reward_bonus.
                func_name = (
                    "compute_reward_bonus"
                    if "def compute_reward_bonus" in block
                    else "compute_reward"
                )
                if validate_function(block, func_name, ["scenario_state"]):
                    reward_src = block
                else:
                    _log.warning("%s failed AST validation", func_name)
            elif "def enhance_observation" in block and evolve_observation:
                if validate_function(block, "enhance_observation", ["scenario_state"]):
                    obs_src = block
                else:
                    _log.warning("enhance_observation failed AST validation")

        # Require at least one *requested* function to have been
        # extracted. Matches rendezvous_comm: a response that the
        # config asked to evolve nothing for would never produce a
        # candidate, but in practice evolve_* defaults to True.
        has_any = (evolve_reward and reward_src is not None) or (
            evolve_observation and obs_src is not None
        )
        if not has_any:
            _log.warning("no valid functions extracted from response")
            continue

        candidates.append(
            CandidateCode(
                reward_source=reward_src,
                obs_source=obs_src,
                raw_response=resp,
            )
        )
    return candidates


def validate_function(
    source: str,
    expected_name: str,
    expected_args: list[str],
) -> bool:
    """AST-validate one extracted function block.

    Five checks (all must pass):

    1. Source parses as valid Python (no SyntaxError).
    2. Source contains a ``def <expected_name>`` at any nesting depth.
    3. That function's positional args equal ``expected_args`` exactly.
    4. Every ``import X`` lists a module whose top-level name is in
       :data:`ALLOWED_IMPORTS`.
    5. Every ``from X import Y`` references a module whose top-level
       name is in :data:`ALLOWED_IMPORTS` (relative imports rejected
       via the ``node.module is None`` branch — defensive though
       LLM-generated code rarely emits them).

    Returns True iff all checks pass; logs a one-line warning at the
    first failure and returns False.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _log.warning("syntax error in generated code: %s", exc)
        return False

    func_def: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == expected_name:
            func_def = node
            break
    if func_def is None:
        _log.warning("function %r not found in code", expected_name)
        return False

    actual_args = [a.arg for a in func_def.args.args]
    if actual_args != expected_args:
        _log.warning(
            "function %r args %s != expected %s",
            expected_name,
            actual_args,
            expected_args,
        )
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in ALLOWED_IMPORTS:
                    _log.warning("disallowed import: %s", alias.name)
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                _log.warning("relative import not allowed in generated code")
                return False
            module = node.module.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                _log.warning("disallowed import from: %s", node.module)
                return False

    return True
