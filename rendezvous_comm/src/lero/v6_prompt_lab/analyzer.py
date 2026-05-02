"""AST-based static analysis for v6 prompt-lab.

Detects cross-source operations in `enhance_observation` Python code:
expressions where one operand depends on `lidar_targets` and the
other on `lidar_agents` (or other distinct input families). This is
the structural signature of S3b-local's coordination signals
(`settle_signal`, `rendezvous_pressure`) that v6 candidates uniformly
fail to produce.

Why AST not regex: many v6 candidates assign intermediate variables
(`t_close = lidar_t < r; a_close = lidar_a < r; ...`) and only later
use them. Regex over the source can miss the cross-source binary op
when the operands are intermediate names. AST data-flow handles this
correctly by propagating the source-set through assignments.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# Input keys the runtime patched scenario actually provides in
# obs_state_mode=local. Used as the leaf "sources" for data-flow.
INPUT_KEYS: Set[str] = {
    "lidar_targets",
    "lidar_agents",
    "agent_pos",
    "agent_vel",
    "agent_idx",
    "n_agents",
    "n_targets",
    "covering_range",
    "agents_per_target_required",
    "messages",
    "collision_rew",
    "time_penalty",
}

# The two sources whose combination is the structural signature of
# coordination cues. A binary op with operands tracing to both is a
# cross-source operation.
COORDINATION_SOURCES = ("lidar_targets", "lidar_agents")


@dataclass
class CrossSourceOp:
    """A single detected cross-source operation."""
    op_name: str           # e.g. "BinOp.Mult", "Compare.And"
    line: int
    left_sources: Tuple[str, ...]
    right_sources: Tuple[str, ...]
    code_excerpt: str = ""


@dataclass
class CodeAnalysis:
    ast_valid: bool
    has_enhance_observation: bool
    cross_source_ops: List[CrossSourceOp] = field(default_factory=list)
    var_sources: Dict[str, Set[str]] = field(default_factory=dict)
    n_returned_features: int = 0
    # v7 feature-stack-completeness checks (post deep-analysis 2026-04-30):
    has_directional_encoding: bool = False  # cos/sin of argmin ray index
    has_role_one_hot: bool = False           # F.one_hot or [:, agent_idx] = 1.0
    has_motion_feature: bool = False          # norm(agent_vel) or scalar velocity magnitude
    has_covering_range_threshold: bool = False  # uses covering_range as threshold
    error: str = ""

    @property
    def n_cross_source(self) -> int:
        return len(self.cross_source_ops)

    @property
    def has_cross_source(self) -> bool:
        return self.n_cross_source > 0

    @property
    def touches_both_lidars(self) -> bool:
        """True if any op crosses lidar_targets and lidar_agents."""
        target = "lidar_targets"
        agent = "lidar_agents"
        for op in self.cross_source_ops:
            if (target in op.left_sources and agent in op.right_sources) \
                    or (agent in op.left_sources and target in op.right_sources):
                return True
        return False

    @property
    def feature_stack_score(self) -> int:
        """0-5 integer score: how complete is the feature stack vs S3b-local?

        Counts presence of: cross_source ops, directional encoding,
        role one-hot, motion feature, covering_range threshold.
        S3b-local's iter-1 winner scores 5/5; v6 candidates typically
        score 1-2/5.
        """
        return (
            int(self.has_cross_source)
            + int(self.has_directional_encoding)
            + int(self.has_role_one_hot)
            + int(self.has_motion_feature)
            + int(self.has_covering_range_threshold)
        )


# ── AST data-flow ────────────────────────────────────────────────


def _extract_subscript_key(node: ast.AST) -> str | None:
    """If node is `scenario_state["lidar_targets"]` or similar,
    return the string key. Otherwise return None.
    """
    if not isinstance(node, ast.Subscript):
        return None
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value
    return None


def _extract_get_call(node: ast.AST) -> str | None:
    """If node is `scenario_state.get("lidar_targets", ...)`, return
    the key. Otherwise None.
    """
    if not (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return None
    return node.args[0].value


# Type/shape metadata attributes — using them does NOT mean the
# expression carries the underlying tensor's data dependence.
# `lidar_t.dtype` is a torch.dtype object, not data; using it does
# not propagate "lidar_targets" into the result.
_METADATA_ATTRS = {
    "dtype", "device", "shape", "ndim", "requires_grad", "size",
    "is_cuda", "is_floating_point",
}


class _SourceVisitor(ast.NodeVisitor):
    """Compute the set of INPUT_KEYS that an arbitrary expression
    transitively depends on, given a current var_sources mapping.

    Skips metadata-only paths (`.dtype`, `.shape`, etc.) because these
    return type/shape metadata that doesn't flow data into downstream
    computations for the purpose of detecting coordination operations.
    """

    def __init__(self, var_sources: Dict[str, Set[str]]):
        self.var_sources = var_sources
        self.found: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        sources = self.var_sources.get(node.id)
        if sources:
            self.found.update(sources)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = _extract_subscript_key(node)
        if key is not None and key in INPUT_KEYS:
            self.found.add(key)
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        key = _extract_get_call(node)
        if key is not None and key in INPUT_KEYS:
            self.found.add(key)
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # `x.dtype`, `x.shape`, etc. — metadata only; skip recursion.
        if node.attr in _METADATA_ATTRS:
            return
        self.generic_visit(node)


def _expr_sources(expr: ast.AST,
                  var_sources: Dict[str, Set[str]]) -> Set[str]:
    v = _SourceVisitor(var_sources)
    v.visit(expr)
    return v.found


# ── Main analyzer ────────────────────────────────────────────────


def analyze_inner_code(code: str) -> CodeAnalysis:
    """Parse and statically analyze an `enhance_observation` source
    string. Returns a CodeAnalysis with cross-source op count + the
    list of detected ops.
    """
    out = CodeAnalysis(ast_valid=False, has_enhance_observation=False)
    try:
        tree = ast.parse(code)
        out.ast_valid = True
    except SyntaxError as e:
        out.error = f"SyntaxError: {e}"
        return out

    # Find the enhance_observation function.
    target_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "enhance_observation":
            target_fn = node
            break
    if target_fn is None:
        out.error = "no enhance_observation function found"
        return out
    out.has_enhance_observation = True

    # Walk function body in order, propagate var_sources, flag binary
    # ops whose two operands have disjoint source-sets that together
    # cover lidar_targets AND lidar_agents.
    var_sources: Dict[str, Set[str]] = {}

    for stmt in ast.walk(target_fn):
        # Track assignments
        if isinstance(stmt, ast.Assign):
            rhs_sources = _expr_sources(stmt.value, var_sources)
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    var_sources[tgt.id] = rhs_sources
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            var_sources[elt.id] = rhs_sources

        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            rhs_sources = _expr_sources(stmt.value, var_sources)
            if isinstance(stmt.target, ast.Name):
                var_sources[stmt.target.id] = rhs_sources

        elif isinstance(stmt, ast.AugAssign):
            rhs_sources = _expr_sources(stmt.value, var_sources)
            if isinstance(stmt.target, ast.Name):
                prev = var_sources.get(stmt.target.id, set())
                var_sources[stmt.target.id] = prev | rhs_sources

    out.var_sources = {k: set(v) for k, v in var_sources.items()}

    # Now detect cross-source binary operations: walk again, for each
    # BinOp, BoolOp, Compare check if left/right operand source-sets
    # together cover both lidar_targets and lidar_agents.
    target = "lidar_targets"
    agent = "lidar_agents"

    def _is_cross_source(left_src: Set[str], right_src: Set[str]) -> bool:
        a = (target in left_src and agent in right_src)
        b = (agent in left_src and target in right_src)
        return a or b

    code_lines = code.splitlines()

    for node in ast.walk(target_fn):
        if isinstance(node, ast.BinOp):
            ls = _expr_sources(node.left, var_sources)
            rs = _expr_sources(node.right, var_sources)
            if _is_cross_source(ls, rs):
                out.cross_source_ops.append(CrossSourceOp(
                    op_name=f"BinOp.{type(node.op).__name__}",
                    line=node.lineno,
                    left_sources=tuple(sorted(ls)),
                    right_sources=tuple(sorted(rs)),
                    code_excerpt=code_lines[node.lineno - 1].strip()
                    if 1 <= node.lineno <= len(code_lines) else "",
                ))
        elif isinstance(node, ast.Compare):
            # Compare: x < y, treats left and the first comparator as a
            # binary op for our purposes.
            if node.comparators:
                ls = _expr_sources(node.left, var_sources)
                rs = _expr_sources(node.comparators[0], var_sources)
                if _is_cross_source(ls, rs):
                    out.cross_source_ops.append(CrossSourceOp(
                        op_name="Compare",
                        line=node.lineno,
                        left_sources=tuple(sorted(ls)),
                        right_sources=tuple(sorted(rs)),
                        code_excerpt=code_lines[node.lineno - 1].strip()
                        if 1 <= node.lineno <= len(code_lines) else "",
                    ))
        elif isinstance(node, ast.BoolOp) and len(node.values) >= 2:
            ls = _expr_sources(node.values[0], var_sources)
            rs = _expr_sources(node.values[1], var_sources)
            if _is_cross_source(ls, rs):
                out.cross_source_ops.append(CrossSourceOp(
                    op_name=f"BoolOp.{type(node.op).__name__}",
                    line=node.lineno,
                    left_sources=tuple(sorted(ls)),
                    right_sources=tuple(sorted(rs)),
                    code_excerpt=code_lines[node.lineno - 1].strip()
                    if 1 <= node.lineno <= len(code_lines) else "",
                ))

    # Estimate returned-feature count: best-effort from Return /
    # torch.cat with a list literal. Not exact (some candidates use
    # dynamic concatenation) but good enough for ranking.
    out.n_returned_features = _estimate_n_features(target_fn)

    # v7 feature-stack-completeness checks. Source-text grep is fine
    # here; these patterns are simple enough and rarely have false
    # positives. We use the original code string, not the AST, because
    # we want to catch e.g. `torch.cos(angle)` which can be written
    # multiple ways.
    out.has_directional_encoding = bool(
        ("torch.cos(" in code and "torch.sin(" in code)
        or ("F.cos(" in code and "F.sin(" in code)
        or ("math.cos(" in code and "math.sin(" in code)
    )
    out.has_role_one_hot = bool(
        "F.one_hot(" in code
        or "one_hot(" in code
        or "[:, agent_idx]" in code
        or "[: , agent_idx]" in code
    )
    out.has_motion_feature = bool(
        "agent_vel" in code
        and ("torch.linalg.norm" in code or "torch.norm" in code
             or "agent_vel ** 2" in code or "agent_vel.abs" in code
             or ".pow(2)" in code or "speed" in code)
    )
    # Threshold check: does the code reference covering_range as a
    # comparison threshold? Distinct from just reading the value.
    out.has_covering_range_threshold = bool(
        "covering_range" in code
        and ("< " + "covering_range" in code.replace("\n", " ")
             or "covering_range" + " >" in code.replace("\n", " ")
             or "<= covering_range" in code.replace("\n", " ")
             or "(lidar_t < cover_r" in code
             or "(lidar_a < cover_r" in code
             or "cover_r = " in code)  # local alias for covering_range
    )

    return out


# ── v8 helpers: count gated vs dense features ───────────────────


def count_gated_features(code: str) -> int:
    """Count cover-zone-gated features by source-text grep.

    A "gated feature" is one whose computation involves multiplying or
    masking by `near_t`, `near_a`, `(lidar_x < cover_r)`, etc. — i.e.,
    its value is zero outside the cover zone. PPO at 1M frames learns
    poorly from too many gated features (the agent rarely visits the
    cover zone early in training, so signal stays near zero).

    Source-text grep (not full AST data-flow) — false positives possible
    but acceptable for ranking. Looks for the patterns inner LLMs
    actually emit.
    """
    if not code:
        return 0
    n = 0
    # Pattern 1: explicit multiplication by a near_x / *_close boolean
    for needle in [
        "* near_t", "near_t *",
        "* near_a", "near_a *",
        "* t_close", "t_close *",
        "* a_close", "a_close *",
    ]:
        n += code.count(needle)
    # Pattern 2: inline conjunction of (lidar_x < cover_r) factors
    import re as _re
    pattern = _re.compile(
        r"\(\s*lidar_[ta][^\)]*<\s*cover_r[^\)]*\)\s*\.?(to|float)?[\s\.\*\(\)]*"
        r"\*\s*\(\s*lidar_[ta][^\)]*<\s*cover_r"
    )
    n += len(pattern.findall(code))
    return n


def count_dense_features(analysis: "CodeAnalysis", code: str) -> int:
    """Estimate dense-signal feature count.

    A "dense" feature produces informative values everywhere in state
    space (mean, std, min, count, normalized count, signed difference
    of normalized counts, boundary distance, role one-hot, etc.). We
    count by subtracting gated from total — imprecise but useful for
    ranking.
    """
    n_gated = count_gated_features(code)
    return max(0, analysis.n_returned_features - n_gated)


def _estimate_n_features(fn: ast.FunctionDef) -> int:
    """Best-effort count of features in the returned tensor.

    Looks for any `torch.cat([...])` or `torch.stack([...])` call
    anywhere in the function body and counts the list elements. Many
    candidates assign the cat result to a variable (e.g. `feats =
    torch.cat([...]); return feats`) so we walk the whole body, not
    just the Return statement.
    """
    counts = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("cat", "stack")
                    and node.args
                    and isinstance(node.args[0], ast.List)):
                counts.append(len(node.args[0].elts))
    # Take the max — likely the actual return concat (others may be
    # intermediate stacks for direction encoding etc.)
    return max(counts) if counts else 0


# ── Convenience for batch analysis ──────────────────────────────


@dataclass
class BatchSummary:
    n_total: int
    n_ast_valid: int
    n_with_enhance_observation: int
    n_with_any_cross_source: int
    n_touches_both_lidars: int
    avg_cross_source_ops: float
    avg_returned_features: float


def summarize_batch(analyses: List[CodeAnalysis]) -> BatchSummary:
    n = len(analyses)
    if n == 0:
        return BatchSummary(0, 0, 0, 0, 0, 0.0, 0.0)
    return BatchSummary(
        n_total=n,
        n_ast_valid=sum(1 for a in analyses if a.ast_valid),
        n_with_enhance_observation=sum(
            1 for a in analyses if a.has_enhance_observation
        ),
        n_with_any_cross_source=sum(1 for a in analyses if a.has_cross_source),
        n_touches_both_lidars=sum(
            1 for a in analyses if a.touches_both_lidars
        ),
        avg_cross_source_ops=sum(a.n_cross_source for a in analyses) / n,
        avg_returned_features=sum(a.n_returned_features for a in analyses) / n,
    )
