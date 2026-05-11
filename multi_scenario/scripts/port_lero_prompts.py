"""F9.2 — One-shot port of rendezvous_comm prompts → multi_scenario Jinja .j2.

Reads each ``rendezvous_comm/src/lero/prompts/<version>/{system,initial_user,
feedback}.txt``, translates ``$variable`` → ``{{ variable }}``, writes
the result as ``.j2`` under ``adapters/prompts/<version>/``. ``meta.yaml``
is copied verbatim (it carries author/date/description; no syntax to
translate).

Idempotent: run again after a rendezvous_comm prompt revision and the
.j2 files re-sync. Byte-parity test
(``tests/integration/prompts/test_prompt_byte_parity.py``) catches any
divergence at CI time.

Usage:
    python -m scripts.port_lero_prompts
"""

import re
import shutil
from pathlib import Path


_VAR_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
_TARGET_VERSIONS = (
    "v1",
    "v1_global",
    "v2",
    "v2_min",
    "v2_fewshot",
    "v2_twofn",
    "v2_fewshot_k2_local",
)
_TEMPLATES = ("system", "initial_user", "feedback")


def _translate(content: str) -> str:
    """``$variable`` → ``{{ variable }}``.

    Idempotent on already-Jinja text (``{{ var }}`` doesn't match
    ``$var``). No support for ``${var}`` / ``$$`` because the
    rendezvous_comm prompts never used those forms (verified via grep).
    """
    return _VAR_RE.sub(r"{{ \1 }}", content)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "rendezvous_comm" / "src" / "lero" / "prompts"
    dst_root = (
        repo_root / "multi_scenario" / "src" / "multi_scenario" / "adapters" / "prompts"
    )

    for version in _TARGET_VERSIONS:
        src_dir = src_root / version
        dst_dir = dst_root / version
        if not src_dir.is_dir():
            print(f"  ✗ skip {version}: source dir missing")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for template in _TEMPLATES:
            src_file = src_dir / f"{template}.txt"
            dst_file = dst_dir / f"{template}.j2"
            if not src_file.is_file():
                print(f"  ✗ skip {version}/{template}: source missing")
                continue
            content = src_file.read_text(encoding="utf-8")
            dst_file.write_text(_translate(content), encoding="utf-8")
        meta_src = src_dir / "meta.yaml"
        if meta_src.is_file():
            shutil.copyfile(meta_src, dst_dir / "meta.yaml")
        print(f"  ✓ ported {version}")


if __name__ == "__main__":
    main()
