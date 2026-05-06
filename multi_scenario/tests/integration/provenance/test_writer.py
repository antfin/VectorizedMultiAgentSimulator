"""F2.7 tests: ProvenanceWriter — config_hash, code_hash, library_versions."""

from pathlib import Path

from multi_scenario.adapters.provenance.writer import ProvenanceWriter
from multi_scenario.domain.hashing import compute_config_hash
from multi_scenario.domain.models import ExperimentConfig


def _cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "test", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1},
            "evaluation": {"interval_iters": 1, "episodes": 1},
        }
    )


def test_writer_produces_provenance_with_correct_config_hash():
    """ProvenanceWriter's config_hash matches compute_config_hash on the same data."""
    cfg = _cfg()
    writer = ProvenanceWriter()
    p = writer(cfg)
    assert p.config_hash == compute_config_hash(cfg.model_dump(mode="json"))


def test_writer_includes_populated_library_versions():
    """library_versions has every field as a non-empty string."""
    p = ProvenanceWriter()(_cfg())
    lv = p.library_versions
    for value in (
        lv.python,
        lv.torch,
        lv.vmas,
        lv.benchmarl,
        lv.multi_scenario,
    ):
        assert isinstance(value, str) and value != ""


def test_writer_with_no_source_files_yields_empty_code_hash():
    """code_hash falls back to a stable sentinel when no files are configured."""
    p = ProvenanceWriter(hashed_source_files=())(_cfg())
    assert p.code_hash == "sha256:empty"
    assert p.hashed_source_files == []


def test_writer_with_real_files_produces_real_code_hash(tmp_path: Path):
    """When given concrete files, code_hash is a real sha256."""
    f = tmp_path / "a.py"
    f.write_text("print('hi')\n", encoding="utf-8")
    p = ProvenanceWriter(hashed_source_files=[f])(_cfg())
    assert p.code_hash.startswith("sha256:")
    assert p.code_hash != "sha256:empty"
    assert p.hashed_source_files == [str(f)]
