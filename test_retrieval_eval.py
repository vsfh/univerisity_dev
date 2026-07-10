from pathlib import Path
import importlib
import sys

from retrieval.config import load_config
from retrieval.eval import _checkpoint_path, evaluate


def test_eval_dry_run_resolves_relative_checkpoint(tmp_path: Path) -> None:
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(
        """
exp_name: eval
save_dir: /tmp/retrieval_eval
model:
  type: siglip
eval:
  output_dir: eval_results/retrieval/eval
  checkpoint: last.pth
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))

    result = evaluate(cfg, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["checkpoint"] == "/tmp/retrieval_eval/last.pth"
    assert _checkpoint_path(cfg) == "/tmp/retrieval_eval/last.pth"


def test_legacy_eval_import_does_not_require_unused_grounding_encoders(monkeypatch) -> None:
    retrieval_dir = Path(__file__).resolve().parent / "retrieval"
    monkeypatch.syspath_prepend(str(retrieval_dir))
    sys.modules.pop("retrieval.eval_retrieval", None)

    module = importlib.import_module("retrieval.eval_retrieval")

    assert hasattr(module, "_build_model_and_io")
