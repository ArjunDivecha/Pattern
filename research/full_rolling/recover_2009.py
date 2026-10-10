"""Run the exclusively reassigned 2009 fit after the Air loader stalled."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pattern.research.full_cache import atomic_json
from research.full_rolling.train import OUT, fit


def main() -> None:
    cfg = json.loads((OUT / "config.json").read_text())
    job = next(j for j in cfg["jobs"] if j["name"] == "2009_rolling5_seed101")
    status = OUT / job["name"] / "status.json"
    current = json.loads(status.read_text()) if status.exists() else {}
    if current.get("state") == "complete":
        return
    atomic_json(status, {"state": "reassigned_loading", "worker": "main_recovery"})
    try:
        fit(job, cfg)
    except BaseException as exc:
        atomic_json(status, {"state": "failed", "worker": "main_recovery", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
