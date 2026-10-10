"""Run only explicitly assigned full-study jobs on a remote worker."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pattern.research.full_cache import atomic_json
from research.full_rolling.train import OUT, fit


def main() -> None:
    names = set(sys.argv[1:])
    if not names:
        raise SystemExit("job names required")
    cfg = json.loads((OUT / "config.json").read_text())
    jobs = [j for j in cfg["jobs"] if j["name"] in names]
    if {j["name"] for j in jobs} != names:
        raise ValueError("unknown job name")
    for job in jobs:
        status = OUT / job["name"] / "status.json"
        current = json.loads(status.read_text()) if status.exists() else {}
        if current.get("state") == "complete":
            continue
        try:
            fit(job, cfg)
        except BaseException as exc:
            atomic_json(status, {"state": "failed", "worker": "air_selected", "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
