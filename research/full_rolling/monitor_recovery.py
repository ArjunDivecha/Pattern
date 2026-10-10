"""Synchronize surviving Air jobs while local workers continue in memory."""
from pathlib import Path
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/full_rolling_clean_2003_2026_seed101"
HOST = "air@arjuns-macbook-air.taild3d657.ts.net"
REMOTE = "/Users/air/pattern-worker"
REL = "runs/full_rolling_clean_2003_2026_seed101"
SKIP = {"2009_rolling5_seed101"}


def main() -> None:
    cfg = json.loads((OUT / "config.json").read_text())
    while True:
        states = []
        for job in cfg["jobs"]:
            name = job["name"]
            if name not in SKIP and job["worker"] == "air":
                subprocess.run(
                    ["rsync", "-az", "--exclude=images.npy", "--exclude=sample.parquet",
                     f"{HOST}:{REMOTE}/{REL}/{name}/", str(OUT / name) + "/"],
                    check=True,
                )
            status = OUT / name / "status.json"
            states.append(json.loads(status.read_text()).get("state") if status.exists() else "queued")
        (OUT / "status.json").write_text(json.dumps({
            "state": "complete" if states.count("complete") == len(cfg["jobs"]) else "running",
            "completed": states.count("complete"),
            "failed": states.count("failed"),
            "recovery": "Air 2009 reassigned to main; Air 2015/2021 selected worker",
        }, indent=2))
        if states.count("complete") == len(cfg["jobs"]):
            return
        if "failed" in states:
            raise RuntimeError("full rolling fit failed")
        time.sleep(60)


if __name__ == "__main__":
    main()
