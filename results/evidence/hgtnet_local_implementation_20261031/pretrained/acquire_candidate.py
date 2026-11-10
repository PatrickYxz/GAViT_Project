"""Bounded manual official-candidate acquisition, separate from model loader."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "checkpoints/hgtnet_pretrained_20261031"
EVIDENCE = Path(__file__).resolve().parent
URLS = [
    "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
    "https://download-r2.pytorch.org/models/swin_b-68c6b09e.pth",
]
report = {"candidate": "torchvision.Swin_B_Weights.IMAGENET1K_V1", "is_original_hgtnet_weights": False,
          "started_utc": datetime.now(timezone.utc).isoformat(), "attempts": []}
start = time.monotonic()
for index, url in enumerate(URLS):
    partial = OUT / f"swin_b-68c6b09e.attempt{index + 1}.part"
    log = EVIDENCE / f"download_attempt{index + 1}.log"
    with log.open("wb") as stream:
        result = subprocess.run(["curl", "--fail", "--location", "--proto", "=https", "--tlsv1.2", "--connect-timeout", "20", "--max-time", "140", "--output", str(partial), url], stdout=stream, stderr=subprocess.STDOUT)
    entry = {"url": url, "exit_code": result.returncode, "bytes": partial.stat().st_size if partial.exists() else 0,
             "log": str(log.relative_to(ROOT))}
    report["attempts"].append(entry)
    if result.returncode == 0:
        digest = hashlib.file_digest(partial.open("rb"), "sha256").hexdigest()
        entry["sha256"] = digest
        if digest.startswith("68c6b09e"):
            final = OUT / "swin_b-68c6b09e.pth"
            if final.exists():
                raise RuntimeError("Refusing to overwrite candidate file")
            partial.rename(final)
            report.update(status="downloaded_official_candidate_bytes", path=str(final), url=url,
                          sha256=digest, bytes=final.stat().st_size)
            break
        entry["error"] = "actual hash does not match official filename prefix"
else:
    report["status"] = "pending_official_candidate_acquisition"
report["elapsed_seconds"] = time.monotonic() - start
report["ended_utc"] = datetime.now(timezone.utc).isoformat()
(EVIDENCE / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
