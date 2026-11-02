#!/usr/bin/env python
"""Compare the released System-1 router vs the ground-truth oracle (GTY)."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/mr/Desktop/Fireworks")
sys.path.insert(0, str(ROOT))
RELEASE = ROOT / "training/router_releases/router_8ac97088cc"

from router.eval_router import router_vs_gt

r = router_vs_gt(RELEASE)
print(json.dumps({k: v for k, v in r.items() if k not in ("preds", "records")}, indent=2))
print("\nPrediction distribution (top 8):")
n = r["test_examples"]
for m, c in Counter(r["preds"]).most_common(8):
    print(f"  {m:28s} {c:5d}  {100*c/n:5.1f}%")
