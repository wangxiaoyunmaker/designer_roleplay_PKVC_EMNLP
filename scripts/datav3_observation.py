"""One-command runner for all data_v3 statistical observation layers.

Usage:
    .venv/bin/python scripts/datav3_observation.py            # 跑全部
    .venv/bin/python scripts/datav3_observation.py --skip L5  # 跳过指定 layer
    .venv/bin/python scripts/datav3_observation.py --only L4 L5  # 只跑指定
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_pipeline.eval.datav3_stats import (  # noqa: E402
    audit,
    bridge,
    cluster,
    cluster_multi,
    efa,
    gmm_behavior,
    health,
    lasso_30_items,
    lasso_check,
    mapping,
    summary,
    visualize,
)


LAYERS = [
    ("L1", "health.run", health.run),
    ("L7", "audit.run_l7", audit.run_l7),
    ("L1b", "audit.run_l1b", audit.run_l1b),
    ("L2", "efa.run", efa.run),
    ("L3", "cluster.run", cluster.run),
    ("L3-multi", "cluster_multi.run", cluster_multi.run),
    ("L3-GMM", "gmm_behavior.run", gmm_behavior.run),
    ("L3-visual", "visualize.run", visualize.run),
    ("L4", "mapping.run", mapping.run),
    ("L4b", "lasso_check.run", lasso_check.run),
    ("L4c", "lasso_30_items.run", lasso_30_items.run),
    ("L5", "bridge.run", bridge.run),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="layer names to skip")
    ap.add_argument("--only", nargs="*", default=None, help="run only these layers")
    args = ap.parse_args()

    for name, label, fn in LAYERS:
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        if args.only is not None and name not in args.only:
            continue
        print(f"\n=== {name} ({label}) ===")
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL: {e}")
            continue
        print(f"  done in {time.time() - t0:.1f}s")

    print("\n=== SUMMARY ===")
    summary.run()
    print("Wrote reports/datav3/SUMMARY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
