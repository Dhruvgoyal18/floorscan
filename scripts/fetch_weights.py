"""Fetch learned-model weights for the video/photo tiers into the local Hugging Face cache.

Run once with network (``make weights``); afterwards the pipeline runs offline (it loads with
local_files_only=True). Nothing here is needed for the LiDAR tier.

  MapAnything (Apache-2.0 weights)          facebook/map-anything-apache        ~1.2 GB
  Depth-Anything-V2 Metric Indoor Small     depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf   ~100 MB
"""
from __future__ import annotations

import argparse
import sys

MODELS = {
    "mapanything": "facebook/map-anything-apache",
    "depthanything": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(MODELS), default=None)
    a = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install -e '.[models]' first (make setup-models)")
    for name, repo in MODELS.items():
        if a.only and name != a.only:
            continue
        print(f"fetching {name}: {repo}", flush=True)
        path = snapshot_download(repo_id=repo)
        print(f"  -> {path}")
    print("done. The video/photo tiers now run offline.")


if __name__ == "__main__":
    main()
