#!/usr/bin/env python3
"""Downloads the large dataset JSON files this repo needs but doesn't store in git.

These files (AgentAuditor/data/*.json, ASSEBench/dataset/*.json,
ASSEBench/category/{safety,security}/{f,r,s}.json) live in a GitHub Release instead of git
history, to keep clones of this repo lightweight. Git LFS was the first choice for this, but
GitHub blocks LFS object uploads to public forks (this repo is a fork) - see the "data-v1"
release's own notes for the underlying reason.

Run this once after cloning, and again any time the release is updated:
    python3 fetch_data.py

Requires the GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`) - the release
is only fetched via `gh release download`, no other credentials needed.
"""
import subprocess
import shutil
import sys
from pathlib import Path

REPO = "SaiShravanthReddy/AgentAuditor-ASSEBench"
RELEASE_TAG = "data-v1"

# asset filename in the release -> destination path(s) relative to repo root.
# Some destinations are duplicated (identical content checked into two paths in this repo before
# the migration) or renamed (to avoid filename collisions within the single release).
ASSET_MAP = {
    "AgentJudge-loose.json": ["AgentAuditor/data/AgentJudge-loose.json", "ASSEBench/dataset/AgentJudge-loose.json"],
    "AgentJudge-safety.json": ["AgentAuditor/data/AgentJudge-safety.json", "ASSEBench/dataset/AgentJudge-safety.json"],
    "AgentJudge-security.json": ["AgentAuditor/data/AgentJudge-security.json", "ASSEBench/dataset/AgentJudge-security.json"],
    "AgentJudge-strict.json": ["AgentAuditor/data/AgentJudge-strict.json", "ASSEBench/dataset/AgentJudge-strict.json"],
    "agentharm.json": ["AgentAuditor/data/agentharm.json"],
    "cnfinbench-harmful-unblocked.json": ["AgentAuditor/data/cnfinbench-harmful-unblocked.json"],
    "cnfinbench-harmful-unblocked.json.meta.json": ["AgentAuditor/data/cnfinbench-harmful-unblocked.json.meta.json"],
    "cnfinbench-harmful.json": ["AgentAuditor/data/cnfinbench-harmful.json"],
    "cnfinbench-harmful.json.meta.json": ["AgentAuditor/data/cnfinbench-harmful.json.meta.json"],
    "cnfinbench-harmless-unblocked.json": ["AgentAuditor/data/cnfinbench-harmless-unblocked.json"],
    "cnfinbench-harmless-unblocked.json.meta.json": ["AgentAuditor/data/cnfinbench-harmless-unblocked.json.meta.json"],
    "cnfinbench-harmless.json": ["AgentAuditor/data/cnfinbench-harmless.json"],
    "cnfinbench-harmless.json.meta.json": ["AgentAuditor/data/cnfinbench-harmless.json.meta.json"],
    "cnfinbench-pooled.json": ["AgentAuditor/data/cnfinbench-pooled.json"],
    "cnfinbench-pooled.json.meta.json": ["AgentAuditor/data/cnfinbench-pooled.json.meta.json"],
    "rjudge.json": ["AgentAuditor/data/rjudge.json"],
    "category-safety-f.json": ["ASSEBench/category/safety/f.json"],
    "category-safety-r.json": ["ASSEBench/category/safety/r.json"],
    "category-safety-s.json": ["ASSEBench/category/safety/s.json"],
    "category-security-f.json": ["ASSEBench/category/security/f.json"],
    "category-security-r.json": ["ASSEBench/category/security/r.json"],
    "category-security-s.json": ["ASSEBench/category/security/s.json"],
}


def main():
    if shutil.which("gh") is None:
        print("ERROR: GitHub CLI ('gh') not found. Install it and run 'gh auth login' first.",
              file=sys.stderr)
        sys.exit(1)

    repo_root = Path(__file__).parent.resolve()
    staging_dir = repo_root / ".data_release_staging"
    staging_dir.mkdir(exist_ok=True)

    print(f"Downloading {len(ASSET_MAP)} assets from {REPO}@{RELEASE_TAG}...")
    result = subprocess.run(
        ["gh", "release", "download", RELEASE_TAG, "--repo", REPO,
         "--dir", str(staging_dir), "--clobber"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: gh release download failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    placed = 0
    missing = []
    for asset_name, destinations in ASSET_MAP.items():
        src = staging_dir / asset_name
        if not src.exists():
            missing.append(asset_name)
            continue
        for dest_rel in destinations:
            dest = repo_root / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            placed += 1

    shutil.rmtree(staging_dir)

    if missing:
        print(f"WARNING: {len(missing)} expected assets were not found in the release: {missing}",
              file=sys.stderr)

    print(f"Placed {placed} files across {len(ASSET_MAP)} assets. Done.")


if __name__ == "__main__":
    main()
