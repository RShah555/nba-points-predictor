"""Deploy the app to a public Hugging Face Space without publishing the code.

    python deploy/space.py --space <hf-user>/nba-predictor [--deploy-key path/to/key]

The Space gets only a Dockerfile (pinned to the current, pushed commit), a README and a snapshot
of the game logs (public NBA data, so startup does not have to rebuild them). The code is pulled
from the private GitHub repo at build time with a read-only deploy key stored as a Space secret.
Run it again after pushing new commits to redeploy.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: NBA Predictor
emoji: 🏀
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Next game projections, fantasy points and prop value
---

Projects every NBA player's next game: points, rebounds, assists and more as full ranges,
fantasy points for ESPN, Yahoo and Sleeper, start/sit odds and betting line value.
Live data from NBA.com and ESPN's injury report. Informational only.
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="<hf-user>/<space-name>")
    parser.add_argument("--repo", default="RShah555/nba-points-predictor", help="private GitHub repo")
    parser.add_argument("--deploy-key", type=Path, help="private half of the read-only GitHub deploy key")
    args = parser.parse_args()

    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Commit your changes first: the Space builds from a pushed commit.")
    if not git("branch", "-r", "--contains", sha):  # no remote branch has this commit
        raise SystemExit("Push first: the Space builds from GitHub.")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    if args.deploy_key:
        secret = base64.b64encode(args.deploy_key.read_bytes()).decode()
        api.add_space_secret(args.space, "DEPLOY_KEY", secret, description="read-only GitHub deploy key")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        template = (ROOT / "deploy" / "Dockerfile.space").read_text(encoding="utf-8")
        (out / "Dockerfile").write_text(template.replace("{repo}", args.repo).replace("{sha}", sha), encoding="utf-8")
        (out / "README.md").write_text(SPACE_README, encoding="utf-8")
        (out / "data").mkdir()
        from nba_predictor.seasons import current_season, previous_season

        for season in (previous_season(current_season()), current_season()):
            src = ROOT / "data" / f"player_logs_{season}.csv"
            if src.exists():
                shutil.copy(src, out / "data" / src.name)
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=out,
            commit_message=f"Deploy {sha[:7]}",
            delete_patterns=["data/*"],
        )
    user, name = args.space.split("/")
    print(f"Deployed {sha[:7]}. Building at https://huggingface.co/spaces/{args.space}")
    print(f"App URL once built: https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
