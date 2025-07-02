#!/usr/bin/env python3
"""Synchronise unchecked tasks in AGENTS.md to GitHub issues."""
import argparse
import os
import re
from typing import List, Tuple

import requests


def parse_tasks(path: str) -> List[Tuple[str, bool]]:
    pattern = re.compile(r"^- \[( |x)\] (.+)")
    tasks: List[Tuple[str, bool]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if m:
                checked = m.group(1) == "x"
                title = m.group(2).strip()
                tasks.append((title, checked))
    return tasks


def issue_exists(repo: str, token: str, title: str) -> bool:
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "open", "per_page": 100}
    headers = {"Authorization": f"token {token}"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    for issue in r.json():
        if issue.get("title") == title:
            return True
    return False


def create_issue(repo: str, token: str, title: str, body: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {"Authorization": f"token {token}"}
    json = {"title": title, "body": body}
    r = requests.post(url, headers=headers, json=json, timeout=10)
    r.raise_for_status()


def ensure_milestones(repo: str, token: str, names: List[str]) -> None:
    """Create milestones if they do not already exist."""
    url = f"https://api.github.com/repos/{repo}/milestones"
    headers = {"Authorization": f"token {token}"}
    r = requests.get(url, headers=headers, params={"state": "all"}, timeout=10)
    r.raise_for_status()
    existing = {m.get("title") for m in r.json()}
    for name in names:
        if name not in existing:
            requests.post(url, headers=headers, json={"title": name}, timeout=10)


def main(args: argparse.Namespace) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN environment variable not set")

    tasks = parse_tasks(args.file)
    body = "Created from AGENTS.md task list"
    for title, checked in tasks:
        if not checked and not issue_exists(args.repo, token, title):
            print(f"Creating issue: {title}")
            create_issue(args.repo, token, title, body)

    if args.milestones:
        ensure_milestones(args.repo, token, args.milestones)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync AGENTS.md tasks to GitHub issues"
    )
    parser.add_argument("--repo", required=True, help="<owner>/<repo>")
    parser.add_argument("--file", default="AGENTS.md", help="Path to AGENTS.md")
    parser.add_argument(
        "--milestones",
        nargs="*",
        default=[],
        help="Create these milestones if they do not exist",
    )
    args = parser.parse_args()
    main(args)
