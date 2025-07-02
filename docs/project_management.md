# Project Management

This document outlines how to keep the GitHub Projects board in sync with the
roadmap defined in `AGENTS.md`.

## Sync unchecked tasks to issues

The script `scripts/sync_tasks_to_github.py` reads `AGENTS.md` and creates a
GitHub issue for every unchecked task. Set the `GITHUB_TOKEN` environment
variable to an access token with issue permissions and run:

```bash
python scripts/sync_tasks_to_github.py --repo <owner>/<repo>
```

Existing open issues with the same title will be skipped.

## Automating project updates

You can invoke the sync script from a GitHub Action whenever a pull request is
merged. Configure the workflow with the repository name and provide the token
via `secrets.GITHUB_TOKEN`.

## Managing milestones

Pass the `--milestones` option to create missing milestones automatically.

```bash
python scripts/sync_tasks_to_github.py --repo <owner>/<repo> \
  --milestones v0.2-ssl v0.3-transformer v1.0-release
```
