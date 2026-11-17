#!/usr/bin/env python3
"""Tracked Git hook entry points for commit validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from precheck_service import (
    ValidationDatabase,
    default_cache_root,
    outgoing_commits,
    repository_root,
    result_fingerprint_matches,
    require_ready,
)


def pre_commit(root: Path):
    cache_root = default_cache_root(root)
    require_ready(root, cache_root)
    subprocess.run(
        [sys.executable, root / "scripts/format.py", "--check"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)


def post_commit(root: Path):
    cache_root = default_cache_root(root)
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    commit, inserted = database.enqueue(root, "HEAD")
    database.close()
    print(f"precheck_enqueued={commit} new={str(inserted).lower()}")


def pre_push(root: Path, remote_name: str | None):
    updates = sys.stdin.read().splitlines()
    try:
        commits = outgoing_commits(root, updates, remote_name)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not commits:
        return
    cache_root = default_cache_root(root)
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    missing = []
    blocked = []
    for commit in commits:
        row = database.connection.execute(
            "SELECT * FROM jobs WHERE commit_sha=?", (commit,)
        ).fetchone()
        if row is None:
            missing.append(commit)
            database.enqueue(root, commit)
        elif row["state"] != "passed" or not result_fingerprint_matches(
            root, cache_root, commit, row
        ):
            state = row["state"] if row["state"] != "passed" else "stale"
            blocked.append((commit, state, row["error_summary"]))
            if state == "stale":
                with database.immediate():
                    database.connection.execute(
                        """UPDATE jobs SET state='queued', slot=NULL,
                             worker_token=NULL, worker_pid=NULL, started_at=NULL,
                             heartbeat_at=NULL, finished_at=NULL, exit_code=NULL,
                             error_summary='validation fingerprint changed',
                             notified_at=NULL WHERE commit_sha=? AND state='passed'""",
                        (commit,),
                    )
    database.close()
    for commit in missing:
        print(f"unvalidated outgoing commit: {commit}", file=sys.stderr)
    for commit, state, error in blocked:
        print(
            f"outgoing commit is not validated: {commit} state={state}"
            + (f" error={error}" if error else ""),
            file=sys.stderr,
        )
    if missing or blocked:
        print(
            "wait with: python scripts/precheck_status.py --outgoing "
            "--wait --require-passed",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hook", choices=("pre-commit", "post-commit", "pre-push"))
    parser.add_argument("--remote-name")
    args = parser.parse_args()
    root = repository_root()
    if args.hook == "pre-commit":
        pre_commit(root)
    elif args.hook == "post-commit":
        post_commit(root)
    else:
        pre_push(root, args.remote_name)


if __name__ == "__main__":
    main()
