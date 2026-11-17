#!/usr/bin/env python3
"""Durable background validation for exact Git commit snapshots."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid


PROTOCOL_VERSION = 1
HEARTBEAT_SECONDS = 2.0
HEALTH_TIMEOUT_SECONDS = 10.0
TERMINAL_STATES = {"passed", "failed", "infrastructure_failed", "cancelled"}
ZERO_OID = "0" * 40
PRECHECK_BASELINE_FILE = ".precheck-baseline"


def console_python() -> Path:
    executable = Path(sys.executable)
    if os.name == "nt" and executable.name.lower() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.is_file():
            return candidate
    return executable


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(command, *, cwd=None, check=True, capture_output=False, env=None):
    return subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture_output,
        env=env,
    )


def git(root: Path, *arguments: str) -> str:
    return run(
        ["git", *arguments], cwd=root, capture_output=True
    ).stdout.strip()


def repository_root(path: Path | str = ".") -> Path:
    return Path(git(Path(path).resolve(), "rev-parse", "--show-toplevel")).resolve()


def repository_id(root: Path) -> str:
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return hashlib.sha256(str(common.resolve()).encode("utf-8")).hexdigest()[:16]


def default_cache_root(root: Path) -> Path:
    override = os.environ.get("UDONCL_PRECHECK_CACHE_ROOT")
    if override:
        base = Path(override).expanduser()
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        base /= "ucl"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        base /= "udoncl/precheck"
    return (base / repository_id(root)).resolve()


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def age_seconds(value: str | None) -> float:
    if not value:
        return float("inf")
    timestamp = dt.datetime.fromisoformat(value)
    return (dt.datetime.now(dt.timezone.utc) - timestamp).total_seconds()


class ValidationDatabase:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS service (
              singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
              token TEXT NOT NULL,
              pid INTEGER NOT NULL,
              hostname TEXT NOT NULL,
              protocol_version INTEGER NOT NULL,
              started_at TEXT NOT NULL,
              heartbeat_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
              commit_sha TEXT PRIMARY KEY,
              tree_sha TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN (
                'queued', 'running', 'passed', 'failed',
                'infrastructure_failed', 'cancelled')),
              attempt INTEGER NOT NULL DEFAULT 0,
              slot INTEGER,
              worker_token TEXT,
              worker_pid INTEGER,
              enqueued_at TEXT NOT NULL,
              started_at TEXT,
              heartbeat_at TEXT,
              finished_at TEXT,
              platform TEXT,
              unity_version TEXT,
              native_passed INTEGER,
              native_failed INTEGER,
              unity_passed INTEGER,
              unity_failed INTEGER,
              unity_skipped INTEGER,
              cleanup_passed INTEGER,
              dependency_fingerprint TEXT,
              validation_fingerprint TEXT,
              exit_code INTEGER,
              error_summary TEXT,
              notified_at TEXT
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              commit_sha TEXT NOT NULL REFERENCES jobs(commit_sha)
                ON DELETE CASCADE,
              kind TEXT NOT NULL,
              path TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              PRIMARY KEY(commit_sha, kind)
            );
            CREATE TABLE IF NOT EXISTS leases (
              resource TEXT PRIMARY KEY,
              owner_job TEXT NOT NULL,
              worker_token TEXT NOT NULL,
              worker_pid INTEGER NOT NULL,
              heartbeat_at TEXT NOT NULL
            );
            """
        )
        schema = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if schema is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', '1')"
            )
        elif schema["value"] != "1":
            raise RuntimeError(f"unsupported precheck database schema {schema['value']}")

    def close(self):
        self.connection.close()

    @contextlib.contextmanager
    def immediate(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def service_row(self):
        return self.connection.execute(
            "SELECT * FROM service WHERE singleton=1"
        ).fetchone()

    def service_healthy(self) -> bool:
        row = self.service_row()
        return bool(
            row
            and row["protocol_version"] == PROTOCOL_VERSION
            and age_seconds(row["heartbeat_at"]) <= HEALTH_TIMEOUT_SECONDS
            and process_alive(row["pid"])
        )

    def acquire_service(self, token: str):
        with self.immediate():
            row = self.service_row()
            if row and row["token"] != token and self.service_healthy():
                raise RuntimeError(
                    f"precheck coordinator is already running as pid {row['pid']}"
                )
            timestamp = now()
            self.connection.execute(
                """INSERT OR REPLACE INTO service(
                     singleton, token, pid, hostname, protocol_version,
                     started_at, heartbeat_at)
                   VALUES(1, ?, ?, ?, ?, ?, ?)""",
                (
                    token,
                    os.getpid(),
                    socket.gethostname(),
                    PROTOCOL_VERSION,
                    timestamp,
                    timestamp,
                ),
            )
            self.connection.execute(
                "DELETE FROM metadata WHERE key='stop_requested'"
            )

    def heartbeat_service(self, token: str):
        self.connection.execute(
            "UPDATE service SET heartbeat_at=?, pid=? WHERE singleton=1 AND token=?",
            (now(), os.getpid(), token),
        )

    def release_service(self, token: str):
        self.connection.execute(
            "DELETE FROM service WHERE singleton=1 AND token=?", (token,)
        )

    def stop_requested(self, token: str) -> bool:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='stop_requested'"
        ).fetchone()
        return bool(row and row["value"] == token)

    def enqueue(self, root: Path, commit: str) -> tuple[str, bool]:
        commit_sha = git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
        tree_sha = git(root, "show", "-s", "--format=%T", commit_sha)
        with self.immediate():
            inserted = self.connection.execute(
                """INSERT OR IGNORE INTO jobs(
                     commit_sha, tree_sha, state, enqueued_at)
                   VALUES(?, ?, 'queued', ?)""",
                (commit_sha, tree_sha, now()),
            ).rowcount
        return commit_sha, bool(inserted)

    def claim(self, slot: int, worker_token: str):
        with self.immediate():
            row = self.connection.execute(
                """SELECT * FROM jobs WHERE state='queued'
                   ORDER BY enqueued_at, commit_sha LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            changed = self.connection.execute(
                """UPDATE jobs SET state='running', attempt=attempt+1,
                     slot=?, worker_token=?, worker_pid=NULL,
                     started_at=?, heartbeat_at=?
                   WHERE commit_sha=? AND state='queued'""",
                (slot, worker_token, now(), now(), row["commit_sha"]),
            ).rowcount
            return row["commit_sha"] if changed else None

    def set_worker_pid(self, commit: str, token: str, pid: int):
        self.connection.execute(
            """UPDATE jobs SET worker_pid=?, heartbeat_at=?
               WHERE commit_sha=? AND state='running' AND worker_token=?""",
            (pid, now(), commit, token),
        )

    def heartbeat_job(self, commit: str, token: str):
        self.connection.execute(
            """UPDATE jobs SET heartbeat_at=? WHERE commit_sha=?
               AND state='running' AND worker_token=?""",
            (now(), commit, token),
        )

    def acquire_lease(self, resource: str, commit: str, token: str) -> bool:
        with self.immediate():
            row = self.connection.execute(
                "SELECT * FROM leases WHERE resource=?", (resource,)
            ).fetchone()
            if row and process_alive(row["worker_pid"]):
                return row["worker_token"] == token
            self.connection.execute("DELETE FROM leases WHERE resource=?", (resource,))
            self.connection.execute(
                """INSERT INTO leases(resource, owner_job, worker_token,
                   worker_pid, heartbeat_at) VALUES(?, ?, ?, ?, ?)""",
                (resource, commit, token, os.getpid(), now()),
            )
            return True

    def heartbeat_lease(self, resource: str, token: str):
        self.connection.execute(
            "UPDATE leases SET heartbeat_at=? WHERE resource=? AND worker_token=?",
            (now(), resource, token),
        )

    def release_lease(self, resource: str, token: str):
        self.connection.execute(
            "DELETE FROM leases WHERE resource=? AND worker_token=?",
            (resource, token),
        )

    def finish(self, commit: str, token: str, result: dict, artifacts: list[dict]):
        state = result["state"]
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state {state}")
        with self.immediate():
            changed = self.connection.execute(
                """UPDATE jobs SET state=?, finished_at=?, platform=?,
                     unity_version=?, native_passed=?, native_failed=?,
                     unity_passed=?, unity_failed=?, unity_skipped=?,
                     cleanup_passed=?, dependency_fingerprint=?,
                     validation_fingerprint=?, exit_code=?, error_summary=?
                   WHERE commit_sha=? AND state='running' AND worker_token=?""",
                (
                    state,
                    now(),
                    result.get("platform"),
                    result.get("unity_version"),
                    result.get("native_passed"),
                    result.get("native_failed"),
                    result.get("unity_passed"),
                    result.get("unity_failed"),
                    result.get("unity_skipped"),
                    result.get("cleanup_passed"),
                    result.get("dependency_fingerprint"),
                    result.get("validation_fingerprint"),
                    result.get("exit_code"),
                    result.get("error_summary"),
                    commit,
                    token,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("job lease was lost before result commit")
            for artifact in artifacts:
                self.connection.execute(
                    """INSERT OR REPLACE INTO artifacts(
                         commit_sha, kind, path, sha256, size_bytes)
                       VALUES(?, ?, ?, ?, ?)""",
                    (
                        commit,
                        artifact["kind"],
                        artifact["path"],
                        artifact["sha256"],
                        artifact["size_bytes"],
                    ),
                )

    def recover(self):
        with self.immediate():
            rows = self.connection.execute(
                "SELECT * FROM jobs WHERE state='running'"
            ).fetchall()
            for row in rows:
                stale = age_seconds(row["heartbeat_at"]) > HEALTH_TIMEOUT_SECONDS * 3
                dead = row["worker_pid"] and not process_alive(row["worker_pid"])
                if stale or dead:
                    state = "queued" if row["attempt"] < 2 else "infrastructure_failed"
                    self.connection.execute(
                        """UPDATE jobs SET state=?, slot=NULL, worker_token=NULL,
                           worker_pid=NULL, error_summary=? WHERE commit_sha=?""",
                        (state, "worker heartbeat expired", row["commit_sha"]),
                    )


def file_artifact(path: Path, kind: str) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "kind": kind,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def ensure_within(path: Path, parent: Path):
    path.resolve().relative_to(parent.resolve())


def remove_owned(path: Path, parent: Path):
    ensure_within(path, parent)
    target = path
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        target = Path("\\\\?\\" + str(path.resolve()))
    if path.is_dir():
        def clear_read_only(function, failing_path, _error):
            os.chmod(failing_path, stat.S_IWRITE)
            function(failing_path)

        shutil.rmtree(target, onexc=clear_read_only)
    else:
        target.unlink(missing_ok=True)


def sync_unity_project(template: Path, destination: Path, cache_root: Path):
    ensure_within(destination, cache_root)
    contaminated = destination / ".udoncl-contaminated"
    if contaminated.exists():
        remove_owned(destination, cache_root)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("Assets", "Packages", "ProjectSettings"):
        source = template / name
        if not source.is_dir():
            raise RuntimeError(f"Unity template is missing {name}: {template}")
        target = destination / name
        if target.exists():
            remove_owned(target, destination)
        shutil.copytree(source, target)
    for name in ("Temp", "Logs", "Obj"):
        target = destination / name
        if target.exists():
            remove_owned(target, destination)


def directory_fingerprint(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        base = root / name
        if not base.is_dir():
            digest.update(f"missing:{name}\n".encode())
            continue
        for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                digest.update(f"dir:{relative}\n".encode())
            elif path.is_file():
                digest.update(f"file:{relative}\n".encode())
                digest.update(path.read_bytes())
            else:
                digest.update(f"other:{relative}\n".encode())
    return digest.hexdigest()


def prepare_worktree(root: Path, destination: Path, commit: str, cache_root: Path):
    ensure_within(destination, cache_root)
    checkout_marker = destination.parent / ".udoncl-lf-checkout"
    if (destination / ".git").exists() and not checkout_marker.is_file():
        for partial in (destination / ".deps", *destination.glob(".deps.new-*")):
            if partial.exists():
                remove_owned(partial, destination)
        run(["git", "worktree", "remove", "--force", destination], cwd=root)
        if destination.exists():
            remove_owned(destination, cache_root)
    if not (destination / ".git").exists():
        if destination.exists():
            remove_owned(destination, cache_root)
        run(["git", "worktree", "prune"], cwd=root)
        run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "worktree",
                "add",
                "--detach",
                destination,
                commit,
            ],
            cwd=root,
        )
        checkout_marker.write_text("core.autocrlf=false\n", encoding="utf-8")
    else:
        run(["git", "-c", "core.autocrlf=false", "reset", "--hard"], cwd=destination)
        run(["git", "clean", "-fd"], cwd=destination)
        run(
            ["git", "-c", "core.autocrlf=false", "checkout", "--detach", commit],
            cwd=destination,
        )
    actual = git(destination, "rev-parse", "HEAD")
    if actual != commit:
        raise RuntimeError(f"worker checked out {actual}, expected {commit}")
    for partial in destination.glob(".deps.new-*"):
        remove_owned(partial, destination)


def hardlink_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def clone_dependency_tree(source: Path, destination: Path):
    temporary = destination.with_name(destination.name + ".new-" + uuid.uuid4().hex)
    shutil.copytree(source, temporary, copy_function=hardlink_or_copy)
    os.replace(temporary, destination)


def clone_dependency_prefixes(
    source: Path, destination: Path, names: tuple[str, ...]
):
    temporary = destination.with_name(destination.name + ".new-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        for name in names:
            prefix = source / name
            if prefix.is_dir():
                shutil.copytree(
                    prefix, temporary / name, copy_function=hardlink_or_copy
                )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def bootstrap_dependencies(
    root: Path,
    worktree: Path,
    cache_root: Path,
    database: ValidationDatabase,
    commit: str,
    token: str,
    log,
):
    local = worktree / ".deps"
    lock_digest = dependency_lock_fingerprint(worktree, commit)
    shared = cache_root / "dependencies" / lock_digest
    lease_name = f"dependencies:{lock_digest}"
    required = ("llvm23", "onnx-mlir")
    if local.exists():
        remove_owned(local, worktree)
    while not database.acquire_lease(lease_name, commit, token):
        time.sleep(1)
    try:
        marker = shared / ".udoncl-cache-fingerprint"
        shared_valid = (
            shared.is_dir()
            and all((shared / name).exists() for name in required)
            and marker.is_file()
            and marker.read_text(encoding="utf-8").strip().startswith(
                f"locks={lock_digest}\n"
            )
        )
        if shared.exists() and not shared_valid:
            remove_owned(shared, cache_root)
        if not shared.exists():
            shared.parent.mkdir(parents=True, exist_ok=True)
            seed = root / ".deps"
            if seed.is_dir():
                clone_dependency_prefixes(seed, local, required)
            else:
                local.mkdir(parents=True)
            subprocess.run(
                [console_python(), worktree / "scripts/install_llvm.py"],
                cwd=worktree,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            subprocess.run(
                [console_python(), worktree / "scripts/install_onnx_mlir.py"],
                cwd=worktree,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            receipt_digest = dependency_fingerprint(worktree)
            (local / ".udoncl-cache-fingerprint").write_text(
                f"locks={lock_digest}\nreceipts={receipt_digest}\n",
                encoding="utf-8",
            )
            os.replace(local, shared)
        clone_dependency_tree(shared, local)
    finally:
        database.release_lease(lease_name, token)


def unity_version(unity: Path) -> str:
    match = re.search(r"Editor[/\\]([^/\\]+)[/\\]Editor", str(unity))
    return match.group(1) if match else "unknown"


def configured_unity(root: Path) -> tuple[Path, Path]:
    values = parse_dotenv(root / ".env")
    unity_value = os.environ.get("UNITY_EDITOR_PATH") or values.get("UNITY_EDITOR_PATH")
    template_value = os.environ.get("UDONCL_UNITY_TEST_PROJECT") or values.get(
        "UDONCL_UNITY_TEST_PROJECT"
    )
    if not unity_value or not template_value:
        raise RuntimeError("UNITY_EDITOR_PATH and UDONCL_UNITY_TEST_PROJECT are required")
    return (
        Path(unity_value).expanduser().resolve(),
        Path(template_value).expanduser().resolve(),
    )


def dependency_lock_fingerprint(root: Path, commit: str) -> str:
    digest = hashlib.sha256()
    for relative in ("cmake/llvm-lock.json", "cmake/onnx-mlir-lock.json"):
        content = git(root, "show", f"{commit}:{relative}")
        digest.update(relative.encode())
        digest.update(content.encode())
    return digest.hexdigest()


def validation_fingerprint(
    root: Path,
    cache_root: Path,
    commit: str,
    unity: Path,
    template_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"protocol={PROTOCOL_VERSION}\ncommit={commit}\n".encode())
    digest.update(f"unity={unity_version(unity)}\nplatform={platform.system()}\n".encode())
    digest.update(f"template={template_digest}\n".encode())
    digest.update(f"locks={dependency_lock_fingerprint(root, commit)}\n".encode())
    digest.update(Path(__file__).read_bytes())
    lock_digest = dependency_lock_fingerprint(root, commit)
    dependency_root = cache_root / "dependencies" / lock_digest
    if not dependency_root.is_dir():
        dependency_root = root / ".deps"
    for relative in (
        "llvm23/.udoncl-sdk.json",
        "onnx-mlir/.udoncl-onnx-mlir.json",
    ):
        path = dependency_root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes() if path.is_file() else b"missing")
    return digest.hexdigest()


def result_fingerprint_matches(
    root: Path, cache_root: Path, commit: str, row: sqlite3.Row
) -> bool:
    if row["state"] != "passed" or not row["validation_fingerprint"]:
        return False
    try:
        unity, template = configured_unity(root)
        template_digest = directory_fingerprint(
            template, ("Assets", "Packages", "ProjectSettings")
        )
        expected = validation_fingerprint(
            root, cache_root, commit, unity, template_digest
        )
    except (OSError, RuntimeError):
        return False
    return row["validation_fingerprint"] == expected


def dependency_fingerprint(worktree: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        ".deps/llvm23/.udoncl-sdk.json",
        ".deps/onnx-mlir/.udoncl-onnx-mlir.json",
    ):
        path = worktree / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes() if path.is_file() else b"missing")
    return digest.hexdigest()


def execute_job(root: Path, cache_root: Path, commit: str, slot: int, token: str):
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    database.set_worker_pid(commit, token, os.getpid())
    stop = threading.Event()

    def heartbeat():
        heartbeat_database = ValidationDatabase(cache_root / "queue.sqlite3")
        try:
            while not stop.wait(HEARTBEAT_SECONDS):
                heartbeat_database.heartbeat_job(commit, token)
        finally:
            heartbeat_database.close()

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    job_root = cache_root / "jobs" / commit
    job_root.mkdir(parents=True, exist_ok=True)
    log_path = job_root / "precheck.log"
    artifacts = []
    result = {
        "state": "infrastructure_failed",
        "platform": platform.system().lower(),
        "exit_code": 1,
        "cleanup_passed": 0,
    }
    worker_root = cache_root / "w" / str(slot)
    worktree = worker_root / "r"
    # Build products contain deeply nested CMake target paths. Keep them under
    # the short, reusable worker-slot root so Windows' legacy MAX_PATH does not
    # consume another full commit SHA for every generated object name.
    build = worker_root / "b"
    output = worker_root / "o"
    unity_project = worker_root / "u"
    in_use = unity_project / ".udoncl-in-use"
    try:
        testing = os.environ.get("UDONCL_PRECHECK_TESTING") == "1"
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            if testing:
                delay = float(os.environ.get("UDONCL_PRECHECK_TEST_DELAY", "0"))
                time.sleep(delay)
                desired = os.environ.get("UDONCL_PRECHECK_TEST_RESULT", "passed")
                return_code = 0 if desired == "passed" else 1
                log.write(f"test_gate={desired}\n")
            else:
                unity, template = configured_unity(root)
                if not unity.is_file():
                    raise RuntimeError(f"Unity executable does not exist: {unity}")
                prepare_worktree(root, worktree, commit, cache_root)
                remove_owned(build, cache_root)
                remove_owned(output, cache_root)
                if in_use.exists():
                    remove_owned(unity_project, cache_root)
                template_digest = directory_fingerprint(
                    template, ("Assets", "Packages", "ProjectSettings")
                )
                sync_unity_project(template, unity_project, cache_root)
                if template_digest != directory_fingerprint(
                    template, ("Assets", "Packages", "ProjectSettings")
                ):
                    raise RuntimeError("Unity template changed while creating worker clone")
                in_use.write_text(f"commit={commit}\npid={os.getpid()}\n", encoding="utf-8")
                bootstrap_dependencies(
                    root,
                    worktree,
                    cache_root,
                    database,
                    commit,
                    token,
                    log,
                )
                result["dependency_fingerprint"] = dependency_fingerprint(worktree)
                environment = dict(os.environ)
                environment["UDONCL_MODEL_CACHE"] = str(cache_root / "models")
                command = [
                    console_python(),
                    worktree / "scripts/precommit.py",
                    "--commit-snapshot",
                    commit,
                    "--unity",
                    unity,
                    "--project",
                    unity_project,
                    "--build-dir",
                    build,
                    "--output",
                    output,
                ]
                completed = subprocess.run(
                    [str(item) for item in command],
                    cwd=worktree,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
                return_code = completed.returncode
                result["unity_version"] = unity_version(unity)
                result["validation_fingerprint"] = validation_fingerprint(
                    root, cache_root, commit, unity, template_digest
                )
        content = log_path.read_text(encoding="utf-8", errors="replace")
        native = re.findall(r"100% tests passed, (\d+) tests failed out of (\d+)", content)
        if native:
            failed, total = map(int, native[0])
            result["native_failed"] = failed
            result["native_passed"] = total - failed
        unity_match = re.search(r"unity_mcp_tests=passed tests=(\d+) cleanup=passed", content)
        if unity_match:
            result["unity_passed"] = int(unity_match.group(1))
            result["unity_failed"] = 0
            result["unity_skipped"] = 0
            result["cleanup_passed"] = 1
        result["exit_code"] = return_code
        if return_code == 0:
            result["state"] = "passed"
            result["cleanup_passed"] = 1
            (unity_project / ".udoncl-contaminated").unlink(missing_ok=True)
            in_use.unlink(missing_ok=True)
        else:
            result["state"] = "failed"
            result["error_summary"] = "\n".join(content.splitlines()[-20:])[-4000:]
            unity_project.mkdir(parents=True, exist_ok=True)
            (unity_project / ".udoncl-contaminated").write_text(
                f"commit={commit}\n", encoding="utf-8"
            )
    except Exception as error:
        result["state"] = "infrastructure_failed"
        result["error_summary"] = f"{type(error).__name__}: {error}"
        unity_project.mkdir(parents=True, exist_ok=True)
        (unity_project / ".udoncl-contaminated").write_text(
            f"commit={commit}\n", encoding="utf-8"
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(result["error_summary"] + "\n")
    finally:
        stop.set()
        heartbeat_thread.join(timeout=5)
        if log_path.is_file():
            artifacts.append(file_artifact(log_path, "precheck_log"))
        database.finish(commit, token, result, artifacts)
        database.close()
    if result["state"] != "passed":
        notify_failure(root, commit, result.get("error_summary", "validation failed"))


def notify_failure(root: Path, commit: str, summary: str):
    if os.environ.get("UDONCL_PRECHECK_TESTING") == "1":
        return
    title = "UdonCL commit validation failed"
    message = f"{commit[:12]}: {summary.splitlines()[-1][:160]}"
    if os.name == "nt":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            with contextlib.suppress(Exception):
                environment = dict(os.environ)
                environment["UDONCL_NOTIFY_TITLE"] = title
                environment["UDONCL_NOTIFY_MESSAGE"] = message
                script = (
                    "Add-Type -AssemblyName System.Windows.Forms;"
                    "$n=New-Object System.Windows.Forms.NotifyIcon;"
                    "$n.Icon=[System.Drawing.SystemIcons]::Error;"
                    "$n.BalloonTipTitle=$env:UDONCL_NOTIFY_TITLE;"
                    "$n.BalloonTipText=$env:UDONCL_NOTIFY_MESSAGE;"
                    "$n.Visible=$true;$n.ShowBalloonTip(8000);Start-Sleep -Seconds 9;"
                    "$n.Dispose()"
                )
                subprocess.Popen(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                    cwd=Path.home(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
    else:
        notifier = shutil.which("notify-send")
        if notifier:
            with contextlib.suppress(Exception):
                subprocess.Popen(
                    [notifier, title, message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


def terminate_process_tree(process: subprocess.Popen):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


def terminate_pid_tree(pid: int):
    if not process_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if process_alive(pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)


def recover_orphan_workers(database: ValidationDatabase):
    rows = database.connection.execute(
        "SELECT * FROM jobs WHERE state='running'"
    ).fetchall()
    for row in rows:
        if row["worker_pid"]:
            terminate_pid_tree(row["worker_pid"])
    with database.immediate():
        for row in rows:
            state = "queued" if row["attempt"] < 2 else "infrastructure_failed"
            database.connection.execute(
                """UPDATE jobs SET state=?, slot=NULL, worker_token=NULL,
                     worker_pid=NULL, error_summary=? WHERE commit_sha=?
                     AND state='running'""",
                (state, "coordinator restarted during validation", row["commit_sha"]),
            )


def coordinator(root: Path, cache_root: Path, jobs: int, once: bool):
    cache_root.mkdir(parents=True, exist_ok=True)
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    token = uuid.uuid4().hex
    database.acquire_service(token)
    recover_orphan_workers(database)
    database.recover()
    stopping = False

    def stop_service(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_service)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop_service)
    active: dict[int, tuple[subprocess.Popen, str, str]] = {}
    try:
        while not stopping:
            database.heartbeat_service(token)
            if database.stop_requested(token):
                stopping = True
                continue
            for slot, (process, commit, worker_token) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                active.pop(slot)
                row = database.connection.execute(
                    "SELECT state FROM jobs WHERE commit_sha=?", (commit,)
                ).fetchone()
                if row and row["state"] == "running":
                    database.finish(
                        commit,
                        worker_token,
                        {
                            "state": "infrastructure_failed",
                            "platform": platform.system().lower(),
                            "exit_code": code,
                            "cleanup_passed": 0,
                            "error_summary": "worker exited without recording a result",
                        },
                        [],
                    )
            for slot in range(jobs):
                if slot in active:
                    continue
                worker_token = uuid.uuid4().hex
                commit = database.claim(slot, worker_token)
                if not commit:
                    continue
                command = [
                    console_python(),
                    Path(__file__).resolve(),
                    "execute",
                    "--repo",
                    root,
                    "--cache-root",
                    cache_root,
                    "--commit",
                    commit,
                    "--slot",
                    str(slot),
                    "--token",
                    worker_token,
                ]
                process_options = {"cwd": root}
                if os.name == "nt":
                    process_options["creationflags"] = getattr(
                        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                    )
                else:
                    process_options["start_new_session"] = True
                process = subprocess.Popen(
                    [str(item) for item in command], **process_options
                )
                database.set_worker_pid(commit, worker_token, process.pid)
                active[slot] = (process, commit, worker_token)
            queued = database.connection.execute(
                "SELECT count(*) FROM jobs WHERE state='queued'"
            ).fetchone()[0]
            if once and not active and not queued:
                break
            time.sleep(0.5)
    finally:
        if active:
            for process, _, _ in active.values():
                with contextlib.suppress(Exception):
                    terminate_process_tree(process)
        database.release_service(token)
        database.close()


def start_background(root: Path, cache_root: Path, jobs: int):
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    if database.service_healthy():
        database.close()
        return
    database.close()
    cache_root.mkdir(parents=True, exist_ok=True)
    log = (cache_root / "service.log").open("a", encoding="utf-8")
    command = [
        console_python(),
        Path(__file__).resolve(),
        "run",
        "--repo",
        root,
        "--cache-root",
        cache_root,
        "--jobs",
        str(jobs),
    ]
    kwargs = {
        "cwd": root,
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([str(item) for item in command], **kwargs)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        time.sleep(0.2)
        check = ValidationDatabase(cache_root / "queue.sqlite3")
        healthy = check.service_healthy()
        check.close()
        if healthy:
            return
    raise RuntimeError(f"precheck coordinator did not become ready; see {log.name}")


def wait_for_service(cache_root: Path, timeout: float = 15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        database = ValidationDatabase(cache_root / "queue.sqlite3")
        healthy = database.service_healthy()
        database.close()
        if healthy:
            return
        time.sleep(0.2)
    raise RuntimeError("installed precheck service did not become ready")


def require_ready(root: Path, cache_root: Path):
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    if not database.service_healthy():
        database.close()
        raise SystemExit("precheck coordinator is not running or its heartbeat is stale")
    database.close()
    hooks = git(root, "config", "--local", "--get", "core.hooksPath")
    if Path(hooks).as_posix().rstrip("/") != ".githooks":
        raise SystemExit("repository hooks are not installed; run precheck_service.py install")
    values = parse_dotenv(root / ".env")
    unity = os.environ.get("UNITY_EDITOR_PATH") or values.get("UNITY_EDITOR_PATH")
    project = os.environ.get("UDONCL_UNITY_TEST_PROJECT") or values.get(
        "UDONCL_UNITY_TEST_PROJECT"
    )
    if not unity or not Path(unity).expanduser().is_file():
        raise SystemExit("UNITY_EDITOR_PATH is not configured or does not exist")
    if not project or not (Path(project).expanduser() / "Packages/manifest.json").is_file():
        raise SystemExit("UDONCL_UNITY_TEST_PROJECT is not a Unity project template")


def install_hooks(root: Path):
    existing = run(
        ["git", "config", "--local", "--get", "core.hooksPath"],
        cwd=root,
        check=False,
        capture_output=True,
    ).stdout.strip()
    if existing and Path(existing).as_posix().rstrip("/") != ".githooks":
        raise RuntimeError(
            f"core.hooksPath is already set to {existing}; preserve or migrate it first"
        )
    run(["git", "config", "--local", "core.hooksPath", ".githooks"], cwd=root)
    run(
        ["git", "config", "--local", "udoncl.validationPython", console_python()],
        cwd=root,
    )


def install_system_service(root: Path, cache_root: Path, jobs: int):
    identity = repository_id(root)
    if os.name == "nt":
        task_name = f"UdonCL-Precheck-{identity}"
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        executable = pythonw if pythonw.is_file() else Path(sys.executable)
        action = subprocess.list2cmdline(
            [
                str(executable),
                str(Path(__file__).resolve()),
                "run",
                "--repo",
                str(root),
                "--cache-root",
                str(cache_root),
                "--jobs",
                str(jobs),
            ]
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        launcher = cache_root / "start-precheck-service.cmd"
        launcher.write_text("@echo off\r\n" + action + "\r\n", encoding="utf-8")
        task_action = subprocess.list2cmdline(
            ["cmd.exe", "/d", "/c", str(launcher)]
        )
        try:
            run(
                [
                    "schtasks",
                    "/Create",
                    "/TN",
                    task_name,
                    "/SC",
                    "ONLOGON",
                    "/RL",
                    "LIMITED",
                    "/TR",
                    task_action,
                    "/F",
                ]
            )
            powershell = shutil.which("powershell") or shutil.which("pwsh")
            if powershell:
                run(
                    [
                        powershell,
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "$settings=New-ScheduledTaskSettingsSet "
                        "-RestartCount 999 "
                        "-RestartInterval (New-TimeSpan -Minutes 1) "
                        "-MultipleInstances IgnoreNew; "
                        "Set-ScheduledTask -TaskName $args[0] "
                        "-Settings $settings | Out-Null",
                        task_name,
                    ]
                )
            run(["schtasks", "/Run", "/TN", task_name])
            return
        except subprocess.CalledProcessError:
            install_windows_startup_fallback(cache_root, action, identity)
            return
    elif platform.system() == "Linux":
        unit_dir = Path.home() / ".config/systemd/user"
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_name = f"udoncl-precheck-{identity}.service"
        unit = unit_dir / unit_name
        command = " ".join(
            systemd_quote(value)
            for value in (
                sys.executable,
                Path(__file__).resolve(),
                "run",
                "--repo",
                root,
                "--cache-root",
                cache_root,
                "--jobs",
                jobs,
            )
        )
        unit.write_text(
            "[Unit]\nDescription=UdonCL commit validation service\n\n"
            "[Service]\nType=simple\n"
            f"WorkingDirectory={systemd_quote(root)}\n"
            f"ExecStart={command}\n"
            "Restart=on-failure\nRestartSec=3\n\n"
            "[Install]\nWantedBy=default.target\n",
            encoding="utf-8",
        )
        run(["systemctl", "--user", "daemon-reload"])
        run(["systemctl", "--user", "enable", "--now", unit_name])
        return
    else:
        raise RuntimeError(f"unsupported service platform: {platform.system()}")


def systemd_quote(value) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("$", "$$").replace("%", "%%")
    return f'"{escaped}"'


def windows_startup_path(identity: str) -> Path:
    return (
        Path(os.environ["APPDATA"])
        / "Microsoft/Windows/Start Menu/Programs/Startup"
        / f"UdonCL-Precheck-{identity}.vbs"
    )


def install_windows_startup_fallback(cache_root: Path, action: str, identity: str):
    enabled = cache_root / "service-enabled"
    enabled.write_text("enabled\n", encoding="utf-8")
    supervisor = cache_root / "supervise-precheck-service.cmd"
    supervisor.write_text(
        "@echo off\r\n"
        ":restart\r\n"
        f'if not exist "{enabled}" exit /b 0\r\n'
        + action
        + "\r\n"
        "timeout /t 5 /nobreak >nul\r\n"
        "goto restart\r\n",
        encoding="utf-8",
    )
    startup = windows_startup_path(identity)
    startup.parent.mkdir(parents=True, exist_ok=True)
    startup.write_text(
        'Set shell = CreateObject("WScript.Shell")\r\n'
        f'shell.Run """{supervisor}""", 0, False\r\n',
        encoding="utf-8",
    )
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    healthy = database.service_healthy()
    database.close()
    if not healthy:
        subprocess.Popen(
            ["wscript.exe", str(startup)],
            cwd=cache_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def stop_service(cache_root: Path):
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    row = database.service_row()
    if not row or not process_alive(row["pid"]):
        database.close()
        return
    with database.immediate():
        database.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('stop_requested', ?)",
            (row["token"],),
        )
    database.close()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        check = ValidationDatabase(cache_root / "queue.sqlite3")
        stopped = check.service_row() is None
        check.close()
        if stopped or not process_alive(row["pid"]):
            return
        time.sleep(0.2)
    if process_alive(row["pid"]):
        raise RuntimeError(f"precheck coordinator pid {row['pid']} did not stop")


def uninstall_system_service(root: Path):
    identity = repository_id(root)
    if os.name == "nt":
        cache_root = default_cache_root(root)
        (cache_root / "service-enabled").unlink(missing_ok=True)
        windows_startup_path(identity).unlink(missing_ok=True)
        subprocess.run(
            ["schtasks", "/Delete", "/TN", f"UdonCL-Precheck-{identity}", "/F"],
            check=False,
        )
    elif platform.system() == "Linux":
        unit_name = f"udoncl-precheck-{identity}.service"
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", unit_name], check=False
        )
        (Path.home() / ".config/systemd/user" / unit_name).unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)


def unreported_failures(database: ValidationDatabase):
    return database.connection.execute(
        """SELECT * FROM jobs WHERE state IN ('failed', 'infrastructure_failed')
           AND notified_at IS NULL ORDER BY finished_at"""
    ).fetchall()


def print_job(row):
    print(
        f"commit={row['commit_sha']} state={row['state']} attempt={row['attempt']}"
        + (f" error={row['error_summary']}" if row["error_summary"] else "")
    )


def validation_baseline(root: Path) -> str:
    path = root / PRECHECK_BASELINE_FILE
    try:
        baseline = git(root, "show", f"HEAD:{PRECHECK_BASELINE_FILE}").strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"missing tracked push validation baseline: {path}") from error
    if not re.fullmatch(r"[0-9a-fA-F]{40}", baseline):
        raise ValueError(
            f"{PRECHECK_BASELINE_FILE} must contain one full commit SHA"
        )
    result = run(
        ["git", "cat-file", "-e", f"{baseline}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"push validation baseline is not a local commit: {baseline}"
        )
    return baseline.lower()


def require_baseline_ancestor(root: Path, baseline: str, tip: str):
    result = run(
        ["git", "merge-base", "--is-ancestor", baseline, tip],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"pushed tip {tip} is not a descendant of validation baseline "
            f"{baseline}"
        )


def outgoing_commits(
    root: Path,
    updates: list[str],
    remote_name: str | None = None,
) -> list[str]:
    baseline = validation_baseline(root)
    commits: set[str] = set()
    for line in updates:
        fields = line.split()
        if len(fields) != 4:
            continue
        _, local_sha, _, remote_sha = fields
        if set(local_sha) == {"0"}:
            continue
        local_commit = git(root, "rev-parse", "--verify", f"{local_sha}^{{commit}}")
        require_baseline_ancestor(root, baseline, local_commit)
        if set(remote_sha) == {"0"}:
            command = ["rev-list", local_commit]
            if remote_name:
                remote_refs = git(
                    root,
                    "for-each-ref",
                    "--format=%(refname)",
                    f"refs/remotes/{remote_name}/",
                ).splitlines()
                if remote_refs:
                    command.extend(["--not", *remote_refs])
            else:
                command.extend(["--not", "--remotes"])
        else:
            command = ["rev-list", f"{remote_sha}..{local_commit}"]
        candidates = git(root, *command).splitlines()
        if set(remote_sha) == {"0"} and local_commit not in candidates:
            candidates.append(local_commit)
        eligible = set(git(root, "rev-list", f"{baseline}..{local_commit}").splitlines())
        eligible.add(baseline)
        commits.update(commit for commit in candidates if commit in eligible)
    return sorted(commits)


def status_main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect commit validation results")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--require-enqueued", action="store_true")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--unreported", action="store_true")
    parser.add_argument("--outgoing", action="store_true")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--remote-name")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--require-passed", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = repository_root(args.repo)
    cache_root = (args.cache_root or default_cache_root(root)).resolve()
    database = ValidationDatabase(cache_root / "queue.sqlite3")
    commits: list[str] = []
    try:
        if args.commit:
            commits = [
                git(root, "rev-parse", "--verify", f"{args.commit}^{{commit}}")
            ]
        elif args.stdin:
            commits = outgoing_commits(
                root,
                sys.stdin.read().splitlines(),
                args.remote_name,
            )
        elif args.outgoing:
            upstream = run(
                [
                    "git",
                    "rev-parse",
                    "--abbrev-ref",
                    "--symbolic-full-name",
                    "@{u}",
                ],
                cwd=root,
                check=False,
                capture_output=True,
            )
            if upstream.returncode == 0:
                remote_sha = git(root, "rev-parse", upstream.stdout.strip())
                commits = outgoing_commits(
                    root,
                    [f"refs/heads/current HEAD refs/heads/upstream {remote_sha}"],
                )
            else:
                commits = outgoing_commits(
                    root,
                    [f"refs/heads/current HEAD refs/heads/current {ZERO_OID}"],
                )
    except ValueError as error:
        database.close()
        parser.error(str(error))
    if args.failures:
        query = "SELECT * FROM jobs WHERE state IN ('failed','infrastructure_failed')"
        parameters: list[str] = []
        if args.unreported:
            query += " AND notified_at IS NULL"
        if args.commit or args.stdin or args.outgoing:
            if commits:
                query += " AND commit_sha IN (" + ",".join("?" for _ in commits) + ")"
                parameters.extend(commits)
            else:
                query += " AND 0"
        rows = database.connection.execute(
            query + " ORDER BY finished_at", parameters
        ).fetchall()
    else:
        rows = []
        for commit in commits:
            row = database.connection.execute(
                "SELECT * FROM jobs WHERE commit_sha=?", (commit,)
            ).fetchone()
            if row is None and args.require_enqueued:
                database.close()
                raise SystemExit(f"commit is not enqueued: {commit}")
            if row:
                rows.append(row)
    if args.wait and commits:
        while True:
            rows = [
                database.connection.execute(
                    "SELECT * FROM jobs WHERE commit_sha=?", (commit,)
                ).fetchone()
                for commit in commits
            ]
            missing_commits = [
                commit for commit, row in zip(commits, rows) if row is None
            ]
            if missing_commits:
                database.close()
                raise SystemExit(
                    "commits are not enqueued: " + ", ".join(missing_commits)
                )
            if all(row and row["state"] in TERMINAL_STATES for row in rows):
                break
            time.sleep(1)
    if args.json:
        print(json.dumps([dict(row) for row in rows if row], indent=2))
    else:
        for row in rows:
            if row:
                print_job(row)
    if args.failures and args.unreported and rows:
        with database.immediate():
            for row in rows:
                database.connection.execute(
                    "UPDATE jobs SET notified_at=? WHERE commit_sha=?",
                    (now(), row["commit_sha"]),
                )
    failed = [
        row for row in rows if row and row["state"] in {"failed", "infrastructure_failed"}
    ]
    missing = len(rows) != len(commits) if commits else False
    pending = [row for row in rows if row and row["state"] not in TERMINAL_STATES]
    not_passed = [
        row
        for row in rows
        if row
        and (
            row["state"] != "passed"
            or not result_fingerprint_matches(
                root, cache_root, row["commit_sha"], row
            )
        )
    ]
    database.close()
    if args.require_passed and (missing or pending or not_passed):
        raise SystemExit("not every requested commit has passed validation")
    if failed:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "run",
        "start",
        "stop",
        "status",
        "install",
        "uninstall",
        "require-ready",
        "enqueue",
        "retry",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--cache-root", type=Path)
        if name in {"run", "start", "install"}:
            command.add_argument(
                "--jobs", type=int, default=int(os.environ.get("UDONCL_PRECHECK_JOBS", "2"))
            )
        if name == "run":
            command.add_argument("--once", action="store_true")
        if name == "install":
            command.add_argument("--no-system-service", action="store_true")
        if name == "enqueue":
            command.add_argument("commit", nargs="?", default="HEAD")
        if name == "retry":
            command.add_argument("commit", nargs="?", default="HEAD")
    execute = subparsers.add_parser("execute")
    execute.add_argument("--repo", type=Path, required=True)
    execute.add_argument("--cache-root", type=Path, required=True)
    execute.add_argument("--commit", required=True)
    execute.add_argument("--slot", type=int, required=True)
    execute.add_argument("--token", required=True)
    args = parser.parse_args(argv)
    if args.command == "execute":
        execute_job(
            repository_root(args.repo),
            args.cache_root.resolve(),
            args.commit,
            args.slot,
            args.token,
        )
        return
    root = repository_root(args.repo)
    cache_root = (args.cache_root or default_cache_root(root)).resolve()
    if args.command == "run":
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        coordinator(root, cache_root, args.jobs, args.once)
    elif args.command == "start":
        start_background(root, cache_root, args.jobs)
        print(f"precheck_service=ready jobs={args.jobs} database={cache_root / 'queue.sqlite3'}")
    elif args.command == "stop":
        stop_service(cache_root)
        print("precheck_service=stopped")
    elif args.command == "status":
        database = ValidationDatabase(cache_root / "queue.sqlite3")
        service = database.service_row()
        counts = database.connection.execute(
            "SELECT state, count(*) AS count FROM jobs GROUP BY state ORDER BY state"
        ).fetchall()
        healthy = database.service_healthy()
        database.close()
        print(
            f"precheck_service={'ready' if healthy else 'stopped'}"
            + (f" pid={service['pid']}" if service else "")
        )
        for count in counts:
            print(f"jobs_{count['state']}={count['count']}")
    elif args.command == "install":
        install_hooks(root)
        if args.no_system_service:
            start_background(root, cache_root, args.jobs)
        else:
            install_system_service(root, cache_root, args.jobs)
            wait_for_service(cache_root)
        print(f"precheck_service=installed jobs={args.jobs}")
    elif args.command == "require-ready":
        require_ready(root, cache_root)
        print("precheck_service=ready")
    elif args.command == "enqueue":
        database = ValidationDatabase(cache_root / "queue.sqlite3")
        commit, inserted = database.enqueue(root, args.commit)
        database.close()
        print(f"precheck_enqueued={commit} new={str(inserted).lower()}")
    elif args.command == "retry":
        commit = git(root, "rev-parse", "--verify", f"{args.commit}^{{commit}}")
        database = ValidationDatabase(cache_root / "queue.sqlite3")
        with database.immediate():
            changed = database.connection.execute(
                """UPDATE jobs SET state='queued', slot=NULL,
                     worker_token=NULL, worker_pid=NULL, started_at=NULL,
                     heartbeat_at=NULL, finished_at=NULL, exit_code=NULL,
                     error_summary=NULL, notified_at=NULL
                   WHERE commit_sha=? AND state IN (
                     'passed','failed','infrastructure_failed','cancelled')""",
                (commit,),
            ).rowcount
        database.close()
        if changed != 1:
            raise SystemExit(f"commit is not retryable: {commit}")
        print(f"precheck_retried={commit}")
    elif args.command == "uninstall":
        uninstall_system_service(root)
        stop_service(cache_root)
        hooks = run(
            ["git", "config", "--local", "--get", "core.hooksPath"],
            cwd=root,
            check=False,
            capture_output=True,
        ).stdout.strip()
        if Path(hooks).as_posix().rstrip("/") == ".githooks":
            subprocess.run(
                ["git", "config", "--local", "--unset", "core.hooksPath"],
                cwd=root,
                check=False,
            )
        subprocess.run(
            ["git", "config", "--local", "--unset", "udoncl.validationPython"],
            cwd=root,
            check=False,
        )
        print("precheck_service=uninstalled")


if __name__ == "__main__":
    main()
