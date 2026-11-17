import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import time


ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from precheck_service import (  # noqa: E402
    ValidationDatabase,
    coordinator,
    clone_dependency_prefixes,
    directory_fingerprint,
    dependency_lock_fingerprint,
    outgoing_commits,
    prepare_worktree,
    recover_orphan_workers,
    remove_owned,
    validation_fingerprint,
)


def run(command, cwd):
    return subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    ).stdout.strip()


with tempfile.TemporaryDirectory(prefix="udoncl precheck ") as temporary:
    temporary = Path(temporary)
    repository = temporary / "repository with spaces 日本語"
    cache = temporary / "state"
    repository.mkdir()
    dependency_seed = temporary / "dependency seed"
    (dependency_seed / "llvm23").mkdir(parents=True)
    (dependency_seed / "onnx-mlir").mkdir()
    (dependency_seed / "onnx-mlir-src").mkdir()
    (dependency_seed / "llvm23/tool.txt").write_text("llvm", encoding="utf-8")
    (dependency_seed / "onnx-mlir/tool.txt").write_text("onnx", encoding="utf-8")
    dependency_clone = temporary / "dependency clone"
    clone_dependency_prefixes(
        dependency_seed, dependency_clone, ("llvm23", "onnx-mlir")
    )
    assert (dependency_clone / "llvm23/tool.txt").read_text() == "llvm"
    assert (dependency_clone / "onnx-mlir/tool.txt").read_text() == "onnx"
    assert not (dependency_clone / "onnx-mlir-src").exists()
    read_only = temporary / "read-only tree"
    read_only.mkdir()
    read_only_file = read_only / "object.idx"
    read_only_file.write_text("object", encoding="utf-8")
    os.chmod(read_only_file, stat.S_IREAD)
    remove_owned(read_only, temporary)
    assert not read_only.exists()
    run(["git", "init"], repository)
    run(["git", "config", "user.email", "precheck@example.invalid"], repository)
    run(["git", "config", "user.name", "Precheck Test"], repository)
    run(["git", "config", "core.autocrlf", "true"], repository)
    (repository / "cmake").mkdir()
    (repository / "cmake/llvm-lock.json").write_text("{}\n", encoding="utf-8")
    (repository / "cmake/onnx-mlir-lock.json").write_text("{}\n", encoding="utf-8")
    commits = []
    for index in range(3):
        (repository / "value.txt").write_text(str(index), encoding="utf-8")
        run(["git", "add", "."], repository)
        run(["git", "commit", "-m", f"commit {index}"], repository)
        commits.append(run(["git", "rev-parse", "HEAD"], repository))

    lf_worktree = cache / "lf-worker" / "repo"
    prepare_worktree(repository, lf_worktree, commits[2], cache)
    assert (lf_worktree / "cmake/llvm-lock.json").read_bytes() == b"{}\n"

    database = ValidationDatabase(cache / "queue.sqlite3")
    for commit in commits[:2]:
        actual, inserted = database.enqueue(repository, commit)
        assert actual == commit and inserted
        duplicate, inserted = database.enqueue(repository, commit)
        assert duplicate == commit and not inserted
    database.close()

    previous_testing = os.environ.get("UDONCL_PRECHECK_TESTING")
    previous_delay = os.environ.get("UDONCL_PRECHECK_TEST_DELAY")
    previous_result = os.environ.get("UDONCL_PRECHECK_TEST_RESULT")
    os.environ["UDONCL_PRECHECK_TESTING"] = "1"
    os.environ["UDONCL_PRECHECK_TEST_DELAY"] = "1.5"
    os.environ["UDONCL_PRECHECK_TEST_RESULT"] = "passed"
    started = time.monotonic()
    coordinator(repository, cache, 2, True)
    elapsed = time.monotonic() - started
    assert elapsed < 2.3, elapsed

    database = ValidationDatabase(cache / "queue.sqlite3")
    rows = database.connection.execute(
        "SELECT commit_sha, state, cleanup_passed FROM jobs ORDER BY enqueued_at"
    ).fetchall()
    assert [(row["commit_sha"], row["state"]) for row in rows] == [
        (commits[0], "passed"),
        (commits[1], "passed"),
    ]
    assert all(row["cleanup_passed"] == 1 for row in rows)
    assert database.connection.execute(
        "SELECT count(*) FROM artifacts WHERE kind='precheck_log'"
    ).fetchone()[0] == 2

    database.enqueue(repository, commits[2])
    process_options = {}
    if os.name == "nt":
        process_options["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        process_options["start_new_session"] = True
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], **process_options
    )
    with database.immediate():
        database.connection.execute(
            """UPDATE jobs SET state='running', attempt=1, slot=0,
                 worker_token='orphan', worker_pid=?, heartbeat_at=?
               WHERE commit_sha=?""",
            (orphan.pid, "2099-01-01T00:00:00+00:00", commits[2]),
        )
    recover_orphan_workers(database)
    orphan.wait(timeout=10)
    recovered = database.connection.execute(
        "SELECT state, slot, worker_pid FROM jobs WHERE commit_sha=?", (commits[2],)
    ).fetchone()
    assert (recovered["state"], recovered["slot"], recovered["worker_pid"]) == (
        "queued",
        None,
        None,
    )
    database.close()
    os.environ["UDONCL_PRECHECK_TEST_DELAY"] = "0"
    os.environ["UDONCL_PRECHECK_TEST_RESULT"] = "failed"
    coordinator(repository, cache, 1, True)
    database = ValidationDatabase(cache / "queue.sqlite3")
    failed = database.connection.execute(
        "SELECT * FROM jobs WHERE commit_sha=?", (commits[2],)
    ).fetchone()
    assert failed["state"] == "failed"
    assert failed["exit_code"] == 1
    assert "test_gate=failed" in failed["error_summary"]

    (repository / ".precheck-baseline").write_text(
        commits[0] + "\n", encoding="utf-8"
    )
    run(["git", "add", ".precheck-baseline"], repository)
    run(["git", "commit", "-m", "set validation baseline"], repository)
    remote = commits[0]
    updates = [f"refs/heads/main {commits[2]} refs/heads/main {remote}"]
    assert outgoing_commits(repository, updates) == sorted(commits[1:])
    run(["git", "update-ref", "refs/remotes/origin/main", commits[0]], repository)
    run(["git", "update-ref", "refs/remotes/other/main", commits[1]], repository)
    new_branch = [f"refs/heads/topic {commits[2]} refs/heads/topic {'0' * 40}"]
    assert outgoing_commits(repository, new_branch, "origin") == sorted(commits[1:])

    window_repository = temporary / "push window repository"
    window_cache = temporary / "push window state"
    window_repository.mkdir()
    run(["git", "init"], window_repository)
    run(["git", "config", "user.email", "precheck@example.invalid"], window_repository)
    run(["git", "config", "user.name", "Precheck Test"], window_repository)
    (window_repository / "cmake").mkdir()
    (window_repository / "cmake/llvm-lock.json").write_text("{}\n", encoding="utf-8")
    (window_repository / "cmake/onnx-mlir-lock.json").write_text(
        "{}\n", encoding="utf-8"
    )
    window_commits = []
    for index in range(141):
        (window_repository / "value.txt").write_text(str(index), encoding="utf-8")
        run(["git", "add", "."], window_repository)
        run(["git", "commit", "-m", f"window commit {index}"], window_repository)
        window_commits.append(run(["git", "rev-parse", "HEAD"], window_repository))
    (window_repository / ".precheck-baseline").write_text(
        window_commits[116] + "\n", encoding="utf-8"
    )
    run(["git", "add", ".precheck-baseline"], window_repository)
    run(["git", "commit", "-m", "set validation baseline"], window_repository)
    window_update = [
        "refs/heads/main "
        f"{window_commits[-1]} refs/heads/main {window_commits[0]}"
    ]
    eligible = outgoing_commits(window_repository, window_update)
    assert eligible == sorted(window_commits[117:])
    assert len(eligible) == 24
    boundary_update = [
        "refs/heads/boundary "
        f"{window_commits[116]} refs/heads/boundary {window_commits[115]}"
    ]
    assert outgoing_commits(window_repository, boundary_update) == []
    two_ref_updates = window_update + [
        "refs/heads/topic "
        f"{window_commits[120]} refs/heads/topic {window_commits[0]}"
    ]
    assert outgoing_commits(window_repository, two_ref_updates) == sorted(
        window_commits[117:]
    )
    unrelated_tree = run(
        ["git", "rev-parse", f"{window_commits[0]}^{{tree}}"], window_repository
    )
    unrelated = run(
        ["git", "commit-tree", unrelated_tree, "-m", "unrelated"],
        window_repository,
    )
    try:
        outgoing_commits(
            window_repository,
            [f"refs/heads/unrelated {unrelated} refs/heads/unrelated {'0' * 40}"],
        )
        raise AssertionError("unrelated pushed tip was accepted")
    except ValueError as error:
        assert "is not a descendant of validation baseline" in str(error)

    run(
        ["git", "update-ref", "refs/remotes/origin/main", window_commits[0]],
        window_repository,
    )
    run(["git", "config", "branch.master.remote", "origin"], window_repository)
    run(
        ["git", "config", "branch.master.merge", "refs/heads/main"],
        window_repository,
    )
    window_database = ValidationDatabase(window_cache / "queue.sqlite3")
    old_failure, _ = window_database.enqueue(window_repository, window_commits[10])
    anchor_failure, _ = window_database.enqueue(
        window_repository, window_commits[116]
    )
    recent_failure, _ = window_database.enqueue(
        window_repository, window_commits[130]
    )
    with window_database.immediate():
        for commit in (old_failure, anchor_failure, recent_failure):
            window_database.connection.execute(
                """UPDATE jobs SET state='failed', attempt=1, exit_code=1,
                     error_summary='test failure', finished_at=?
                   WHERE commit_sha=?""",
                ("2026-01-07T00:00:00+00:00", commit),
            )
    window_database.close()
    filtered_failures = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts/precheck_status.py",
            "--repo",
            window_repository,
            "--cache-root",
            window_cache,
            "--outgoing",
            "--failures",
            "--unreported",
            "--json",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    assert filtered_failures.returncode == 1
    assert recent_failure in filtered_failures.stdout
    assert anchor_failure not in filtered_failures.stdout
    assert old_failure not in filtered_failures.stdout

    template = temporary / "unity template"
    for name in ("Assets", "Packages", "ProjectSettings"):
        (template / name).mkdir(parents=True)
    (template / "Assets/setting.txt").write_text("one", encoding="utf-8")
    dependencies = (
        cache / "dependencies" / dependency_lock_fingerprint(repository, commits[2])
    )
    (dependencies / "llvm23").mkdir(parents=True)
    (dependencies / "onnx-mlir").mkdir()
    (dependencies / "llvm23/.udoncl-sdk.json").write_text("{}", encoding="utf-8")
    (dependencies / "onnx-mlir/.udoncl-onnx-mlir.json").write_text(
        "{}", encoding="utf-8"
    )
    unity = temporary / "Unity/Hub/Editor/2022.3.22f1/Editor/Unity.exe"
    first_template = directory_fingerprint(
        template, ("Assets", "Packages", "ProjectSettings")
    )
    first_fingerprint = validation_fingerprint(
        repository, cache, commits[2], unity, first_template
    )
    (template / "Assets/setting.txt").write_text("two", encoding="utf-8")
    second_template = directory_fingerprint(
        template, ("Assets", "Packages", "ProjectSettings")
    )
    assert first_template != second_template
    assert first_fingerprint != validation_fingerprint(
        repository, cache, commits[2], unity, second_template
    )
    database.close()

    if previous_testing is None:
        os.environ.pop("UDONCL_PRECHECK_TESTING", None)
    else:
        os.environ["UDONCL_PRECHECK_TESTING"] = previous_testing
    if previous_delay is None:
        os.environ.pop("UDONCL_PRECHECK_TEST_DELAY", None)
    else:
        os.environ["UDONCL_PRECHECK_TEST_DELAY"] = previous_delay
    if previous_result is None:
        os.environ.pop("UDONCL_PRECHECK_TEST_RESULT", None)
    else:
        os.environ["UDONCL_PRECHECK_TEST_RESULT"] = previous_result

print("precheck_service=passed")
