import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

import precheck_service as service  # noqa: E402
from precommit import McpClient  # noqa: E402


with tempfile.TemporaryDirectory(prefix="udoncl precheck cleanup ") as temporary:
    temporary = Path(temporary)

    transient = temporary / "transient"
    transient.mkdir()
    (transient / "locked.log").write_text("locked", encoding="utf-8")
    original_rmtree = service.shutil.rmtree
    original_sleep = service.time.sleep
    attempts = 0

    def transient_rmtree(*args, **kwargs):
        global attempts
        attempts += 1
        if attempts == 1:
            error = PermissionError("transient sharing violation")
            error.winerror = 32
            raise error
        return original_rmtree(*args, **kwargs)

    service.shutil.rmtree = transient_rmtree
    service.time.sleep = lambda _delay: None
    try:
        service.remove_owned(transient, temporary)
    finally:
        service.shutil.rmtree = original_rmtree
        service.time.sleep = original_sleep
    assert attempts == 2
    assert not transient.exists()

    repository = temporary / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "precheck@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Precheck Test"],
        cwd=repository,
        check=True,
    )
    (repository / "value.txt").write_text("value", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repository, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    cache = temporary / "cache"
    database = service.ValidationDatabase(cache / "queue.sqlite3")
    database.enqueue(repository, commit)
    marker = service.worker_slot_contamination_marker(cache, 3)
    marker.parent.mkdir(parents=True)
    marker.write_text("contaminated\n", encoding="utf-8")
    original_remove_owned = service.remove_owned

    def persistently_locked(_path, _parent):
        raise PermissionError("persistent sharing violation")

    service.remove_owned = persistently_locked
    try:
        claimed, retry_after = service.claim_ready_job(
            database, repository, cache, 3, "blocked"
        )
        assert claimed is None and retry_after > time.monotonic()
        assert marker.is_file()
        state = database.connection.execute(
            "SELECT state FROM jobs WHERE commit_sha=?", (commit,)
        ).fetchone()["state"]
        assert state == "queued"
    finally:
        service.remove_owned = original_remove_owned
    claimed, retry_after = service.claim_ready_job(
        database, repository, cache, 3, "recovered"
    )
    assert claimed == commit and retry_after == 0
    assert not service.worker_slot_root(cache, 3).exists()
    database.close()

    child_pid_path = temporary / "child.pid"
    parent_script = temporary / "mcp-parent.py"
    parent_script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(120)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    stderr_path = temporary / "mcp-server.log"
    client = McpClient(
        [sys.executable, parent_script, child_pid_path],
        dict(os.environ),
        stderr_path,
    )
    deadline = time.monotonic() + 10
    while not child_pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert service.process_alive(child_pid)
    deadline = time.monotonic() + 10
    while client._process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client._process.poll() is not None
    client.close()
    deadline = time.monotonic() + 10
    while service.process_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not service.process_alive(child_pid)
    stderr_path.unlink()

print("precheck_cleanup=passed")
