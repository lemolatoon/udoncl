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
        if os.name == "nt":
            service.remove_owned(transient, temporary)
        else:
            # Only Windows sharing violations are retried.
            try:
                service.remove_owned(transient, temporary)
            except PermissionError:
                pass
            else:
                raise AssertionError("a non-Windows removal error was retried")
    finally:
        service.shutil.rmtree = original_rmtree
        service.time.sleep = original_sleep
    if os.name == "nt":
        assert attempts == 2
        assert not transient.exists()
    else:
        assert attempts == 1
        service.remove_owned(transient, temporary)
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

    boundary_pid_path = temporary / "boundary-child.pid"
    boundary_parent = temporary / "boundary-parent.py"
    boundary_parent.write_text(
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(120)'])\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n",
        encoding="utf-8",
    )
    boundary_log = temporary / "snapshot-precommit.log"
    with boundary_log.open("w", encoding="utf-8") as log:
        contained = service.ContainedProcess(
            [sys.executable, boundary_parent, boundary_pid_path],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 10
        while not boundary_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert boundary_pid_path.is_file()
        boundary_child_pid = int(boundary_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 10
        while contained.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert contained.poll() is not None
        assert service.process_alive(boundary_child_pid)
        contained.close()
    deadline = time.monotonic() + 10
    while service.process_alive(boundary_child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not service.process_alive(boundary_child_pid)
    service.remove_owned(boundary_log, temporary)

    singleton_cache = temporary / "singleton-cache"
    singleton_cache.mkdir()
    singleton_database = service.ValidationDatabase(
        singleton_cache / "queue.sqlite3"
    )
    singleton_database.enqueue(repository, commit)
    with singleton_database.immediate():
        singleton_database.connection.execute(
            """UPDATE jobs SET state='running', attempt=1,
                 worker_token='first-coordinator', heartbeat_at=?
               WHERE commit_sha=?""",
            (service.now(), commit),
        )
    singleton_database.close()
    lock_script = temporary / "hold-coordinator-lock.py"
    lock_script.write_text(
        "import pathlib, sys, time\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "from precheck_service import CoordinatorLock\n"
        "with CoordinatorLock(pathlib.Path(sys.argv[1])):\n"
        " print('locked', flush=True)\n"
        " time.sleep(120)\n",
        encoding="utf-8",
    )
    lock_holder = subprocess.Popen(
        [sys.executable, lock_script, singleton_cache / "coordinator.lock"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert lock_holder.stdout.readline().strip() == "locked"
        try:
            service.coordinator(repository, singleton_cache, 1, True)
        except RuntimeError as error:
            assert "already owns" in str(error)
        else:
            raise AssertionError("second coordinator unexpectedly acquired the lock")
        singleton_database = service.ValidationDatabase(
            singleton_cache / "queue.sqlite3"
        )
        singleton_row = singleton_database.connection.execute(
            "SELECT state, worker_token FROM jobs WHERE commit_sha=?", (commit,)
        ).fetchone()
        assert (singleton_row["state"], singleton_row["worker_token"]) == (
            "running",
            "first-coordinator",
        )
        singleton_database.close()
    finally:
        lock_holder.terminate()
        lock_holder.wait(timeout=10)

    stale_live_cache = temporary / "stale-live-cache"
    stale_live_database = service.ValidationDatabase(
        stale_live_cache / "queue.sqlite3"
    )
    stale_live_database.acquire_service("first")
    with stale_live_database.immediate():
        stale_live_database.connection.execute(
            "UPDATE service SET heartbeat_at=? WHERE singleton=1",
            ("2000-01-01T00:00:00+00:00",),
        )
    try:
        stale_live_database.acquire_service("second")
    except RuntimeError as error:
        assert f"pid {os.getpid()}" in str(error)
    else:
        raise AssertionError("a stale heartbeat replaced a live coordinator")
    stale_live_database.release_service("first")
    stale_live_database.close()

    supervised_cache = temporary / "supervised-cache"
    supervised_cache.mkdir()
    (supervised_cache / "service-enabled").write_text(
        "enabled\n", encoding="utf-8"
    )
    original_wait_for_service = service.wait_for_service
    original_popen = service.subprocess.Popen
    waited = []

    def record_supervisor_wait(cache_root, timeout=15):
        waited.append((cache_root, timeout))

    def reject_direct_spawn(*_args, **_kwargs):
        raise AssertionError("manual start spawned beside the installed supervisor")

    # Only the Windows startup fallback installs a supervisor that owns
    # coordinator startup through the service-enabled marker.
    if os.name == "nt":
        service.wait_for_service = record_supervisor_wait
        service.subprocess.Popen = reject_direct_spawn
        try:
            service.start_background(repository, supervised_cache, 2)
        finally:
            service.wait_for_service = original_wait_for_service
            service.subprocess.Popen = original_popen
        assert waited == [(supervised_cache, 15)]

    ownership_cache = temporary / "ownership-cache"
    ownership_process = subprocess.Popen(
        [
            sys.executable,
            ROOT / "scripts/precheck_service.py",
            "run",
            "--repo",
            repository,
            "--cache-root",
            ownership_cache,
            "--jobs",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ownership_database = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ownership_database = service.ValidationDatabase(
                ownership_cache / "queue.sqlite3"
            )
            row = ownership_database.service_row()
            if row and row["pid"] == ownership_process.pid:
                break
            ownership_database.close()
            ownership_database = None
            time.sleep(0.05)
        assert ownership_database is not None
        with ownership_database.immediate():
            ownership_database.connection.execute(
                "UPDATE service SET token='replacement' WHERE singleton=1"
            )
        ownership_database.close()
        ownership_database = None
        ownership_process.wait(timeout=10)
        assert ownership_process.returncode != 0
        ownership_database = service.ValidationDatabase(
            ownership_cache / "queue.sqlite3"
        )
        assert ownership_database.service_row()["token"] == "replacement"
    finally:
        if ownership_database is not None:
            ownership_database.close()
        if ownership_process.poll() is None:
            ownership_process.terminate()
            ownership_process.wait(timeout=10)

print("precheck_cleanup=passed")
