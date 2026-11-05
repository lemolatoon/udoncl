"""Execute the common Host Async contract through LLVM coroutines."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile


def run(command: list[str | Path], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        [str(item) for item in command],
        input=input_text,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result.stdout


udoncl_opt = Path(sys.argv[1])
mlir_opt = Path(sys.argv[2])
mlir_translate = Path(sys.argv[3])
llvm_opt = Path(sys.argv[4])
lli = Path(sys.argv[5])
async_runtime = Path(sys.argv[6])
source = Path(sys.argv[7])
support = Path(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] else None

common_async = run(
    [
        udoncl_opt,
        "--vrc-lower-host-to-async",
        "--vrc-lower-host-async-to-native",
        source,
    ]
)
for fragment in [
    "func.func @add",
    "async.func private @wait_and_add",
    "async.func @run",
    "async.func @run_sequential",
    "!async.value<i32>",
]:
    assert fragment in common_async, (fragment, common_async)
assert "!vrc.event" not in common_async, common_async
assert "vrc.event_as_async" not in common_async, common_async

harness = r"""
  func.func @main() -> i32 {
    %event = "async.runtime.create"() : () -> !async.value<i32>
    %value = arith.constant 35 : i32
    %status = arith.constant 7 : i32
    %result = "async.call"(%event, %value) <{callee = @run}>
        : (!async.value<i32>, i32) -> !async.value<i32>
    "async.runtime.store"(%status, %event)
        : (i32, !async.value<i32>) -> ()
    "async.runtime.set_available"(%event) : (!async.value<i32>) -> ()
    %answer = async.await %result : !async.value<i32>
    %expected = arith.constant 42 : i32
    %success = arith.cmpi eq, %answer, %expected : i32
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %exit = arith.select %success, %zero, %one : i32

    // Force two distinct suspension/resumption cycles. The function completes
    // %between only after resuming from the first event, so the caller cannot
    // complete the second event before the second await is reached.
    %first_event = "async.runtime.create"() : () -> !async.value<i32>
    %second_event = "async.runtime.create"() : () -> !async.value<i32>
    %between = "async.runtime.create"() : () -> !async.value<i32>
    %sequential = "async.call"(%first_event, %second_event, %between, %value)
        <{callee = @run_sequential}>
        : (!async.value<i32>, !async.value<i32>, !async.value<i32>, i32)
          -> !async.value<i32>
    %first_status = arith.constant 3 : i32
    "async.runtime.store"(%first_status, %first_event)
        : (i32, !async.value<i32>) -> ()
    "async.runtime.set_available"(%first_event) : (!async.value<i32>) -> ()
    %observed_first = async.await %between : !async.value<i32>
    %second_status = arith.constant 4 : i32
    "async.runtime.store"(%second_status, %second_event)
        : (i32, !async.value<i32>) -> ()
    "async.runtime.set_available"(%second_event) : (!async.value<i32>) -> ()
    %sequential_answer = async.await %sequential : !async.value<i32>
    %sequential_expected = arith.constant 42 : i32
    %first_ok = arith.cmpi eq, %observed_first, %first_status : i32
    %sequential_ok = arith.cmpi eq, %sequential_answer, %sequential_expected : i32

    // A value completed before the call must take the ready fast path without
    // changing the same Host API semantics.
    %ready_event = "async.runtime.create"() : () -> !async.value<i32>
    %ready_status = arith.constant 7 : i32
    "async.runtime.store"(%ready_status, %ready_event)
        : (i32, !async.value<i32>) -> ()
    "async.runtime.set_available"(%ready_event) : (!async.value<i32>) -> ()
    %ready_result = "async.call"(%ready_event, %value) <{callee = @run}>
        : (!async.value<i32>, i32) -> !async.value<i32>
    %ready_answer = async.await %ready_result : !async.value<i32>
    %ready_ok = arith.cmpi eq, %ready_answer, %expected : i32

    // Non-suspending Host functions remain ordinary func calls.
    %plain_answer = func.call @add(%value, %ready_status) : (i32, i32) -> i32
    %plain_ok = arith.cmpi eq, %plain_answer, %expected : i32

    // OpenCL/runtime failures are ordinary negative cl_int payloads, not
    // infrastructure-level Async errors.
    %error_event = "async.runtime.create"() : () -> !async.value<i32>
    %zero_value = arith.constant 0 : i32
    %error_result = "async.call"(%error_event, %zero_value) <{callee = @run}>
        : (!async.value<i32>, i32) -> !async.value<i32>
    %runtime_error = arith.constant -30 : i32
    "async.runtime.store"(%runtime_error, %error_event)
        : (i32, !async.value<i32>) -> ()
    "async.runtime.set_available"(%error_event) : (!async.value<i32>) -> ()
    %observed_error = async.await %error_result : !async.value<i32>
    %error_ok = arith.cmpi eq, %observed_error, %runtime_error : i32

    %all_ok = arith.andi %success, %first_ok : i1
    %suspends_ok = arith.andi %all_ok, %sequential_ok : i1
    %ready_paths_ok = arith.andi %suspends_ok, %ready_ok : i1
    %ordinary_ok = arith.andi %ready_paths_ok, %plain_ok : i1
    %really_all_ok = arith.andi %ordinary_ok, %error_ok : i1
    %final_exit = arith.select %really_all_ok, %zero, %one : i32
    return %final_exit : i32
  }
"""
module_end = common_async.rfind("}")
assert module_end >= 0, common_async
executable_async = common_async[:module_end] + harness + common_async[module_end:]

lowered = run(
    [
        mlir_opt,
        "--async-func-to-async-runtime",
        "--async-to-async-runtime",
        "--async-runtime-ref-counting",
        "--convert-async-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts",
    ],
    input_text=executable_async,
)
for fragment in [
    "llvm.intr.coro.begin",
    "llvm.intr.coro.suspend",
    "mlirAsyncRuntimeAwaitValueAndExecute",
    "mlirAsyncRuntimeEmplaceValue",
]:
    assert fragment in lowered, (fragment, lowered)

llvm_ir = run([mlir_translate, "--mlir-to-llvmir"], input_text=lowered)
coroutines = run(
    [
        llvm_opt,
        "-S",
        "-passes=coro-early,cgscc(coro-split),coro-cleanup",
    ],
    input_text=llvm_ir,
)
assert "wait_and_add.resume" in coroutines, coroutines
assert "call i8 @llvm.coro.suspend" not in coroutines, coroutines

with tempfile.TemporaryDirectory(prefix="udoncl-native-async-") as temporary:
    module = Path(temporary) / "native-async.ll"
    module.write_text(coroutines, encoding="utf-8")
    command = [lli, f"--dlopen={async_runtime}"]
    if support is not None:
        command.append(f"--dlopen={support}")
    command.append(module)
    result = subprocess.run(
        [str(item) for item in command], text=True, capture_output=True
    )
assert result.returncode == 0, (command, result.stdout, result.stderr)
