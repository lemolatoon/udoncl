"""Run the tiny LLaMA before and after the native UdonHL adapter."""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys


VOCAB, DIM, HIDDEN, HEADS, CONTEXT = 16, 8, 16, 2, 4


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_weights(seed):
    state = seed

    def sample():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        signed = ((state >> 16) & 65535) - 32768
        return f32(signed * (0.125 / 32768.0))

    def array(count):
        return [sample() for _ in range(count)]

    weights = {
        "embedding": array(VOCAB * DIM),
        "wq": array(DIM * DIM),
        "wk": array(DIM * DIM),
        "wv": array(DIM * DIM),
        "wo": array(DIM * DIM),
        "w1": array(HIDDEN * DIM),
        "w2": array(DIM * HIDDEN),
        "w3": array(HIDDEN * DIM),
        "output": array(VOCAB * DIM),
        "attention_norm": [],
        "ffn_norm": [],
        "final_norm": [],
    }
    for _ in range(DIM):
        weights["attention_norm"].append(f32(0.875 + sample()))
        weights["ffn_norm"].append(f32(0.875 + sample()))
        weights["final_norm"].append(f32(0.875 + sample()))
    return weights


def matvec(matrix, vector, rows, columns):
    result = []
    for row in range(rows):
        total = 0.0
        for column in range(columns):
            total = f32(total + f32(matrix[row * columns + column] * vector[column]))
        result.append(total)
    return result


def rms_norm(vector, weight):
    square_sum = 0.0
    for value in vector:
        square_sum = f32(square_sum + f32(value * value))
    scale = f32(1.0 / math.sqrt(f32(square_sum / DIM + 1.0e-5)))
    return [f32(f32(value * scale) * weight[index]) for index, value in enumerate(vector)]


def rope(vector, position):
    result = list(vector)
    head_dimension = DIM // HEADS
    for head in range(HEADS):
        for pair in range(head_dimension // 2):
            angle = f32(position * (1.0 if pair == 0 else 0.01))
            cosine, sine = f32(math.cos(angle)), f32(math.sin(angle))
            index = head * head_dimension + pair * 2
            first, second = result[index], result[index + 1]
            result[index] = f32(f32(first * cosine) - f32(second * sine))
            result[index + 1] = f32(f32(first * sine) + f32(second * cosine))
    return result


def forward(weights, tokens):
    tokens = tokens[-CONTEXT:] or [0]
    state, query, key, value = [], [], [], []
    for position, token in enumerate(tokens):
        current = list(weights["embedding"][(token & 15) * DIM : ((token & 15) + 1) * DIM])
        normalized = rms_norm(current, weights["attention_norm"])
        state.append(current)
        query.append(rope(matvec(weights["wq"], normalized, DIM, DIM), position))
        key.append(rope(matvec(weights["wk"], normalized, DIM, DIM), position))
        value.append(matvec(weights["wv"], normalized, DIM, DIM))
    attention = [[0.0] * DIM for _ in tokens]
    head_dimension = DIM // HEADS
    for position in range(len(tokens)):
        for head in range(HEADS):
            scores = []
            for previous in range(position + 1):
                score = 0.0
                for index in range(head_dimension):
                    lane = head * head_dimension + index
                    score = f32(score + f32(query[position][lane] * key[previous][lane]))
                scores.append(f32(score * 0.5))
            maximum = max(scores)
            exponentials = [f32(math.exp(f32(score - maximum))) for score in scores]
            denominator = 0.0
            for exponential in exponentials:
                denominator = f32(denominator + exponential)
            for index in range(head_dimension):
                lane = head * head_dimension + index
                total = 0.0
                for previous, exponential in enumerate(exponentials):
                    total = f32(total + f32(exponential * value[previous][lane]))
                attention[position][lane] = f32(total / denominator)
    for position in range(len(tokens)):
        projected = matvec(weights["wo"], attention[position], DIM, DIM)
        state[position] = [f32(a + b) for a, b in zip(state[position], projected)]
        normalized = rms_norm(state[position], weights["ffn_norm"])
        gate = matvec(weights["w1"], normalized, HIDDEN, DIM)
        up = matvec(weights["w3"], normalized, HIDDEN, DIM)
        activated = []
        for gate_value, up_value in zip(gate, up):
            silu = f32(gate_value / f32(1.0 + f32(math.exp(-gate_value))))
            activated.append(f32(silu * up_value))
        down = matvec(weights["w2"], activated, DIM, HIDDEN)
        state[position] = [f32(a + b) for a, b in zip(state[position], down)]
    final_state = rms_norm(state[-1], weights["final_norm"])
    return matvec(weights["output"], final_state, VOCAB, DIM)


def generate(weights, prompt, count):
    context = list(prompt[-CONTEXT:])
    output = []
    for _ in range(count):
        logits = forward(weights, context)
        token = max(range(VOCAB), key=lambda index: logits[index])
        output.append(token)
        context = (context + [token])[-CONTEXT:]
    return output


def run(command, *, capture=False):
    result = subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=capture,
        timeout=180,
    )
    if result.returncode:
        if capture:
            sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout if capture else ""


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=root / "build/udoncl")
    parser.add_argument("--output", type=Path, default=root / "build/tiny-llama")
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    object_suffix = ".obj" if sys.platform == "win32" else ".o"
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    llvm_dir = next(
        line.split("=", 1)[1]
        for line in cache.splitlines()
        if line.startswith("LLVM_DIR:")
    )
    sdk = Path(llvm_dir).parents[2]
    clang = sdk / "bin" / ("clang" + suffix)
    translate = build / "tools/vrc-translate" / ("vrc-translate" + suffix)
    opt = build / "tools/vrc-opt" / ("vrc-opt" + suffix)
    mlir_translate = sdk / "bin" / ("mlir-translate" + suffix)
    runtime_flags = ["-fms-runtime-lib=dll"] if sys.platform == "win32" else []
    math_library = [] if sys.platform == "win32" else ["-lm"]

    llvm = output / "tiny-llama.ll"
    imported = output / "tiny-llama.imported.mlir"
    udonhl = output / "tiny-llama.udonhl.mlir"
    lowered = output / "tiny-llama.lowered.mlir"
    lowered_ir = output / "tiny-llama.lowered.ll"
    lowered_object = output / ("tiny-llama.lowered" + object_suffix)
    original = output / ("tiny-llama-original" + suffix)
    transformed = output / ("tiny-llama-lowered" + suffix)
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            "-O0",
            "-Xclang",
            "-disable-O0-optnone",
            "-fomit-frame-pointer",
            "-ffp-contract=off",
            "-S",
            "-emit-llvm",
            source / "tiny_llama.c",
            "-o",
            llvm,
        ]
    )
    run([translate, "--import-llvm", llvm, "-o", imported])
    run([opt, "--vrc-raise-native-host-calls", "--verify-each", imported, "-o", udonhl])
    text = udonhl.read_text(encoding="utf-8")
    for symbol in ["tiny_llama_forward", "sqrtf", "cosf", "sinf", "expf"]:
        if f"udonhl.call @{symbol}" not in text:
            raise RuntimeError(f"tiny LLaMA call graph lost {symbol}")
    run([opt, "--vrc-lower-native-host-calls", "--verify-each", udonhl, "-o", lowered])
    run([mlir_translate, "--mlir-to-llvmir", lowered, "-o", lowered_ir])
    run([clang, *runtime_flags, "-c", lowered_ir, "-o", lowered_object])
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            source / "main.c",
            source / "tiny_llama.c",
            *math_library,
            "-o",
            original,
        ]
    )
    run(
        [
            clang,
            *runtime_flags,
            "-std=c11",
            source / "main.c",
            lowered_object,
            *math_library,
            "-o",
            transformed,
        ]
    )
    original_text = run([original], capture=True)
    transformed_text = run([transformed], capture=True)
    if original_text != transformed_text:
        raise RuntimeError("native and UdonHL-adapted tiny LLaMA output differ")
    records = dict(line.split("=", 1) for line in transformed_text.splitlines())
    actual_logits = [float(value) for value in records["logits"].split(",")]
    actual_generated = [int(value) for value in records["generated"].split(",")]
    weights = make_weights(0x4C4C414D)
    expected_logits = forward(weights, [1, 5, 9, 3])
    expected_generated = generate(weights, [1, 5, 9, 3], 8)
    maximum_error = max(abs(a - b) for a, b in zip(actual_logits, expected_logits))
    if maximum_error > 2.0e-5 or actual_generated != expected_generated:
        raise RuntimeError(
            f"tiny LLaMA differs from oracle: error={maximum_error}, "
            f"tokens={actual_generated}/{expected_generated}"
        )
    print(transformed_text, end="")
    print(f"python_oracle_max_error={maximum_error:.9g}")
    print("native_udonhl_roundtrip=passed")


if __name__ == "__main__":
    main()
