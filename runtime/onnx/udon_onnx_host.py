"""Generate execution-plan-free UdonCL Host C directly from an ONNX model."""

import argparse
import math
from pathlib import Path
import struct

from udon_onnx import parse_model


def _c_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("ONNX initializer contains a non-finite float")
    return repr(float(value)) + "f"


def _float_initializer(tensor: dict) -> list[float]:
    if tensor["data_type_name"] != "float32" or tensor["data_location"] == 1:
        raise ValueError("MNIST Dense requires embedded float32 initializers")
    return list(struct.unpack(f"<{len(tensor['raw']) // 4}f", tensor["raw"]))


def _array(lines: list[str], declaration: str, values: list[float]) -> None:
    lines.append(declaration + " = {")
    for offset in range(0, len(values), 6):
        lines.append("    " + ", ".join(_c_float(value) for value in values[offset : offset + 6]) + ",")
    lines.append("};")


def _mlir_i8_array(data: bytes) -> str:
    signed = (value if value < 128 else value - 256 for value in data)
    return "array<i8: " + ", ".join(str(value) for value in signed) + ">"


def _udonhl_global(name: str, storage_type: str, data: bytes) -> str:
    return (
        '  "udonhl.global"() <{address_space = 0 : i32, alignment = 4 : i64, '
        f"initial_value = {_mlir_i8_array(data)}, linkage = \"internal\", "
        f"storage_type = {storage_type}, sym_name = \"{name}\"}}> : () -> ()"
    )


def _float_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _checked_call(
    lines: list[str],
    index: int,
    callee: str,
    arguments: str,
    signature: str,
) -> None:
    lines.extend(
        [
            f"  ^step{index}:",
            f"    %status{index} = func.call @{callee}({arguments}) : {signature}",
            f"    %ok{index} = arith.cmpi eq, %status{index}, %success : i32",
            f"    cf.cond_br %ok{index}, ^step{index + 1}, "
            f"^cleanup(%status{index} : i32)",
        ]
    )


def emit_mnist_mlir(model: dict) -> str:
    """Lower full MNIST to public UdonDNN and UdonBLAS calls."""
    nodes = model["graph"]["nodes"]
    if [node["op_type"] for node in nodes] != [
        "Conv",
        "Add",
        "Relu",
        "AveragePool",
        "Flatten",
        "Gemm",
        "Add",
    ]:
        raise ValueError("MNIST MLIR requires Conv/Add/Relu/Pool/Flatten/Gemm/Add")
    initializers = model["graph"]["initializers"]
    expected = {
        "conv.weight": [16, 1, 5, 5],
        "conv.bias": [1, 16, 1, 1],
        "dense.weight": [144, 10],
        "dense.bias": [10],
    }
    for name, shape in expected.items():
        if name not in initializers or initializers[name]["shape"] != shape:
            raise ValueError(f"MNIST MLIR initializer {name!r} must have shape {shape}")
    conv_weight = _float_initializer(initializers["conv.weight"])
    conv_bias = _float_initializer(initializers["conv.bias"])
    dense_weight = _float_initializer(initializers["dense.weight"])
    dense_bias = _float_initializer(initializers["dense.bias"])
    packed_dense = [
        dense_weight[feature * 10 + column]
        for column in range(10)
        for feature in range(144)
    ]

    lines = [
        'module attributes {udoncl.onnx_lowering = "library-calls-v1", '
        'udoncl.onnx_profile = "mnist"} {',
        _udonhl_global("mnist_input", "!llvm.array<784 x f32>", bytes(784 * 4)),
        _udonhl_global(
            "mnist_conv_weight", "!llvm.array<400 x f32>", _float_bytes(conv_weight)
        ),
        _udonhl_global(
            "mnist_conv_bias", "!llvm.array<16 x f32>", _float_bytes(conv_bias)
        ),
        _udonhl_global("mnist_conv_output", "!llvm.array<9216 x f32>", bytes(9216 * 4)),
        _udonhl_global("mnist_pool_output", "!llvm.array<144 x f32>", bytes(144 * 4)),
        _udonhl_global(
            "mnist_dense_weight", "!llvm.array<1440 x f32>", _float_bytes(packed_dense)
        ),
        _udonhl_global("mnist_logits", "!llvm.array<10 x f32>", bytes(10 * 4)),
        "",
        '  func.func private @udondnnCreate(!udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.api = "udondnnCreate", udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroy(i32) -> i32 attributes '
        '{udoncl.api = "udondnnDestroy", udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnCreateTensorDescriptor(!udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnSetTensor4dDescriptor(i32, i32, i32, i32, i32, '
        'i32, i32) -> i32 attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroyTensorDescriptor(i32) -> i32 attributes '
        '{udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnCreateFilterDescriptor(!udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnSetFilter4dDescriptor(i32, i32, i32, i32, i32, '
        'i32, i32) -> i32 attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroyFilterDescriptor(i32) -> i32 attributes '
        '{udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnCreateConvolutionDescriptor(!udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnSetConvolution2dDescriptor(i32, i32, i32, i32, '
        'i32, i32, i32, i32, i32) -> i32 attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroyConvolutionDescriptor(i32) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnCreateOpTensorDescriptor(!udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnSetOpTensorDescriptor(i32, i32, i32, i32) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroyOpTensorDescriptor(i32) -> i32 attributes '
        '{udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnCreateActivationDescriptor(!udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnSetActivationDescriptor(i32, i32, i32, f32) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroyActivationDescriptor(i32) -> i32 attributes '
        '{udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnCreatePoolingDescriptor(!udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnSetPooling2dDescriptor(i32, i32, i32, i32, i32, '
        'i32, i32, i32, i32) -> i32 attributes {udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnDestroyPoolingDescriptor(i32) -> i32 attributes '
        '{udoncl.library = "UdonDNN"}',
        '  func.func private @udondnnConvolutionForward(i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.library = "UdonDNN", vrc.async_effect = "may_suspend"}',
        '  func.func private @udondnnOpTensor(i32, i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, '
        'i32, !udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.library = "UdonDNN", vrc.async_effect = "may_suspend"}',
        '  func.func private @udondnnActivationForward(i32, i32, !udonhl.ptr<0>, '
        'i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32 '
        'attributes {udoncl.library = "UdonDNN", vrc.async_effect = "may_suspend"}',
        '  func.func private @udondnnPoolingForward(i32, i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.library = "UdonDNN", vrc.async_effect = "may_suspend"}',
        '  func.func private @udonblasCreate(!udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.library = "UdonBLAS"}',
        '  func.func private @udonblasDestroy(i32) -> i32 attributes '
        '{udoncl.library = "UdonBLAS"}',
        '  func.func private @udonblasSgemm(i32, i32, i32, i32, i32, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32 attributes '
        '{udoncl.library = "UdonBLAS", vrc.async_effect = "may_suspend"}',
        "",
        '  func.func @write_mnist_input(%index: i32, %value: f32) -> i32 attributes {',
        '    vrc.export_id = 1 : i32, vrc.parameter_names = ["index", "value"],',
        '    vrc.public_name = "WriteMnistInput"',
        '  } {',
        '    %four = arith.constant 4 : i32',
        '    %base = udonhl.address_of @mnist_input : <0>',
        '    %offset = arith.muli %index, %four : i32',
        '    %position = "udonhl.ptr_add"(%base, %offset) : '
        '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
        '    "udonhl.store"(%value, %position) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    %success = arith.constant 0 : i32',
        '    return %success : i32',
        '  }',
        "",
        '  func.func @run_mnist() -> i32 attributes {',
        '    vrc.async_effect = "may_suspend", vrc.export_id = 2 : i32,',
        '    vrc.public_name = "RunMnist"',
        '  } {',
        '    %success = arith.constant 0 : i32',
        '    %zero = arith.constant 0 : i32',
        '    %one = arith.constant 1 : i32',
        '    %four = arith.constant 4 : i32',
        '    %fzero = arith.constant 0.0 : f32',
        '    %fone = arith.constant 1.0 : f32',
        '    %null = "udonhl.inttoptr"(%zero) : (i32) -> !udonhl.ptr<0>',
        '    %alpha = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %beta0 = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %beta1 = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    "udonhl.store"(%fone, %alpha) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    "udonhl.store"(%fzero, %beta0) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    "udonhl.store"(%fone, %beta1) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
    ]
    slot_names = [
        "dnn_handle",
        "blas_handle",
        "input_desc",
        "conv_desc",
        "bias_desc",
        "pool_desc",
        "filter_desc",
        "convolution_desc",
        "op_desc",
        "activation_desc",
        "pooling_desc",
    ]
    for name in slot_names:
        lines.extend(
            [
                f'    %{name}_slot = "udonhl.alloca"(%four) <{{alignment = 4 : i64}}> : '
                '(i32) -> !udonhl.ptr<0>',
                f'    "udonhl.store"(%zero, %{name}_slot) <{{alignment = 4 : i64}}> : '
                '(i32, !udonhl.ptr<0>) -> ()',
            ]
        )
    lines.extend(
        [
            '    %image = udonhl.address_of @mnist_input : <0>',
            '    %conv_weight = udonhl.address_of @mnist_conv_weight : <0>',
            '    %conv_bias = udonhl.address_of @mnist_conv_bias : <0>',
            '    %conv_output = udonhl.address_of @mnist_conv_output : <0>',
            '    %pool_output = udonhl.address_of @mnist_pool_output : <0>',
            '    %dense_weight = udonhl.address_of @mnist_dense_weight : <0>',
            '    %logits = udonhl.address_of @mnist_logits : <0>',
        ]
    )
    for digit, value in enumerate(dense_bias):
        lines.extend(
            [
                f'    %logit_offset{digit} = arith.constant {digit * 4} : i32',
                f'    %logit{digit} = "udonhl.ptr_add"(%logits, %logit_offset{digit}) : '
                '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
                f'    %dense_bias{digit} = arith.constant {_c_float(value)[:-1]} : f32',
                f'    "udonhl.store"(%dense_bias{digit}, %logit{digit}) '
                '<{alignment = 4 : i64}> : (f32, !udonhl.ptr<0>) -> ()',
            ]
        )
    lines.append('    cf.br ^step0')

    calls = [
        ("udondnnCreate", "%dnn_handle_slot", "(!udonhl.ptr<0>) -> i32"),
        ("udonblasCreate", "%blas_handle_slot", "(!udonhl.ptr<0>) -> i32"),
        ("udondnnCreateTensorDescriptor", "%input_desc_slot", "(!udonhl.ptr<0>) -> i32"),
        ("udondnnCreateTensorDescriptor", "%conv_desc_slot", "(!udonhl.ptr<0>) -> i32"),
        ("udondnnCreateTensorDescriptor", "%bias_desc_slot", "(!udonhl.ptr<0>) -> i32"),
        ("udondnnCreateTensorDescriptor", "%pool_desc_slot", "(!udonhl.ptr<0>) -> i32"),
        ("udondnnCreateFilterDescriptor", "%filter_desc_slot", "(!udonhl.ptr<0>) -> i32"),
        (
            "udondnnCreateConvolutionDescriptor",
            "%convolution_desc_slot",
            "(!udonhl.ptr<0>) -> i32",
        ),
        ("udondnnCreateOpTensorDescriptor", "%op_desc_slot", "(!udonhl.ptr<0>) -> i32"),
        (
            "udondnnCreateActivationDescriptor",
            "%activation_desc_slot",
            "(!udonhl.ptr<0>) -> i32",
        ),
        ("udondnnCreatePoolingDescriptor", "%pooling_desc_slot", "(!udonhl.ptr<0>) -> i32"),
    ]
    for index, (callee, arguments, signature) in enumerate(calls):
        _checked_call(lines, index, callee, arguments, signature)

    load_step = len(calls)
    lines.append(f"  ^step{load_step}:")
    for name in slot_names:
        lines.append(
            f'    %{name} = "udonhl.load"(%{name}_slot) <{{alignment = 4 : i64}}> : '
            '(!udonhl.ptr<0>) -> i32'
        )
    lines.append(f"    cf.br ^step{load_step + 1}")
    configured_calls = [
        ("udondnnSetTensor4dDescriptor", "%input_desc, %zero, %zero, %one, %one, %c28, %c28", "(i32, i32, i32, i32, i32, i32, i32) -> i32"),
        ("udondnnSetTensor4dDescriptor", "%conv_desc, %zero, %zero, %one, %c16, %c24, %c24", "(i32, i32, i32, i32, i32, i32, i32) -> i32"),
        ("udondnnSetTensor4dDescriptor", "%bias_desc, %zero, %zero, %one, %c16, %one, %one", "(i32, i32, i32, i32, i32, i32, i32) -> i32"),
        ("udondnnSetTensor4dDescriptor", "%pool_desc, %zero, %zero, %one, %c16, %c3, %c3", "(i32, i32, i32, i32, i32, i32, i32) -> i32"),
        ("udondnnSetFilter4dDescriptor", "%filter_desc, %zero, %zero, %c16, %one, %c5, %c5", "(i32, i32, i32, i32, i32, i32, i32) -> i32"),
        ("udondnnSetConvolution2dDescriptor", "%convolution_desc, %zero, %zero, %one, %one, %one, %one, %one, %zero", "(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32"),
        ("udondnnSetOpTensorDescriptor", "%op_desc, %zero, %zero, %zero", "(i32, i32, i32, i32) -> i32"),
        ("udondnnSetActivationDescriptor", "%activation_desc, %one, %zero, %fzero", "(i32, i32, i32, f32) -> i32"),
        ("udondnnSetPooling2dDescriptor", "%pooling_desc, %c2, %zero, %c8, %c8, %zero, %zero, %c8, %c8", "(i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32"),
    ]
    constants = {"c2": 2, "c3": 3, "c5": 5, "c8": 8, "c16": 16, "c24": 24, "c28": 28, "c144": 144, "c10": 10}
    lines.insert(lines.index(f"  ^step{load_step}:"), "")
    # Constants dominate all configuration and execution blocks.
    entry_branch = lines.index('    cf.br ^step0')
    for name, value in reversed(list(constants.items())):
        lines.insert(entry_branch, f"    %{name} = arith.constant {value} : i32")
    step = load_step + 1
    for callee, arguments, signature in configured_calls:
        _checked_call(lines, step, callee, arguments, signature)
        step += 1
    operations = [
        ("udondnnConvolutionForward", "%dnn_handle, %alpha, %input_desc, %image, %filter_desc, %conv_weight, %convolution_desc, %zero, %null, %zero, %beta0, %conv_desc, %conv_output", "(i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32"),
        ("udondnnOpTensor", "%dnn_handle, %op_desc, %alpha, %conv_desc, %conv_output, %alpha, %bias_desc, %conv_bias, %beta0, %conv_desc, %conv_output", "(i32, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32"),
        ("udondnnActivationForward", "%dnn_handle, %activation_desc, %alpha, %conv_desc, %conv_output, %beta0, %conv_desc, %conv_output", "(i32, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32"),
        ("udondnnPoolingForward", "%dnn_handle, %pooling_desc, %alpha, %conv_desc, %conv_output, %beta0, %pool_desc, %pool_output", "(i32, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>) -> i32"),
        ("udonblasSgemm", "%blas_handle, %zero, %zero, %one, %c10, %c144, %alpha, %pool_output, %one, %dense_weight, %c144, %beta1, %logits, %one", "(i32, i32, i32, i32, i32, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32"),
    ]
    for callee, arguments, signature in operations:
        _checked_call(lines, step, callee, arguments, signature)
        step += 1
    lines.extend([f"  ^step{step}:", "    cf.br ^cleanup(%success : i32)", "  ^cleanup(%result: i32):"])
    for name in slot_names:
        lines.append(
            f'    %cleanup_{name} = "udonhl.load"(%{name}_slot) '
            '<{alignment = 4 : i64}> : (!udonhl.ptr<0>) -> i32'
        )
    destroy_calls = [
        ("udondnnDestroyPoolingDescriptor", "%cleanup_pooling_desc"),
        ("udondnnDestroyActivationDescriptor", "%cleanup_activation_desc"),
        ("udondnnDestroyOpTensorDescriptor", "%cleanup_op_desc"),
        ("udondnnDestroyConvolutionDescriptor", "%cleanup_convolution_desc"),
        ("udondnnDestroyFilterDescriptor", "%cleanup_filter_desc"),
        ("udondnnDestroyTensorDescriptor", "%cleanup_pool_desc"),
        ("udondnnDestroyTensorDescriptor", "%cleanup_bias_desc"),
        ("udondnnDestroyTensorDescriptor", "%cleanup_conv_desc"),
        ("udondnnDestroyTensorDescriptor", "%cleanup_input_desc"),
        ("udonblasDestroy", "%cleanup_blas_handle"),
        ("udondnnDestroy", "%cleanup_dnn_handle"),
    ]
    for index, (callee, argument) in enumerate(destroy_calls):
        lines.append(f"    %ignored{index} = func.call @{callee}({argument}) : (i32) -> i32")
    lines.extend(
        [
            "    return %result : i32",
            "  }",
            "",
            '  func.func @read_mnist_output(%index: i32) -> f32 attributes {',
            '    vrc.export_id = 3 : i32, vrc.parameter_names = ["index"],',
            '    vrc.public_name = "ReadMnistOutput"',
            '  } {',
            '    %four = arith.constant 4 : i32',
            '    %base = udonhl.address_of @mnist_logits : <0>',
            '    %offset = arith.muli %index, %four : i32',
            '    %position = "udonhl.ptr_add"(%base, %offset) : '
            '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
            '    %value = "udonhl.load"(%position) <{alignment = 4 : i64}> : '
            '(!udonhl.ptr<0>) -> f32',
            '    return %value : f32',
            '  }',
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def emit_mnist_dense_mlir(model: dict) -> str:
    """Lower the MNIST Dense tail to UdonHL and the public UdonBLAS ABI."""
    nodes = model["graph"]["nodes"]
    dense_index = next(
        (index for index, node in enumerate(nodes) if node["op_type"] == "Gemm"),
        None,
    )
    if dense_index is None or dense_index + 1 >= len(nodes):
        raise ValueError("MNIST Dense MLIR requires Gemm followed by Add")
    dense = nodes[dense_index]
    bias_add = nodes[dense_index + 1]
    if (
        bias_add["op_type"] != "Add"
        or dense["outputs"][0] not in bias_add["inputs"]
        or len(dense["inputs"]) != 2
        or len(bias_add["inputs"]) != 2
    ):
        raise ValueError("MNIST Dense MLIR requires Gemm output plus one bias")
    _, weight_name = dense["inputs"]
    bias_name = next(name for name in bias_add["inputs"] if name != dense["outputs"][0])
    initializers = model["graph"]["initializers"]
    if initializers[weight_name]["shape"] != [144, 10]:
        raise ValueError("MNIST Dense MLIR requires [1,144] x [144,10]")
    weight = _float_initializer(initializers[weight_name])
    bias = _float_initializer(initializers[bias_name])
    if len(weight) != 1440 or len(bias) != 10:
        raise ValueError("MNIST Dense MLIR initializer size mismatch")
    packed_weight = [
        weight[feature * 10 + column]
        for column in range(10)
        for feature in range(144)
    ]

    weight_bytes = struct.pack(f"<{len(packed_weight)}f", *packed_weight)
    lines = [
        'module attributes {udoncl.onnx_lowering = "library-calls-v1", '
        'udoncl.onnx_profile = "mnist-dense"} {',
        _udonhl_global("mnist_dense_input", "!llvm.array<144 x f32>", bytes(144 * 4)),
        _udonhl_global("mnist_dense_weight", "!llvm.array<1440 x f32>", weight_bytes),
        _udonhl_global("mnist_dense_output", "!llvm.array<10 x f32>", bytes(10 * 4)),
        '',
        '  func.func private @udonblasCreate(!udonhl.ptr<0>) -> i32 attributes '
        '{udoncl.api = "udonblasCreate", udoncl.library = "UdonBLAS"}',
        '  func.func private @udonblasDestroy(i32) -> i32 attributes '
        '{udoncl.api = "udonblasDestroy", udoncl.library = "UdonBLAS"}',
        '  func.func private @udonblasSgemm(i32, i32, i32, i32, i32, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, i32, '
        '!udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32 attributes '
        '{udoncl.api = "udonblasSgemm", udoncl.library = "UdonBLAS"}',
        '',
        '  func.func @write_mnist_dense_input(%index: i32, %value: f32) -> i32 attributes {',
        '    vrc.export_id = 1 : i32,',
        '    vrc.parameter_names = ["index", "value"],',
        '    vrc.public_name = "WriteMnistDenseInput"',
        '  } {',
        '    %four = arith.constant 4 : i32',
        '    %input = udonhl.address_of @mnist_dense_input : <0>',
        '    %offset = arith.muli %index, %four : i32',
        '    %position = "udonhl.ptr_add"(%input, %offset) : '
        '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
        '    "udonhl.store"(%value, %position) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    %success = arith.constant 0 : i32',
        '    return %success : i32',
        '  }',
        '',
        '  func.func @run_mnist_dense() -> i32 attributes {',
        '    vrc.async_effect = "may_suspend",',
        '    vrc.export_id = 2 : i32,',
        '    vrc.public_name = "RunMnistDense"',
        '  } {',
        '    %four = arith.constant 4 : i32',
        '    %input = udonhl.address_of @mnist_dense_input : <0>',
        '    %weight = udonhl.address_of @mnist_dense_weight : <0>',
        '    %output = udonhl.address_of @mnist_dense_output : <0>',
        '    %handle_slot = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %alpha = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %beta = "udonhl.alloca"(%four) <{alignment = 4 : i64}> : '
        '(i32) -> !udonhl.ptr<0>',
        '    %one = arith.constant 1.0 : f32',
        '    "udonhl.store"(%one, %alpha) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
        '    "udonhl.store"(%one, %beta) <{alignment = 4 : i64}> : '
        '(f32, !udonhl.ptr<0>) -> ()',
    ]
    for index in range(10):
        lines.extend(
            [
                f'    %offset{index} = arith.constant {index * 4} : i32',
                f'    %output{index} = "udonhl.ptr_add"(%output, %offset{index}) : '
                '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
                f'    %bias{index} = arith.constant {_c_float(bias[index])[:-1]} : f32',
                f'    "udonhl.store"(%bias{index}, %output{index}) '
                '<{alignment = 4 : i64}> : (f32, !udonhl.ptr<0>) -> ()',
            ]
        )
    lines.extend(
        [
            '    %success = arith.constant 0 : i32',
            '    %op_n = arith.constant 0 : i32',
            '    %m = arith.constant 1 : i32',
            '    %n = arith.constant 10 : i32',
            '    %k = arith.constant 144 : i32',
            '    %lda = arith.constant 1 : i32',
            '    %ldb = arith.constant 144 : i32',
            '    %ldc = arith.constant 1 : i32',
            '    %create_status = func.call @udonblasCreate(%handle_slot) : '
            '(!udonhl.ptr<0>) -> i32',
            '    %created = arith.cmpi eq, %create_status, %success : i32',
            '    cf.cond_br %created, ^sgemm, ^return(%create_status : i32)',
            '  ^sgemm:',
            '    %handle = "udonhl.load"(%handle_slot) <{alignment = 4 : i64}> : '
            '(!udonhl.ptr<0>) -> i32',
            '    %sgemm_status = func.call @udonblasSgemm('
            '%handle, %op_n, %op_n, %m, %n, %k, %alpha, %input, %lda, '
            '%weight, %ldb, %beta, %output, %ldc) : '
            '(i32, i32, i32, i32, i32, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, '
            'i32, !udonhl.ptr<0>, i32, !udonhl.ptr<0>, !udonhl.ptr<0>, i32) -> i32',
            '    %computed = arith.cmpi eq, %sgemm_status, %success : i32',
            '    cf.cond_br %computed, ^destroy, ^return(%sgemm_status : i32)',
            '  ^destroy:',
            '    %destroy_status = func.call @udonblasDestroy(%handle) : (i32) -> i32',
            '    cf.br ^return(%destroy_status : i32)',
            '  ^return(%status: i32):',
            '    return %status : i32',
            '  }',
            '',
            '  func.func @read_mnist_dense_output(%index: i32) -> f32 attributes {',
            '    vrc.export_id = 3 : i32,',
            '    vrc.parameter_names = ["index"],',
            '    vrc.public_name = "ReadMnistDenseOutput"',
            '  } {',
            '    %four = arith.constant 4 : i32',
            '    %output = udonhl.address_of @mnist_dense_output : <0>',
            '    %offset = arith.muli %index, %four : i32',
            '    %position = "udonhl.ptr_add"(%output, %offset) : '
            '(!udonhl.ptr<0>, i32) -> !udonhl.ptr<0>',
            '    %value = "udonhl.load"(%position) <{alignment = 4 : i64}> : '
            '(!udonhl.ptr<0>) -> f32',
            '    return %value : f32',
            '  }',
            '}',
            '',
        ]
    )
    return "\n".join(lines)


def emit_mnist_dense_host(model: dict) -> str:
    """Lower the trained MNIST Dense tail to the public UdonBLAS SGEMM ABI."""
    nodes = model["graph"]["nodes"]
    dense_index = next(
        (index for index, node in enumerate(nodes) if node["op_type"] == "Gemm"),
        None,
    )
    if dense_index is None or dense_index + 1 >= len(nodes):
        raise ValueError("MNIST Dense profile requires Gemm followed by Add")
    dense = nodes[dense_index]
    bias_add = nodes[dense_index + 1]
    if (
        bias_add["op_type"] != "Add"
        or dense["outputs"][0] not in bias_add["inputs"]
        or len(dense["inputs"]) != 2
        or len(bias_add["inputs"]) != 2
    ):
        raise ValueError("MNIST Dense profile requires Gemm output plus one bias")
    _, weight_name = dense["inputs"]
    bias_name = next(name for name in bias_add["inputs"] if name != dense["outputs"][0])
    initializers = model["graph"]["initializers"]
    if initializers[weight_name]["shape"] != [144, 10]:
        raise ValueError("MNIST Dense profile requires [1,144] x [144,10]")
    weight = _float_initializer(initializers[weight_name])
    bias = _float_initializer(initializers[bias_name])
    if len(weight) != 1440 or len(bias) != 10:
        raise ValueError("MNIST Dense initializer size mismatch")

    # ONNX stores [K,N] row-major. cuBLAS consumes the same logical matrix as
    # [K,N] column-major, so pack one contiguous K column per output feature.
    packed_weight = [weight[feature * 10 + column] for column in range(10) for feature in range(144)]
    lines = [
        "/* Generated from ONNX by udon-onnx-host; no execution plan is used. */",
        "#include <udonblas.h>",
        "",
        "static float mnist_dense_input[144];",
    ]
    _array(lines, "static float mnist_dense_weight[1440]", packed_weight)
    lines.extend(
        [
            "static float mnist_dense_output[10];",
            "",
            "int write_mnist_dense_input(int index, float value) {",
            "  mnist_dense_input[index] = value;",
            "  return UDONBLAS_STATUS_SUCCESS;",
            "}",
            "",
            "int run_mnist_dense(void) {",
            "  udonblasHandle_t handle = 0;",
            "  const float alpha = 1.0f;",
            "  const float beta = 1.0f;",
            "  udonblasStatus_t status;",
            *[
                f"  mnist_dense_output[{column}] = {_c_float(bias[column])};"
                for column in range(10)
            ],
            "  status = udonblasCreate(&handle);",
            "  if (status != UDONBLAS_STATUS_SUCCESS)",
            "    return status;",
            "  status = udonblasSgemm(handle, UDONBLAS_OP_N, UDONBLAS_OP_N, 1, 10, 144,",
            "                         &alpha, mnist_dense_input, 1, mnist_dense_weight, 144,",
            "                         &beta, mnist_dense_output, 1);",
            "  if (status != UDONBLAS_STATUS_SUCCESS)",
            "    return status;",
            "  return udonblasDestroy(handle);",
            "}",
            "",
            "float read_mnist_dense_output(int index) {",
            "  return mnist_dense_output[index];",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def emit_mnist_host(model: dict) -> str:
    """Lower the complete trained MNIST graph to three OpenCL AOT kernels."""
    nodes = model["graph"]["nodes"]
    if [node["op_type"] for node in nodes] != [
        "Conv",
        "Add",
        "Relu",
        "AveragePool",
        "Flatten",
        "Gemm",
        "Add",
    ]:
        raise ValueError("MNIST profile requires Conv/Add/Relu/Pool/Flatten/Gemm/Add")
    initializers = model["graph"]["initializers"]
    required = {
        "conv.weight": [16, 1, 5, 5],
        "conv.bias": [1, 16, 1, 1],
        "dense.weight": [144, 10],
        "dense.bias": [10],
    }
    for name, shape in required.items():
        if name not in initializers or initializers[name]["shape"] != shape:
            raise ValueError(f"MNIST initializer {name!r} must have shape {shape}")
    conv_weight = _float_initializer(initializers["conv.weight"])
    conv_bias = _float_initializer(initializers["conv.bias"])
    dense_weight = _float_initializer(initializers["dense.weight"])
    dense_bias = _float_initializer(initializers["dense.bias"])
    packed = [0.0] * 10800
    packed[784:1184] = conv_weight
    packed[1184:1200] = conv_bias
    packed[1200:2640] = [
        dense_weight[feature * 10 + digit]
        for digit in range(10)
        for feature in range(144)
    ]
    packed[2640:2650] = dense_bias

    lines = [
        "/* Generated from ONNX by udon-onnx-host; no execution plan is used. */",
        "#include <CL/cl.h>",
        "",
    ]
    _array(lines, "static float mnist_packed[10800]", packed)
    lines.extend(
        [
            "static float mnist_logits[10];",
            "",
            "int write_mnist_input(int index, float value) {",
            "  if (index < 0 || index >= 784)",
            "    return CL_INVALID_VALUE;",
            "  mnist_packed[index] = value;",
            "  return CL_SUCCESS;",
            "}",
            "",
            "int run_mnist(void) {",
            "  cl_platform_id platform = 0;",
            "  cl_device_id device = 0;",
            "  cl_context context = 0;",
            "  cl_command_queue queue = 0;",
            "  cl_program program = 0;",
            "  cl_kernel conv = 0, pool = 0, dense = 0;",
            "  cl_mem first = 0, second = 0;",
            "  const char *source = \"udoncl-aot\";",
            "  const size_t global = 10800, local = 1;",
            "  cl_int status = CL_SUCCESS;",
            "  if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS ||",
            "      clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, 0) != CL_SUCCESS)",
            "    return CL_DEVICE_NOT_FOUND;",
            "  context = clCreateContext(0, 1, &device, 0, 0, &status);",
            "  if (!context || status != CL_SUCCESS)",
            "    return status;",
            "  queue = clCreateCommandQueue(context, device, 0, &status);",
            "  first = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(mnist_packed), 0, &status);",
            "  second = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(mnist_packed), 0, &status);",
            "  program = clCreateProgramWithSource(context, 1, &source, 0, &status);",
            "  if (!queue || !first || !second || !program || status != CL_SUCCESS ||",
            "      clBuildProgram(program, 0, 0, \"-cl-std=CL1.2\", 0, 0) != CL_SUCCESS)",
            "    return CL_OUT_OF_RESOURCES;",
            "  conv = clCreateKernel(program, \"mnist_conv\", &status);",
            "  pool = clCreateKernel(program, \"mnist_pool\", &status);",
            "  dense = clCreateKernel(program, \"mnist_dense\", &status);",
            "  if (!conv || !pool || !dense || status != CL_SUCCESS)",
            "    return status;",
            "#define SET_BUFFERS(kernel, input, output) \\",
            "  do { \\",
            "    if (clSetKernelArg(kernel, 0, sizeof(input), &(input)) != CL_SUCCESS || \\",
            "        clSetKernelArg(kernel, 1, sizeof(output), &(output)) != CL_SUCCESS) \\",
            "      return CL_INVALID_KERNEL_ARGS; \\",
            "  } while (0)",
            "  SET_BUFFERS(conv, first, second);",
            "  SET_BUFFERS(pool, second, first);",
            "  SET_BUFFERS(dense, first, second);",
            "#undef SET_BUFFERS",
            "  if (clEnqueueWriteBuffer(queue, first, CL_TRUE, 0, sizeof(mnist_packed),",
            "                           mnist_packed, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueNDRangeKernel(queue, conv, 1, 0, &global, &local, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueNDRangeKernel(queue, pool, 1, 0, &global, &local, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueNDRangeKernel(queue, dense, 1, 0, &global, &local, 0, 0, 0) != CL_SUCCESS ||",
            "      clEnqueueReadBuffer(queue, second, CL_TRUE, 0, sizeof(mnist_logits),",
            "                          mnist_logits, 0, 0, 0) != CL_SUCCESS ||",
            "      clFinish(queue) != CL_SUCCESS)",
            "    return CL_OUT_OF_RESOURCES;",
            "  clReleaseKernel(dense);",
            "  clReleaseKernel(pool);",
            "  clReleaseKernel(conv);",
            "  clReleaseProgram(program);",
            "  clReleaseMemObject(second);",
            "  clReleaseMemObject(first);",
            "  clReleaseCommandQueue(queue);",
            "  clReleaseContext(context);",
            "  return CL_SUCCESS;",
            "}",
            "",
            "float read_mnist_output(int index) {",
            "  return mnist_logits[index];",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--mlir-output", type=Path)
    parser.add_argument("--profile", choices=("mnist", "mnist-dense"), required=True)
    args = parser.parse_args()
    model = parse_model(args.model.read_bytes())
    source = emit_mnist_host(model) if args.profile == "mnist" else emit_mnist_dense_host(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    if args.mlir_output is not None:
        mlir = (
            emit_mnist_mlir(model)
            if args.profile == "mnist"
            else emit_mnist_dense_mlir(model)
        )
        args.mlir_output.parent.mkdir(parents=True, exist_ok=True)
        args.mlir_output.write_text(mlir, encoding="utf-8")
    print(f"udon_onnx_host_profile={args.profile} bytes={len(source.encode())}")


if __name__ == "__main__":
    main()
