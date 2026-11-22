"""Write the MNIST CNN input and CPU reference logits of the Unity MNIST test.

The generic MNIST U# test (PublicMnistOnnxRuntimeLifecycleTests) runs the
examples/onnx-mnist artifact on this input and compares its logits with the
reference computed here from the trained weights in examples/mnist-cnn.
Both files are little-endian float32 arrays.
"""

import importlib.util
from pathlib import Path
import struct
import sys


def main():
    root, output = (Path(value).resolve() for value in sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "mnist_onnx_generator", root / "examples/mnist-cnn/generate_onnx.py"
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    header = (root / "examples/mnist-cnn/weights.h").read_text(encoding="utf-8")
    conv_weight = generator.read_weight_array(header, "kMnistConvWeight", 400)
    conv_bias = generator.read_weight_array(header, "kMnistConvBias", 16)
    dense_weight = generator.read_weight_array(header, "kMnistDenseWeight", 1440)
    dense_bias = generator.read_weight_array(header, "kMnistDenseBias", 10)

    image = [((index * 17 + 3) % 101) / 100.0 for index in range(28 * 28)]
    # Valid 5x5 convolution with 16 channels, bias and ReLU.
    relu = []
    for channel in range(16):
        for output_y in range(24):
            for output_x in range(24):
                value = 0.0
                for kernel_y in range(5):
                    for kernel_x in range(5):
                        value += (
                            image[(output_y + kernel_y) * 28 + output_x + kernel_x]
                            * conv_weight[(channel * 5 + kernel_y) * 5 + kernel_x]
                        )
                relu.append(max(value + conv_bias[channel], 0.0))
    # 8x8 average pooling to 16x3x3 features in channel-major order.
    pooled = []
    for channel in range(16):
        for output_y in range(3):
            for output_x in range(3):
                value = 0.0
                for window_y in range(8):
                    for window_x in range(8):
                        input_y = output_y * 8 + window_y
                        input_x = output_x * 8 + window_x
                        value += relu[(channel * 24 + input_y) * 24 + input_x]
                pooled.append(value / 64.0)
    # Dense layer; the trained weights index features as pixel * 16 + channel.
    logits = []
    for label in range(10):
        partials = [
            sum(
                pooled[feature]
                * dense_weight[label * 144 + (feature % 9) * 16 + feature // 9]
                for feature in range(chunk * 16, (chunk + 1) * 16)
            )
            for chunk in range(9)
        ]
        logits.append(sum(partials) + dense_bias[label])

    for name, values in (
        ("unity-mnist-input.bytes", image),
        ("unity-mnist-reference.bytes", logits),
    ):
        (output / name).write_bytes(struct.pack(f"<{len(values)}f", *values))
    predicted = max(range(10), key=lambda label: logits[label])
    print(f"mnist_reference=written predicted={predicted} output={output}")


if __name__ == "__main__":
    main()
