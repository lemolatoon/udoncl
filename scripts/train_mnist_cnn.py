"""Train the small udoncl MNIST CNN and emit deterministic C fixtures.

This is a maintainer tool, not a build dependency. It downloads the canonical
MNIST gzip files into --cache and requires NumPy.
"""

import argparse
import gzip
from pathlib import Path
import struct
import urllib.request

import numpy as np


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images": "train-images-idx3-ubyte.gz",
    "train-labels": "train-labels-idx1-ubyte.gz",
    "test-images": "t10k-images-idx3-ubyte.gz",
    "test-labels": "t10k-labels-idx1-ubyte.gz",
}


def download(cache: Path, name: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / FILES[name]
    if not path.exists():
        print(f"downloading {path.name}", flush=True)
        urllib.request.urlretrieve(BASE_URL + path.name, path)
    return path


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051 or rows != 28 or columns != 28:
            raise RuntimeError(f"unexpected MNIST image header in {path}")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, rows, columns)


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise RuntimeError(f"unexpected MNIST label header in {path}")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count)


def windows(images: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(images, (5, 5), axis=(1, 2))


def forward(images, conv_weight, conv_bias, dense_weight, dense_bias):
    patches = windows(images)
    conv = np.einsum("byxij,kij->byxk", patches, conv_weight, optimize=True)
    conv += conv_bias
    relu = np.maximum(conv, 0.0)
    pooled = relu.reshape(len(images), 6, 4, 6, 4, 8).mean(axis=(2, 4))
    features = pooled.reshape(len(images), 288)
    logits = features @ dense_weight.T + dense_bias
    return patches, conv, features, logits


def accuracy(images, labels, parameters, batch_size=512):
    predictions = []
    for begin in range(0, len(images), batch_size):
        batch = images[begin : begin + batch_size].astype(np.float32) / 255.0
        predictions.append(np.argmax(forward(batch, *parameters)[-1], axis=1))
    return np.mean(np.concatenate(predictions) == labels)


def train(train_images, train_labels, test_images, test_labels, epochs):
    rng = np.random.default_rng(0x55444F4E)
    conv_weight = rng.normal(0.0, 0.12, (8, 5, 5)).astype(np.float32)
    conv_bias = np.zeros(8, dtype=np.float32)
    dense_weight = rng.normal(0.0, 0.04, (10, 288)).astype(np.float32)
    dense_bias = np.zeros(10, dtype=np.float32)
    parameters = [conv_weight, conv_bias, dense_weight, dense_bias]
    moments = [np.zeros_like(value) for value in parameters]
    velocities = [np.zeros_like(value) for value in parameters]
    step = 0
    batch_size = 128

    for epoch in range(epochs):
        order = rng.permutation(len(train_images))
        total_loss = 0.0
        for begin in range(0, len(order), batch_size):
            selection = order[begin : begin + batch_size]
            images = train_images[selection].astype(np.float32) / 255.0
            labels = train_labels[selection]
            patches, conv, features, logits = forward(images, *parameters)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            total_loss -= np.log(probabilities[np.arange(len(labels)), labels] + 1e-12).sum()

            probabilities[np.arange(len(labels)), labels] -= 1.0
            probabilities /= len(labels)
            dense_weight_gradient = probabilities.T @ features
            dense_bias_gradient = probabilities.sum(axis=0)
            feature_gradient = probabilities @ dense_weight
            pool_gradient = feature_gradient.reshape(len(labels), 6, 6, 8)
            relu_gradient = np.broadcast_to(
                pool_gradient[:, :, None, :, None, :] / 16.0,
                (len(labels), 6, 4, 6, 4, 8),
            ).reshape(len(labels), 24, 24, 8)
            conv_gradient = relu_gradient * (conv > 0.0)
            conv_weight_gradient = np.einsum(
                "byxk,byxij->kij", conv_gradient, patches, optimize=True
            )
            conv_bias_gradient = conv_gradient.sum(axis=(0, 1, 2))
            gradients = [
                conv_weight_gradient,
                conv_bias_gradient,
                dense_weight_gradient,
                dense_bias_gradient,
            ]

            step += 1
            learning_rate = 0.001
            for value, gradient, moment, velocity in zip(
                parameters, gradients, moments, velocities
            ):
                moment *= 0.9
                moment += 0.1 * gradient
                velocity *= 0.999
                velocity += 0.001 * gradient * gradient
                corrected_moment = moment / (1.0 - 0.9**step)
                corrected_velocity = velocity / (1.0 - 0.999**step)
                value -= learning_rate * corrected_moment / (
                    np.sqrt(corrected_velocity) + 1e-8
                )

        score = accuracy(test_images, test_labels, parameters)
        print(
            f"epoch={epoch + 1} loss={total_loss / len(train_images):.5f} "
            f"test_accuracy={score:.5f}",
            flush=True,
        )
    return parameters


def c_floats(values):
    return ",\n  ".join(
        ", ".join(f"{float(value):.9g}f" for value in row)
        for row in np.asarray(values).reshape(-1, 8)
    )


def emit_weights(path: Path, parameters):
    conv_weight, conv_bias, dense_weight, dense_bias = parameters
    path.write_text(
        "// Generated by scripts/train_mnist_cnn.py; seed 0x55444f4e.\n"
        "#pragma once\n"
        f"static const float kMnistConvWeight[200] = {{\n  {c_floats(conv_weight)}\n}};\n"
        f"static const float kMnistConvBias[8] = {{\n  {c_floats(conv_bias)}\n}};\n"
        f"static const float kMnistDenseWeight[2880] = {{\n  {c_floats(dense_weight)}\n}};\n"
        f"static const float kMnistDenseBias[10] = {{\n  {', '.join(f'{float(x):.9g}f' for x in dense_bias)}\n}};\n",
        encoding="utf-8",
    )


def emit_samples(path: Path, images, labels, parameters):
    selected = []
    for digit in range(10):
        for index in np.flatnonzero(labels == digit):
            image = images[index].astype(np.float32) / 255.0
            prediction = int(np.argmax(forward(image[None], *parameters)[-1][0]))
            if prediction == digit:
                selected.append(images[index])
                break
        else:
            raise RuntimeError(f"no correctly classified sample for {digit}")
    rows = []
    for image in selected:
        rows.append(",\n    ".join(
            ", ".join(str(int(value)) for value in row)
            for row in image.reshape(-1, 28)
        ))
    body = ",\n  ".join("{\n    " + row + "\n  }" for row in rows)
    path.write_text(
        "// Ten canonical MNIST test images selected by scripts/train_mnist_cnn.py.\n"
        "#pragma once\n"
        f"static const unsigned char kMnistSamples[10][784] = {{\n  {body}\n}};\n",
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=root / "build/mnist-data")
    parser.add_argument("--output", type=Path, default=root / "examples/mnist-cnn")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_images = load_images(download(args.cache, "train-images"))
    train_labels = load_labels(download(args.cache, "train-labels"))
    test_images = load_images(download(args.cache, "test-images"))
    test_labels = load_labels(download(args.cache, "test-labels"))
    parameters = train(
        train_images, train_labels, test_images, test_labels, args.epochs
    )
    args.output.mkdir(parents=True, exist_ok=True)
    emit_weights(args.output / "weights.h", parameters)
    emit_samples(args.output / "samples.h", test_images, test_labels, parameters)
    score = accuracy(test_images, test_labels, parameters)
    (args.output / "training.txt").write_text(
        f"architecture=conv5x5x8-relu-avgpool4x4-dense10\n"
        f"epochs={args.epochs}\nseed=0x55444f4e\ntest_accuracy={score:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
