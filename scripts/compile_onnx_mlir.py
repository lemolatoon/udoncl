"""Orchestrate the locked C++ ONNX import and MLIR pass pipeline."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def canonicalize(
    model,
    output,
    frontend,
    dimensions=(),
    input_shapes=(),
    lower_to_host=False,
    static_data_directory=None,
    host_metadata=None,
):
    model = model.resolve()
    output = output.resolve()
    if not model.is_file():
        raise RuntimeError(f'ONNX model does not exist: {model}')
    if not frontend.is_file():
        raise RuntimeError(f'UdonCL ONNX frontend does not exist: {frontend}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=output.name + '.onnx-mlir-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        transformed = temporary / output.name
        temporary_static_data = temporary / 'static-tensors'
        temporary_metadata = temporary / 'host-metadata.json'
        specialization = (
            '--udoncl-specialize-onnx-inputs=' +
            f'dimensions={";".join(dimensions)} ' +
            f'input-shapes={";".join(input_shapes)}')
        command = [
            frontend, model, specialization, '--udoncl-infer-onnx-shapes',
            '--verify-each=false', '--shape-inference',
            '--simplify-shape-related-ops-onnx', '--constprop-onnx',
            '--standard-func-return', '--canonicalize',
        ]
        if lower_to_host:
            command.extend([
                '--udoncl-canonicalize-onnx-library-patterns',
                '--udoncl-infer-onnx-shapes',
                '--udoncl-simplify-onnx-empty-concat',
                '--udoncl-materialize-onnx-constants',
                '--udoncl-externalize-onnx-initializers=' +
                f'static-data-directory={temporary_static_data}',
                '--udoncl-identify-onnx-sequence-state',
                '--udoncl-lower-onnx-to-host',
                '--udoncl-validate-onnx-host-graph',
                '--udoncl-plan-onnx-host-graph',
                '--udoncl-materialize-onnx-host-globals',
                '--udoncl-materialize-onnx-host-model-registry',
                '--udoncl-attach-onnx-host-metadata',
                '--udoncl-finalize-onnx-host',
                '--udoncl-cleanup-onnx-host-source',
                '--canonicalize',
                '--udoncl-emit-onnx-host-metadata=' +
                f'output={temporary_metadata}',
                '--emit-bytecode',
            ])
        else:
            command.extend([
                '--mlir-print-debuginfo',
                '--mlir-elide-elementsattrs-if-larger=64',
            ])
        command.extend(['-o', transformed])
        subprocess.run(command, check=True)
        if lower_to_host:
            if static_data_directory is not None:
                static_data_directory = static_data_directory.resolve()
                static_data_directory.parent.mkdir(parents=True, exist_ok=True)
                if static_data_directory.exists():
                    shutil.rmtree(static_data_directory)
                os.replace(temporary_static_data, static_data_directory)
            if host_metadata is not None:
                host_metadata = host_metadata.resolve()
                host_metadata.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_metadata, host_metadata)
        os.replace(transformed, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--frontend', type=Path, required=True,
                        help=argparse.SUPPRESS)
    parser.add_argument('--dimension', action='append', default=[])
    parser.add_argument('--input-shape', action='append', default=[])
    parser.add_argument('--lower-to-host', action='store_true')
    parser.add_argument('--static-data-directory', type=Path)
    parser.add_argument('--host-metadata', type=Path)
    args = parser.parse_args()
    try:
        canonicalize(args.model, args.output, args.frontend.resolve(),
                     args.dimension,
                     args.input_shape, args.lower_to_host,
                     args.static_data_directory, args.host_metadata)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
