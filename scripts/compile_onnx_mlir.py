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
    importer,
    optimizer,
    host_optimizer=None,
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
    for tool in (importer, optimizer):
        if not tool.is_file():
            raise RuntimeError(f'ONNX MLIR tool does not exist: {tool}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=output.name + '.onnx-mlir-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        imported = temporary / 'imported.mlir'
        portable = temporary / ('portable-' + output.name)
        transformed = temporary / output.name
        temporary_static_data = temporary / 'static-tensors'
        temporary_metadata = temporary / 'host-metadata.json'
        import_command = [importer, model, '-o', imported]
        for dimension in dimensions:
            import_command.extend(['--dim', dimension])
        for input_shape in input_shapes:
            import_command.extend(['--input-shape', input_shape])
        subprocess.run(import_command, check=True)
        optimize_command = [
            optimizer, imported, '--verify-each=false', '--shape-inference',
            '--simplify-shape-related-ops-onnx', '--constprop-onnx',
            '--standard-func-return', '--canonicalize',
        ]
        if lower_to_host:
            if host_optimizer is None or not host_optimizer.is_file():
                raise RuntimeError(
                    f'UdonCL Host optimizer does not exist: {host_optimizer}')
            optimize_command.extend([
                '--udoncl-canonicalize-onnx-library-patterns',
                '--udoncl-infer-onnx-shapes',
                '--shape-inference',
                '--simplify-shape-related-ops-onnx',
                '--constprop-onnx',
                '--udoncl-infer-onnx-shapes',
                '--udoncl-materialize-onnx-constants',
                '--udoncl-externalize-onnx-initializers=' +
                f'static-data-directory={temporary_static_data}',
                '--udoncl-identify-onnx-sequence-state',
                '--udoncl-lower-onnx-to-host',
                '--canonicalize',
                '--udoncl-emit-bytecode',
            ])
        else:
            optimize_command.extend([
                '--mlir-print-debuginfo',
                '--mlir-elide-elementsattrs-if-larger=64',
            ])
        optimize_command.extend(['-o', portable if lower_to_host else transformed])
        subprocess.run(optimize_command, check=True)
        if lower_to_host:
            subprocess.run([
                host_optimizer, portable, '--udoncl-finalize-onnx-host',
                '--canonicalize',
                '--udoncl-emit-onnx-host-metadata=' +
                f'output={temporary_metadata}',
                '--emit-bytecode', '-o', transformed,
            ], check=True)
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
    parser.add_argument('--importer', type=Path, required=True)
    parser.add_argument('--optimizer', type=Path, required=True)
    parser.add_argument('--host-optimizer', type=Path)
    parser.add_argument('--dimension', action='append', default=[])
    parser.add_argument('--input-shape', action='append', default=[])
    parser.add_argument('--lower-to-host', action='store_true')
    parser.add_argument('--static-data-directory', type=Path)
    parser.add_argument('--host-metadata', type=Path)
    args = parser.parse_args()
    try:
        canonicalize(args.model, args.output, args.importer.resolve(),
                     args.optimizer.resolve(),
                     args.host_optimizer.resolve()
                     if args.host_optimizer else None, args.dimension,
                     args.input_shape, args.lower_to_host,
                     args.static_data_directory, args.host_metadata)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
