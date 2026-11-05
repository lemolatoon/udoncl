"""Orchestrate the locked C++ ONNX import and MLIR pass pipeline."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

STRUCTURAL_OPERATORS = {'Constant', 'EntryPoint', 'NoValue', 'Return'}
OPERATOR_PATTERN = re.compile(r'(?<![A-Za-z0-9_])onnx\.([A-Z][A-Za-z0-9_]*)')


def operators(mlir):
    return set(OPERATOR_PATTERN.findall(mlir)) - STRUCTURAL_OPERATORS


def canonicalize(
    model,
    output,
    importer,
    optimizer,
    supported=(),
    dimensions=(),
    input_shapes=(),
    lower_to_host=False,
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
        transformed = temporary / output.name
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
            optimize_command.extend([
                '--udoncl-infer-onnx-shapes',
                '--udoncl-materialize-onnx-constants',
                '--udoncl-lower-onnx-to-host',
                '--one-shot-bufferize=bufferize-function-boundaries copy-before-write',
                '--canonicalize',
                '--udoncl-emit-bytecode',
            ])
        else:
            optimize_command.extend([
                '--mlir-print-debuginfo',
                '--mlir-elide-elementsattrs-if-larger=64',
            ])
        optimize_command.extend(['-o', transformed])
        subprocess.run(optimize_command, check=True)
        if lower_to_host:
            os.replace(transformed, output)
            return
        text = transformed.read_text(encoding='utf-8')
        if supported:
            unsupported = sorted(operators(text) - set(supported))
            if unsupported:
                raise RuntimeError(
                    'Unsupported ONNX operators after canonicalization: ' +
                    ', '.join(unsupported))
        os.replace(transformed, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--importer', type=Path, required=True)
    parser.add_argument('--optimizer', type=Path, required=True)
    parser.add_argument('--supported-operator', action='append', default=[])
    parser.add_argument('--dimension', action='append', default=[])
    parser.add_argument('--input-shape', action='append', default=[])
    parser.add_argument('--lower-to-host', action='store_true')
    args = parser.parse_args()
    try:
        canonicalize(args.model, args.output, args.importer.resolve(),
                     args.optimizer.resolve(), args.supported_operator,
                     args.dimension, args.input_shape, args.lower_to_host)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
