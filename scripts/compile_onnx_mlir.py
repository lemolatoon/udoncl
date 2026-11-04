"""Import an ONNX protobuf and run the locked ONNX-to-ONNX MLIR pipeline."""
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
        subprocess.run([
            optimizer, imported, '--verify-each=false', '--shape-inference',
            '--simplify-shape-related-ops-onnx', '--constprop-onnx',
            '--standard-func-return', '--canonicalize',
            '--mlir-print-debuginfo',
            '--mlir-elide-elementsattrs-if-larger=64', '-o', transformed],
            check=True)
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
    args = parser.parse_args()
    try:
        canonicalize(args.model, args.output, args.importer.resolve(),
                     args.optimizer.resolve(), args.supported_operator,
                     args.dimension, args.input_shape)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
