"""Import an ONNX protobuf and run the locked ONNX-to-ONNX MLIR pipeline."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

STRUCTURAL_OPERATORS = {'EntryPoint', 'Return'}
OPERATOR_PATTERN = re.compile(r'(?<![A-Za-z0-9_])onnx\.([A-Za-z0-9_]+)')


def operators(mlir):
    return set(OPERATOR_PATTERN.findall(mlir)) - STRUCTURAL_OPERATORS


def canonicalize(model, output, importer, optimizer, supported=()):
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
        subprocess.run(
            [importer, model, '-o', imported], check=True)
        subprocess.run([
            optimizer, imported, '--onnx-op-transform', '--canonicalize',
            '-o', transformed], check=True)
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
    args = parser.parse_args()
    try:
        canonicalize(args.model, args.output, args.importer.resolve(),
                     args.optimizer.resolve(), args.supported_operator)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
