"""Parse canonical ONNX-MLIR text into an in-memory lowering graph.

This is deliberately not a serialized execution plan.  The graph exists only
between the ONNX canonicalization and Udon library-call lowering stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int | None, ...] | None
    element_type: str

    @property
    def rank(self) -> int | None:
        return None if self.shape is None else len(self.shape)


@dataclass(frozen=True)
class Value:
    name: str
    type: TensorType | None
    initializer: str | None = None
    ssa_name: str | None = None


@dataclass(frozen=True)
class Operator:
    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_types: tuple[TensorType | None, ...]
    output_types: tuple[TensorType | None, ...]
    attributes: dict[str, object] = field(default_factory=dict)
    node_name: str | None = None
    location: str | None = None


@dataclass(frozen=True)
class Graph:
    name: str
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    initializers: tuple[Value, ...]
    operators: tuple[Operator, ...]


_FUNC = re.compile(r"^\s*func\.func @([A-Za-z0-9_.$-]+)\((.*)\) -> (.*?) \{$")
_LOCATION = re.compile(r'^\s*#(loc[0-9]+) = loc\("((?:[^"\\]|\\.)*)"\)\s*$')
_VALUE_NAME = re.compile(r"%[A-Za-z0-9_.$-]+")
_INTEGER = re.compile(r"[-+]?[0-9]+(?:\s*:\s*[a-z]+[0-9]+)?$")
_FLOAT = re.compile(
    r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    r"(?:\s*:\s*f[0-9]+)?$"
)


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "([{<":
            stack.append(character)
        elif character in ")]}>":
            if not stack or stack[-1] != pairs[character]:
                raise ValueError(f"unbalanced canonical ONNX MLIR near {text!r}")
            stack.pop()
        elif character == separator and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if quoted or stack:
        raise ValueError(f"unbalanced canonical ONNX MLIR near {text!r}")
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _balanced(text: str, start: int, opening: str, closing: str) -> tuple[str, int]:
    if start >= len(text) or text[start] != opening:
        raise ValueError(f"expected {opening!r} in canonical ONNX MLIR")
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError(f"unterminated {opening!r} in canonical ONNX MLIR")


def parse_tensor_type(text: str) -> TensorType | None:
    text = text.strip()
    if text == "none":
        return None
    if not text.startswith("tensor<") or not text.endswith(">"):
        raise ValueError(f"unsupported canonical ONNX value type {text!r}")
    body = text[7:-1]
    if body.startswith("*x"):
        return TensorType(None, body[2:])
    fields = body.split("x")
    element_type = fields[-1]
    dimensions: list[int | None] = []
    for dimension in fields[:-1]:
        if dimension == "?":
            dimensions.append(None)
        elif dimension.isdigit():
            dimensions.append(int(dimension))
        else:
            raise ValueError(f"unsupported canonical ONNX dimension {dimension!r}")
    return TensorType(tuple(dimensions), element_type)


def _parse_type_list(text: str) -> tuple[TensorType | None, ...]:
    text = text.strip()
    if text.startswith("("):
        body, end = _balanced(text, 0, "(", ")")
        if text[end:].strip():
            raise ValueError(f"unexpected text after ONNX result types: {text[end:]!r}")
        return tuple(_parse_annotated_type(item) for item in _split_top_level(body))
    return (parse_tensor_type(text),)


def _parse_annotated_type(text: str) -> TensorType | None:
    match = re.match(r"(tensor<[^>]+>|none)(?:\s+.*)?$", text.strip())
    if not match:
        raise ValueError(f"unsupported canonical ONNX value type {text!r}")
    return parse_tensor_type(match.group(1))


def _parse_attribute_value(text: str) -> object:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return bytes(text[1:-1], "utf-8").decode("unicode_escape")
    if text.startswith("[") and text.endswith("]"):
        return tuple(_parse_attribute_value(item) for item in _split_top_level(text[1:-1]))
    if _INTEGER.fullmatch(text):
        return int(text.split(":", 1)[0].strip())
    if _FLOAT.fullmatch(text):
        return float(text.split(":", 1)[0].strip())
    return text


def _parse_attributes(text: str) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for opening, closing in (("<{", "}>"), ("{", "}")):
        cursor = 0
        while True:
            start = text.find(opening, cursor)
            if start < 0:
                break
            body, end = _balanced(text, start + len(opening) - 1, "{", "}")
            cursor = end + (1 if closing == "}>" else 0)
            for item in _split_top_level(body):
                if "=" not in item:
                    continue
                name, value = item.split("=", 1)
                attributes[name.strip()] = _parse_attribute_value(value)
    return attributes


def _location(line: str, locations: dict[str, str]) -> str | None:
    match = re.search(r"\bloc\((#[A-Za-z0-9_]+)\)\s*$", line)
    if not match:
        return None
    return locations.get(match.group(1)[1:], match.group(1))


def _result_names(text: str) -> tuple[str, ...]:
    names = tuple(_VALUE_NAME.findall(text))
    if not names:
        raise ValueError(f"ONNX operation has no SSA result in {text!r}")
    return names


def _parse_operator(line: str, locations: dict[str, str]) -> Operator | None:
    if " = " not in line:
        return None
    result_text, remainder = line.strip().split(" = ", 1)
    match = re.match(r'(?:"onnx\.([A-Za-z0-9_]+)"|onnx\.([A-Za-z0-9_]+))', remainder)
    if not match:
        return None
    op_type = match.group(1) or match.group(2)
    if op_type == "Constant":
        return None
    cursor = match.end()
    while cursor < len(remainder) and remainder[cursor].isspace():
        cursor += 1
    operands, cursor = _balanced(remainder, cursor, "(", ")")
    type_marker = remainder.find(" : (", cursor)
    if type_marker < 0:
        raise ValueError(f"ONNX {op_type} has no function type")
    input_body, after_inputs = _balanced(remainder, type_marker + 3, "(", ")")
    arrow = remainder.find("->", after_inputs)
    if arrow < 0:
        raise ValueError(f"ONNX {op_type} has no result type")
    result_text_type = remainder[arrow + 2 :]
    result_text_type = re.sub(r"\s+loc\(.*\)\s*$", "", result_text_type).strip()
    attributes = _parse_attributes(remainder[cursor:type_marker])
    domain = ""
    if op_type == "Custom" and (
        "function_name" in attributes or "domain_name" in attributes
    ):
        function_name = attributes.pop("function_name", None)
        domain_name = attributes.pop("domain_name", None)
        if not isinstance(function_name, str) or not isinstance(domain_name, str):
            raise ValueError("canonical ONNX Custom operation lacks function_name or domain_name")
        op_type = function_name
        domain = domain_name
    node_name = attributes.pop("onnx_node_name", None)
    return Operator(
        op_type=op_type,
        domain=domain,
        inputs=tuple(_VALUE_NAME.findall(operands)),
        outputs=_result_names(result_text),
        input_types=tuple(
            parse_tensor_type(item) for item in _split_top_level(input_body)
        ),
        output_types=_parse_type_list(result_text_type),
        attributes=attributes,
        node_name=node_name if isinstance(node_name, str) else None,
        location=_location(line, locations),
    )


def parse(text: str) -> Graph:
    """Parse one canonicalized ONNX entry graph from textual MLIR."""
    lines = text.splitlines()
    locations = {
        match.group(1): bytes(match.group(2), "utf-8").decode("unicode_escape")
        for line in lines
        if (match := _LOCATION.match(line))
    }
    function = next((_FUNC.match(line) for line in lines if _FUNC.match(line)), None)
    if function is None:
        raise ValueError("canonical ONNX MLIR has no func.func entry graph")
    function_name, argument_text, result_text = function.groups()
    inputs: list[Value] = []
    for argument in _split_top_level(argument_text):
        match = re.match(r"(%[A-Za-z0-9_.$-]+):\s*(tensor<[^>]+>)(.*)$", argument)
        if not match:
            raise ValueError(f"unsupported ONNX function argument {argument!r}")
        attributes = _parse_attributes(match.group(3))
        public_name = attributes.get("onnx.name")
        inputs.append(
            Value(
                str(public_name) if isinstance(public_name, str) else match.group(1)[1:],
                parse_tensor_type(match.group(2)),
                ssa_name=match.group(1),
            )
        )

    declared_outputs = _parse_type_list(result_text)
    initializers: list[Value] = []
    initializer_ssa: dict[str, Value] = {}
    structural_values: set[str] = set()
    operators: list[Operator] = []
    return_values: tuple[str, ...] | None = None
    for line_number, line in enumerate(lines, start=1):
        if "onnx.Constant" in line and " = " in line:
            result_name = line.strip().split(" = ", 1)[0]
            location = _location(line, locations)
            initializer = (
                location[len("Initializer_") :]
                if location and location.startswith("Initializer_")
                else None
            )
            type_match = re.search(r":\s*(tensor<[^>]+>)\s+loc\(", line)
            if initializer is None or type_match is None:
                raise ValueError(
                    f"line {line_number}: ONNX Constant lacks initializer name or tensor type"
                )
            value = Value(
                initializer,
                parse_tensor_type(type_match.group(1)),
                initializer,
                result_name,
            )
            initializers.append(value)
            initializer_ssa[result_name] = value
            continue
        if line.lstrip().startswith(("onnx.Return", "func.return", "return ")):
            return_values = tuple(_VALUE_NAME.findall(line.split(":", 1)[0]))
            continue
        try:
            operator = _parse_operator(line, locations)
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
        if operator is not None:
            if operator.op_type == "NoValue":
                structural_values.update(operator.outputs)
            else:
                operators.append(operator)

    if return_values is None:
        raise ValueError("canonical ONNX MLIR has no onnx.Return")
    if len(return_values) != len(declared_outputs):
        raise ValueError("ONNX return count does not match function result count")
    produced = {output for operator in operators for output in operator.outputs}
    arguments = set(re.findall(r"%[A-Za-z0-9_.$-]+", argument_text))
    known = produced | arguments | set(initializer_ssa) | structural_values
    for operator in operators:
        missing = [value for value in operator.inputs if value not in known]
        if missing:
            raise ValueError(
                f"ONNX {operator.op_type} references undefined SSA value {missing[0]}"
            )
    missing_returns = [value for value in return_values if value not in known]
    if missing_returns:
        raise ValueError(f"ONNX return references undefined SSA value {missing_returns[0]}")
    output_names = [f"output_{index}" for index in range(len(declared_outputs))]
    result_attributes = re.findall(r'onnx\.name\s*=\s*"([^"]+)"', result_text)
    if len(result_attributes) == len(output_names):
        output_names = result_attributes
    return Graph(
        name=function_name,
        inputs=tuple(inputs),
        outputs=tuple(
            Value(name, type_, ssa_name=ssa)
            for name, type_, ssa in zip(output_names, declared_outputs, return_values)
        ),
        initializers=tuple(initializers),
        operators=tuple(operators),
    )
