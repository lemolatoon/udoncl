"""Interpret the generated dispatchers of a finalized ONNX Host module.

The ONNX Host lowering emits udononnxGeneratedCreate, BindMemory and Run as
Host MLIR that becomes Udon. This module runs those functions on the
generic-form text of the finalized module (`--mlir-print-op-generic`), with
the Device and OpenCL calls they make recorded instead of executed:

- every `__vrc_opencl_dispatch_<kernel>` call becomes a dispatch of that
  kernel with its raw 32-bit arguments (buffer handles and scalars, in kernel
  argument order);
- `__vrc_opencl_static_tensor` returns the handle of a static tensor whose
  words come from the module's `udoncl.static_tensors`;
- buffer copies, fills and uploads are recorded too;
- buffers the Host creates get fresh handles.

The recorded trace is what the native execution tests run on the native
fragment renderer (synthesized kernels) and on the CPU builds of the library
kernels, so the scalars the Host computes for a run (extents, pitches,
strides) are the ones under test. Only the subset of arith, cf and udonhl
the generated dispatchers use is implemented; anything else raises.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field


class InterpreterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pointer:
    kind: str  # null, global, alloca, buffer, opaque
    key: object = None
    offset: int = 0

    def advanced(self, bytes_: int) -> "Pointer":
        if self.kind not in ("global", "alloca"):
            raise InterpreterError(f"pointer arithmetic on {self}")
        return Pointer(self.kind, self.key, self.offset + bytes_)


NULL = Pointer("null")


@dataclass
class Operation:
    results: list
    name: str
    operands: list
    successors: list
    properties: str
    result_types: list
    operand_types: list
    line: str


@dataclass
class Block:
    arguments: list
    operations: list = field(default_factory=list)


@dataclass
class Function:
    name: str
    blocks: dict
    order: list


def _matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quoted = False
    for index in range(start, len(text)):
        character = text[index]
        if character == '"' and text[index - 1] != "\\":
            quoted = not quoted
        if quoted:
            continue
        if character in opening:
            depth += 1
        elif character in closing:
            depth -= 1
            if depth == 0:
                return index
    raise InterpreterError(f"unbalanced {opening}{closing}: {text[start:start + 80]}")


def _split_top(text: str) -> list:
    parts, depth, current, quoted = [], 0, [], False
    for character in text:
        if character == '"':
            quoted = not quoted
        if not quoted:
            if character in "([{<":
                depth += 1
            elif character in ")]}>":
                depth -= 1
            elif character == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
        current.append(character)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def _parse_operation(line: str) -> Operation:
    stripped = line.strip()
    results = []
    match = re.match(r'^(%[^=]+?) = "', stripped)
    if match:
        for result in match.group(1).split(","):
            result = result.strip()
            if ":" in result:
                name, count = result.split(":")
                results.extend(f"{name}#{index}" for index in range(int(count)))
            else:
                results.append(result)
        stripped = stripped[match.end() - 1 :]
    name_end = stripped.index('"', 1)
    name = stripped[1:name_end]
    position = name_end + 1
    close = _matching(stripped, position, "(", ")")
    operands = [value for value in _split_top(stripped[position + 1 : close])]
    position = close + 1
    successors = []
    if stripped[position : position + 1] == "[":
        close = _matching(stripped, position, "[", "]")
        successors = _split_top(stripped[position + 1 : close])
        position = close + 1
    properties = ""
    if stripped[position : position + 3] == " <{":
        close = _matching(stripped, position + 1, "<", ">")
        properties = stripped[position + 3 : close - 1]
        position = close + 1
    if stripped[position : position + 2] == " {":
        close = _matching(stripped, position + 1, "{", "}")
        position = close + 1
    signature = stripped[position:].strip()
    if not signature.startswith(": ("):
        raise InterpreterError(f"unsupported operation syntax: {line.strip()}")
    close = _matching(signature, 2, "(", ")")
    operand_types = _split_top(signature[3:close])
    result_text = signature[close + 1 :].strip()
    if not result_text.startswith("->"):
        raise InterpreterError(f"unsupported operation syntax: {line.strip()}")
    result_text = result_text[2:].strip()
    if result_text.startswith("("):
        result_types = _split_top(result_text[1:-1])
    else:
        result_types = [result_text]
    return Operation(
        results,
        name,
        operands,
        successors,
        properties,
        result_types,
        operand_types,
        line.strip(),
    )


def _property(operation: Operation, key: str):
    for part in _split_top(operation.properties):
        if part.startswith(f"{key} = "):
            return part[len(key) + 3 :]
    return None


def _integer_attribute(text: str) -> int:
    return int(text.split(":")[0].strip(), 0)


def _width(type_: str) -> int:
    if type_ == "index":
        return 64
    match = re.fullmatch(r"i(\d+)", type_)
    if not match:
        raise InterpreterError(f"not an integer type: {type_}")
    return int(match.group(1))


def _wrap(value: int, width: int) -> int:
    value &= (1 << width) - 1
    if width > 1 and value >> (width - 1):
        value -= 1 << width
    return value


def _unsigned(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def _block_label(line: str):
    match = re.match(r"^\s*(\^bb\d+)(?:\((.*)\))?:", line)
    if not match:
        return None
    arguments = []
    if match.group(2):
        arguments = [part.split(":")[0].strip() for part in _split_top(match.group(2))]
    return match.group(1), arguments


def _parse_functions(lines: list) -> dict:
    functions = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if '"func.func"()' in line and line.rstrip().endswith("({"):
            name = re.search(r'sym_name = "([^"]+)"', line).group(1)
            indent = len(line) - len(line.lstrip())
            body = []
            index += 1
            while not (
                lines[index].startswith(" " * indent + "})")
                and len(lines[index]) - len(lines[index].lstrip()) == indent
            ):
                body.append(lines[index])
                index += 1
            blocks, order = {}, []
            current = None
            for body_line in body:
                if not body_line.strip():
                    continue
                label = _block_label(body_line)
                if label:
                    current = Block(label[1])
                    blocks[label[0]] = current
                    order.append(label[0])
                    continue
                if current is None:
                    current = Block([])
                    blocks["^entry"] = current
                    order.append("^entry")
                if body_line.strip().startswith('"') or body_line.strip().startswith("%"):
                    current.operations.append(_parse_operation(body_line))
                else:
                    raise InterpreterError(f"unsupported line: {body_line.strip()}")
            functions[name] = Function(name, blocks, order)
        index += 1
    return functions


def _dense_values(text: str):
    text = text.strip()
    if text.startswith('"0x'):
        return bytes.fromhex(text[3:-1])
    numbers = re.findall(r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?|inf|nan)", text)
    if "true" in text or "false" in text:
        return [1 if word == "true" else 0 for word in re.findall(r"true|false", text)]
    return [float(number) if re.search(r"[.eEn]", number) else int(number) for number in numbers]


def static_tensor_words(module_text: str) -> dict:
    """The Device words of every static tensor of the module, by id."""
    tensors = {}
    start = module_text.find("udoncl.static_tensors = [")
    if start < 0:
        return tensors
    start = module_text.index("[", start)
    end = _matching(module_text, start, "[", "]")
    for entry in _split_top(module_text[start + 1 : end]):
        entry = entry.strip()[1:-1]
        fields = {}
        for part in _split_top(entry):
            key, _, value = part.partition(" = ")
            fields[key.strip()] = value.strip()
        data = fields["data"]
        dense = data[data.index("<") + 1 : _matching(data, data.index("<"), "<", ">")]
        shape_text = re.search(r"tensor<([^>]*)>", data[data.rindex(":") :]).group(1)
        dims = [int(value) for value in shape_text.split("x")[:-1]]
        count = 1
        for dim in dims:
            count *= dim
        data_type = fields["data_type"].strip('"')
        values = _dense_values(dense)
        if isinstance(values, bytes):
            raw = values
        else:
            if len(values) == 1 and count != 1:
                values = values * count
            if len(values) != count:
                raise InterpreterError(f"static tensor {fields['id']}: {len(values)} != {count}")
            if data_type == "float32":
                raw = struct.pack(f"<{count}f", *values)
            elif data_type == "int64":
                raw = struct.pack(f"<{count}q", *values)
            elif data_type == "int32":
                raw = struct.pack(f"<{count}i", *values)
            elif data_type == "bool":
                raw = struct.pack(f"<{count}f", *[float(value) for value in values])
            else:
                raise InterpreterError(f"static tensor data type {data_type}")
        raw += b"\0" * (-len(raw) % 4)
        tensors[_integer_attribute(fields["id"])] = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return tensors


class Host:
    """The generated dispatchers of one finalized ONNX Host module."""

    def __init__(self, generic_text: str):
        self.text = generic_text
        lines = generic_text.splitlines()
        self.functions = _parse_functions(lines)
        self.globals = {}
        for line in lines:
            if '"udonhl.global"()' not in line:
                continue
            name = re.search(r'sym_name = "([^"]+)"', line).group(1)
            initial = re.search(r"initial_value = (-?\d+) : i\d+", line)
            pointer = "storage_type = !udonhl.ptr" in line
            if pointer:
                self.globals[name] = NULL
            elif initial:
                self.globals[name] = int(initial.group(1))
        self.static_tensors = static_tensor_words(generic_text)
        self.memory = {}
        self.allocas = 0
        self.next_handle = 1
        self.buffers = {}  # handle -> words
        self.static_handles = {}
        self.trace = []

    # Buffers the caller binds or the Host creates.
    def create_buffer(self, words: int) -> Pointer:
        handle = self.next_handle
        self.next_handle += 1
        self.buffers[handle] = words
        self.trace.append(("buffer", handle, words))
        return Pointer("buffer", handle)

    def call(self, name: str, *arguments):
        return self._run(self.functions[name], list(arguments))

    # Memory.
    def _load(self, pointer: Pointer):
        if pointer.kind == "global":
            if pointer.offset:
                return self.memory.get((pointer.key, pointer.offset), 0)
            return self.globals.get(pointer.key, 0)
        if pointer.kind == "alloca":
            return self.memory.get((pointer.key, pointer.offset), 0)
        raise InterpreterError(f"load from {pointer}")

    def _store(self, value, pointer: Pointer):
        if pointer.kind == "global" and not pointer.offset:
            self.globals[pointer.key] = value
        elif pointer.kind in ("global", "alloca"):
            self.memory[(pointer.key, pointer.offset)] = value
        else:
            raise InterpreterError(f"store to {pointer}")

    def _external(self, callee: str, arguments: list, result_types: list):
        if callee.startswith("__vrc_opencl_dispatch_"):
            words = []
            for argument in arguments:
                if isinstance(argument, float):
                    words.append(struct.unpack("<I", struct.pack("<f", argument))[0])
                else:
                    words.append(_unsigned(argument, 32))
            self.trace.append(("dispatch", callee[len("__vrc_opencl_dispatch_") :], words))
            return 1
        if callee == "__vrc_opencl_static_tensor":
            identifier, _ = arguments
            if identifier not in self.static_handles:
                pointer = self.create_buffer(len(self.static_tensors[identifier]))
                self.trace.append(("words", pointer.key, self.static_tensors[identifier]))
                self.static_handles[identifier] = pointer.key
            return self.static_handles[identifier]
        if callee == "__vrc_opencl_await":
            return 0
        if callee == "__vrc_opencl_buffer_copy":
            source, destination, source_offset, destination_offset, size = arguments
            self.trace.append(("copy", source, destination, source_offset, destination_offset, size))
            return 1
        if callee == "udonclCreateBufferForQueue":
            _, _, size, error = arguments
            self._store(0, error)
            return self.create_buffer((size + 3) // 4)
        if callee == "udonclResolvePhysicalBuffer":
            memory, out = arguments
            if memory.kind != "buffer":
                return 5
            self._store(memory.key, out)
            return 0
        if callee in (
            "clEnqueueMarkerWithWaitList",
            "clFlush",
            "clFinish",
            "clWaitForEvents",
            "clReleaseEvent",
            "clReleaseMemObject",
            "udonclDeviceBufferDestroy",
        ):
            return 0
        raise InterpreterError(f"unsupported external call {callee}")

    def _run(self, function: Function, arguments: list):
        environment = {}
        label = function.order[0]
        block = function.blocks[label]
        if len(block.arguments) != len(arguments):
            raise InterpreterError(f"{function.name}: {len(arguments)} arguments")
        environment.update(zip(block.arguments, arguments))
        steps = 0
        while True:
            for operation in block.operations:
                steps += 1
                if steps > 1_000_000:
                    raise InterpreterError(f"{function.name} does not terminate")
                values = [environment[name] for name in operation.operands]
                name = operation.name
                if name == "func.return":
                    return values[0] if values else None
                if name == "cf.br":
                    target = operation.successors[0]
                    block = function.blocks[target]
                    environment.update(zip(block.arguments, values))
                    break
                if name == "cf.cond_br":
                    sizes = [int(value) for value in re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", operation.properties).group(1).split(",")]
                    condition = values[0]
                    true_values = values[1 : 1 + sizes[1]]
                    false_values = values[1 + sizes[1] :]
                    if condition:
                        block = function.blocks[operation.successors[0]]
                        environment.update(zip(block.arguments, true_values))
                    else:
                        block = function.blocks[operation.successors[1]]
                        environment.update(zip(block.arguments, false_values))
                    break
                if name == "cf.switch":
                    cases = [int(value) for value in re.search(r"case_values = dense<\[?([^\]>]*)\]?>", operation.properties).group(1).split(",")]
                    flag = values[0]
                    target = operation.successors[0]
                    for case, successor in zip(cases, operation.successors[1:]):
                        if case == flag:
                            target = successor
                    if function.blocks[target].arguments:
                        raise InterpreterError("cf.switch with block arguments")
                    block = function.blocks[target]
                    break
                result = self._evaluate(operation, values)
                if operation.results:
                    environment[operation.results[0]] = result
            else:
                raise InterpreterError(f"{function.name}: block without terminator")

    def _evaluate(self, operation: Operation, values: list):
        name = operation.name
        result_type = operation.result_types[0] if operation.result_types else ""
        if name == "arith.constant":
            text = _property(operation, "value")
            literal, _, type_ = text.rpartition(" : ")
            if type_ in ("f32", "f64"):
                if literal.startswith("0x"):
                    return struct.unpack("<f", struct.pack("<I", int(literal, 16)))[0]
                return float(literal)
            if literal in ("true", "false"):
                return int(literal == "true")
            return _wrap(int(literal, 0), _width(type_))
        if name in ("udonhl.address_of",):
            return Pointer("global", _property(operation, "global").lstrip("@"))
        if name == "udonhl.load":
            return self._load(values[0])
        if name == "udonhl.store":
            self._store(values[0], values[1])
            return None
        if name == "udonhl.alloca":
            self.allocas += 1
            return Pointer("alloca", self.allocas)
        if name == "udonhl.inttoptr":
            return NULL if values[0] == 0 else Pointer("opaque", values[0])
        if name == "udonhl.ptrtoint":
            pointer = values[0]
            if pointer.kind == "null":
                return 0
            if pointer.kind in ("buffer", "opaque"):
                return pointer.key
            raise InterpreterError(f"ptrtoint of {pointer}")
        if name == "func.call":
            callee = _property(operation, "callee").lstrip("@")
            if callee in self.functions and self.functions[callee].order:
                return self._run(self.functions[callee], values)
            return self._external(callee, values, operation.result_types)
        if name.startswith("arith."):
            return self._arith(name[6:], values, operation, result_type)
        raise InterpreterError(f"unsupported operation {operation.line}")

    def _arith(self, name: str, values: list, operation: Operation, result_type: str):
        if name == "select":
            return values[1] if values[0] else values[2]
        if name == "cmpi":
            predicate = _integer_attribute(_property(operation, "predicate"))
            width = _width(operation.operand_types[0])
            left, right = values
            if predicate >= 6:
                left, right = _unsigned(left, width), _unsigned(right, width)
            return int(
                [
                    left == right,
                    left != right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                ][predicate]
            )
        if name in ("extsi", "trunci", "index_cast", "index_castui"):
            return _wrap(values[0], _width(result_type))
        if name == "extui":
            return _unsigned(values[0], _width(operation.operand_types[0]))
        if name in ("sitofp", "uitofp"):
            return float(values[0])
        width = _width(result_type)
        left, right = values
        unsigned_left, unsigned_right = _unsigned(left, width), _unsigned(right, width)
        if name == "addi":
            value = left + right
        elif name == "subi":
            value = left - right
        elif name == "muli":
            value = left * right
        elif name == "divsi":
            value = abs(left) // abs(right) * (1 if (left < 0) == (right < 0) else -1)
        elif name == "divui":
            value = unsigned_left // unsigned_right
        elif name == "remsi":
            value = left - right * (abs(left) // abs(right) * (1 if (left < 0) == (right < 0) else -1))
        elif name == "remui":
            value = unsigned_left % unsigned_right
        elif name == "andi":
            value = left & right
        elif name == "ori":
            value = left | right
        elif name == "xori":
            value = left ^ right
        elif name == "shli":
            value = left << unsigned_right
        elif name == "shrui":
            value = unsigned_left >> unsigned_right
        elif name == "shrsi":
            value = left >> unsigned_right
        elif name == "minsi":
            value = min(left, right)
        elif name == "maxsi":
            value = max(left, right)
        elif name == "minui":
            value = min(unsigned_left, unsigned_right)
        elif name == "maxui":
            value = max(unsigned_left, unsigned_right)
        else:
            raise InterpreterError(f"unsupported arith.{name}")
        return _wrap(value, width)
