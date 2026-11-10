# One conversation and C API per generated Behaviour

Each generated Behaviour owns exactly one ONNX session, Conversation History,
pending turn, response, and K/V state; multiple conversations use multiple
Behaviour instances rather than conversation handles. Its ordinary C API
exports initialize, reset, append, may-suspend generate, read-response, and
dispose, while compiler-generated U# adapts callbacks and `out string` without
putting U# receiver/event parameters in C. One pending turn, structured
immutable string history, and explicit lifecycle statuses keep storage and
callback ownership bounded and unambiguous.
