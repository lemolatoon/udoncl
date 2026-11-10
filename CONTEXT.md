# UdonCL Domain Language

This glossary defines the project-specific language used across UdonCL's AOT
compiler, runtime libraries, generated Unity bindings, and examples.

## ONNX Conversation Runtime

**Conversation Instance**:
A single generated Behaviour's ONNX session together with its one conversation state, including token history and K/V state.
_Avoid_: Conversation handle, session handle

**Prefill Validation Artifact**:
An acceptance artifact that exposes every model input and output needed to compare one prefill with an independent oracle.
_Avoid_: Conversation runtime, production binding

**Conversation Artifact**:
The user-facing artifact that exposes conversation operations while keeping model tensors and physical buffers private.
_Avoid_: Prefill binding, tensor session

**Pending User Turn**:
The immutable user text accepted by a Conversation Instance but not yet incorporated into a generated assistant response.
_Avoid_: Prompt buffer, pending tokens

**Latest Assistant Response**:
The assistant text produced by the most recently completed generation, retained independently of the internal conversation history until replacement or reset.
_Avoid_: Transcript, full conversation

**Committed Conversation State**:
The token history, K/V state, and Latest Assistant Response from the most recently completed turn.
_Avoid_: Current buffers, live session state

**Working Turn**:
The provisional tokens, K/V state, and response being produced for a Pending User Turn before successful commit.
_Avoid_: Partial conversation, current response

**Context Capacity**:
The build-time maximum token history, including the requested generated tokens, that a Conversation Artifact can retain.
_Avoid_: Current context length, model context window

**Conversation History**:
The ordered immutable user and assistant turn references committed by a Conversation Instance.
_Avoid_: Concatenated transcript, prompt string

**Tokenizer Model**:
The immutable vocabulary, merge rules, byte-fallback mapping, special tokens, and chat-template data compiled for one tokenizer.
_Avoid_: Tokenizer session, runtime JSON

**Tokenizer Workspace**:
Caller-owned bounded storage used while encoding text or incrementally decoding tokens.
_Avoid_: Tokenizer global state, tokenizer heap

**Logical Tensor Binding**:
A model input or output identified by its generated name, dtype, logical shape, and capacity independently of its physical Device storage.
_Avoid_: Buffer slot, texture binding
