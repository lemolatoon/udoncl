# TinyLlama chat

A chat with TinyLlama-1.1B-Chat in a VRChat world, written as one Host C file
(`tinyllama_chat.c`) and one OpenCL C kernel (`argmax.cl`). UdonCL compiles
the ONNX model, the tokenizer and the Host into one Unity library whose U#
binding is `UdonclTinyllamaChatBinding`. The TinyLlama Chat Package Sample of
the Unity package (`TinyLlamaChat.cs`) drives it.

## Build

Download and verify the pinned model and tokenizer, then configure this
directory like any UdonCL CMake project:

```sh
python scripts/fetch_tinyllama_onnx.py
python scripts/fetch_tinyllama_tokenizer.py
cmake -S examples/tinyllama-chat -B build-chat -G Ninja \
  -DCMAKE_PREFIX_PATH=<UdonCL SDK> \
  -DUDONCL_TINYLLAMA_ONNX_MODEL=<model_q4.onnx> \
  -DUDONCL_TINYLLAMA_TOKENIZER_DIRECTORY=<tokenizer directory>
cmake --build build-chat --target tinyllama-chat
```

The fetch scripts print the paths. The artifact is
`build-chat/udoncl/tinyllama-chat/artifact`; import it with
**Tools > udoncl > Import Artifacts...**, which installs the program under
`Assets/udoncl-generated/tinyllama_chat`.

Two cache variables shape the artifact:

| Variable | Default | Meaning |
| --- | --- | --- |
| `UDONCL_TINYLLAMA_CONTEXT_CAPACITY` | 2048 | Tokens one conversation holds |
| `UDONCL_TINYLLAMA_MAX_NEW_TOKENS` | 32 | Most tokens of one reply |

The capacity bounds the model's past (`DIMENSION_BOUNDS
past_sequence_length=<capacity - 1>`) and is compiled into the Host as
`TINYLLAMA_CAPACITY`. TinyLlama was trained on 2048 positions.

## Exports

| C export | U# method | What it does |
| --- | --- | --- |
| `initialize()` | `StartInitialize` | Creates the OpenCL objects, the model session and the ArgMax kernel, and starts a new conversation |
| `begin_message(text)` | `TryBeginMessage(text)` | Starts tokenizing the next user message |
| `continue_message()` | `TryContinueMessage` | Tokenizes a bounded slice: 1 while work is pending, 0 once the message is ready, -status on failure |
| `generate()` | `StartGenerate` | Runs the message one token per run, then replies greedily until EOS or the reply limit |
| `read_streamed_text(destination, offset)` | `TryReadStreamedText` | Copies the reply text that is final so far (UTF-16 code units, one per `int`); also while `Generate` runs |
| `read_generated_tokens(destination)` | `TryReadGeneratedTokens` | Copies the token IDs of the last reply |
| `read_conversation_tokens(destination)` | `TryReadConversationTokens` | Copies the committed history of the conversation; returns its length or -status |
| `restore_conversation(tokens, count)` | `StartRestoreConversation` | Continues the conversation from a committed history, one token per run; `count` 0 starts a new one |
| `dispose()` | `TryDispose` | Releases everything; `Initialize` may run again |

The exports return a `udononnxStatus_t` (0 is success), except where the
table says otherwise. A U# behaviour follows the lifecycle of every UdonCL
example: `OnEnable` starts `Initialize`, a trigger such as `Interact` sends a
message (`BeginMessage`, `ContinueMessage` once per frame until it returns
0, `StartGenerate`, then `ReadStreamedText` once per frame), and `OnDisable`
calls `TryDispose`.

## How it works

- **One token per run.** The model is specialized to `batch_size=1` and
  `sequence_length=1`; every `udononnxRun` computes one position `t`. The
  message runs token by token, then the reply: the ArgMax kernel selects the
  next token from the logits, the Host reads that one word back and runs it
  at the next position. There is no batched prefill.
- **KV cache in place.** Each of the 22 layers keeps its K and V in one buffer
  of the whole capacity (`keys[22]`, `values[22]`, created, bound and released
  in loops). A run binds the past at `t` rows and the present at `t + 1` rows
  of the same buffer, so it writes only its own row (in-place Concat).
- **Multi-turn.** A conversation grows: a later message continues after the
  previous reply. The reply's last token, which no run computed, runs first,
  followed by EOS when the reply was cut off. `Dispose` followed by
  `Initialize` starts a new conversation.
- **Conversation sync.** The *committed history* is every token the KV cache
  holds followed by the tokens that close the last reply (its last token, and
  EOS when it was cut off): the whole conversation as the next message sees
  it. `ReadConversationTokens` copies it; `RestoreConversation` on another
  Host rebuilds that Host's KV cache from it, so its next message, reply and
  history are exactly those of the Host that read it. The restore keeps the
  leading tokens its KV cache already holds and runs the rest, one per run
  (up to 2048 runs for a full conversation); a shorter history only moves the
  position back. The closing tokens are in the KV cache after a restore, so
  the next message does not run them again. Only token IDs travel, never the
  KV cache itself (88 MiB at 2048 tokens).
- **Text.** The chat template and the tokenizer tables are compiled from the
  tokenizer files at build time (UdonTokenizers). A long message is spread
  over several `ContinueMessage` calls. The reply is decoded while it is
  generated; `ReadStreamedText` only ever returns text that can no longer
  change.

## Limits

- A message that does not fit the rest of the capacity fails with
  `CAPACITY_EXCEEDED` (13) and leaves the conversation unchanged; a reply ends
  early once the capacity is full. Its history no longer fits the capacity
  either: `ReadConversationTokens` then fails with `CAPACITY_EXCEEDED`.
- `RestoreConversation` rejects a history beyond the capacity with
  `CAPACITY_EXCEEDED` and a token outside the vocabulary with
  `INVALID_VALUE` (1), leaving the conversation unchanged. A successful restore
  drops a message that was tokenized but not generated, and the last reply's
  text and tokens.
- After a failed `Generate`, only `Dispose` and `Initialize` recover.
- Each graph tensor is bound from one export: the token, its position and the
  logits in `initialize`, the mask and the KV cache in `generate`.
