# TinyLlama Chat

`TinyLlamaChat` chats with TinyLlama-1.1B-Chat through
`UdonclTinyllamaChatBinding`, the generated U# binding of the TinyLlama chat
example
([`examples/tinyllama-chat`](https://github.com/lemolatoon/udoncl/tree/master/examples/tinyllama-chat)).
Its one Host C file applies the chat template, tokenizes the messages, runs
the model on the GPU and decodes the reply, which the sample streams to the
Console (and an optional UI **Log Text**) while it is generated. Every player
of an instance shares one conversation.

The sample follows the lifecycle of every UdonCL example:

- `OnEnable` starts `Initialize` one frame later.
- `Interact` sends **User Text** (see below).
- `OnDisable` calls `TryDispose`, which releases the model; the shared
  conversation stays.

## Sharing the conversation

The behaviour uses `BehaviourSyncMode.Manual` and syncs three fields: the
conversation's committed history (`conversationTokens`, at most 2048 token
IDs, from `TryReadConversationTokens`), its text log (`conversationLog`), and
a `busy` flag. The KV cache itself (88 MB) is never synced; the player who
sends a message rebuilds it from the tokens and generates the reply on their
own GPU:

1. `Send` (from `Interact`) does nothing while `busy` is set: another player's
   reply is being generated.
2. It takes ownership (`Networking.SetOwner`), sets `busy` and serializes it.
3. `StartRestoreConversation` restores the synced history into this player's
   Host. Only the tokens after what the Host already holds run, so the player
   who sent the last message restores almost nothing; a player who never sent
   one runs the whole history, one token per model run.
4. `TryBeginMessage`, `TryContinueMessage` once per frame until it returns 0,
   then `StartGenerate`; `TryReadStreamedText` reads the reply while it is
   generated.
5. Once the reply is complete, `TryReadConversationTokens` reads the new
   history. The sample stores it and the extended log, clears `busy` and calls
   `RequestSerialization` once.

The other players never run their Host for a message: `OnDeserialization`
shows the synced log. If the player generating a reply leaves, the next owner
clears `busy` (`OnOwnershipTransferred`); the reply is dropped and the last
synced history stays. A failed step also drops the reply. **Clear** (call
`Clear`, for example from a UI button) syncs an empty history and log, so the
next message starts a new conversation. Alone in an instance, the sample
behaves as a plain chat: every message continues the same conversation.

A conversation holds 2048 tokens. A message that does not fit fails with ONNX
status 13 (`CAPACITY_EXCEEDED`); a reply that fills the conversation is shown
to its sender but not synced. Clear starts over.

Real multi-player sync is checked by hand only. The automated `tinychat` gate
covers the Host side (it reads the history after the first turn, disposes,
initializes and restores it, and the second turn's reply must match the
oracle) and compiles this sample, but no test runs two VRChat clients.

## Setup

The model is too large to ship with the package, so there is no one-click
import. Build the `tinyllama-chat` artifact from the real model and tokenizer
as described in `examples/tinyllama-chat/README.md`, then:

1. Import its `artifact` directory with **Tools → udoncl → Import
   Artifacts...** and choose **Add after Compile** to place the
   `UdonclTinyllamaChatBinding` prefab in the scene.
2. Add `TinyLlamaChat` to a GameObject and assign the
   `UdonclTinyllamaChatBinding` scene instance to its **Binding** field. Edit
   **User Text** as you like, and optionally assign a UI `Text` to **Log
   Text**.
3. Enter Play Mode and interact with the object to send **User Text**; the
   reply is logged to the Console. Interact again to send the next turn of
   the same conversation. To try the sync, build and test the world with two
   clients (**Build & Test** with a client count of 2) and send from either.
