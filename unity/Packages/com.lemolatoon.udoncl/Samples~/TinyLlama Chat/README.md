# TinyLlama Chat

`TinyLlamaChat` chats with TinyLlama-1.1B-Chat through
`UdonclTinyllamaChatBinding`, the generated U# binding of the TinyLlama chat
example
([`examples/tinyllama-chat`](https://github.com/lemolatoon/udoncl/tree/master/examples/tinyllama-chat)).
Its one Host C file applies the chat template, tokenizes the messages, runs
the model on the GPU and decodes the reply, which the sample streams to the
Console while it is generated.

The sample follows the lifecycle of every UdonCL example:

- `OnEnable` starts `Initialize` one frame later.
- `Interact` sends **User Text**: `TryBeginMessage`, then `TryContinueMessage`
  once per frame until it returns 0, then `StartGenerate`. While the reply is
  generated, `TryReadStreamedText` reads the text that is final so far.
- `OnDisable` calls `TryDispose`, which ends the conversation; the next
  `OnEnable` starts a new one.

Every message continues the same conversation until it holds 2048 tokens;
then `ContinueMessage` fails with ONNX status 13 (`CAPACITY_EXCEEDED`).

The model is too large to ship with the package, so there is no one-click
import. Build the `tinyllama-chat` artifact from the real model and tokenizer
as described in `examples/tinyllama-chat/README.md`, then:

1. Import its `artifact` directory with **Tools → udoncl → Import
   Artifacts...** and choose **Add after Compile** to place the
   `UdonclTinyllamaChatBinding` prefab in the scene.
2. Add `TinyLlamaChat` to a GameObject and assign the
   `UdonclTinyllamaChatBinding` scene instance to its **Binding** field. Edit
   **User Text** as you like.
3. Enter Play Mode and interact with the object to send **User Text**; the
   reply is logged to the Console. Interact again to send the next turn of
   the same conversation.
