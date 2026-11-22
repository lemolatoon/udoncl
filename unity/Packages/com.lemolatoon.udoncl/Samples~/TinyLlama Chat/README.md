# TinyLlama Chat

`TinyLlamaChat` chats with TinyLlama-1.1B-Chat through the generated U#
binding of the TinyLlama Chat Artifact, `UdonclTinyLlamaChatBinding`. The C
Host of
[`examples/onnx-transformer-tiny`](https://github.com/lemolatoon/udoncl/tree/master/examples/onnx-transformer-tiny)
applies the chat template, tokenizes the messages, runs the model on the GPU
and decodes the reply, which the sample streams to the Console while it is
generated.

The model is too large to ship with the package, so there is no one-click
import. Build the 4096-token Chat Artifact (`onnx-transformer-tiny-chat-4k`)
from the real model and tokenizer as described in
`examples/onnx-transformer-tiny/README.md`, then:

1. Import its `artifact` directory with **Tools → udoncl → Import
   Artifacts...** and choose **Add after Compile** to place the
   `UdonclTinyLlamaChatBinding` prefab in the scene.
2. Add `TinyLlamaChat` to a GameObject and assign the
   `UdonclTinyLlamaChatBinding` scene instance to its **Binding** field. Edit
   **System Text** and **User Text** as you like.
3. Enter Play Mode. After the model is initialized the sample sends
   **User Text** and logs the reply. Interact with the object to send **User
   Text** again as the next turn of the same conversation.
