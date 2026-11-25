using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class TinyLlamaChat : UdonSharpBehaviour
{
    // The status the C Host returns from its exports (udononnxStatus_t).
    private const int OnnxSuccess = 0;

    [SerializeField]
    private UdonclTinyllamaChatBinding binding;

    [SerializeField]
    private string userText = "What is your name?";

    private bool initialized;
    private bool busy;
    private int[] textChunk;
    private int replyLength;
    private string reply;

    private void OnEnable()
    {
        // One frame later every behaviour, the binding included, is ready.
        SendCustomEventDelayedFrames(nameof(InitializeChat), 1);
    }

    private void OnDisable()
    {
        if (!initialized)
            return;
        initialized = false;
        // Dispose releases the model and ends the conversation; the next
        // OnEnable starts a new one.
        int onnxStatus;
        int status = binding.TryDispose(out onnxStatus);
        if (status != UdonclTinyllamaChatBinding.StatusSuccess || onnxStatus != OnnxSuccess)
            Debug.LogError(
                "TinyLlama: Dispose failed with status " + status + " and ONNX status " + onnxStatus
            );
    }

    public void InitializeChat()
    {
        if (initialized || busy)
            return;
        if (binding == null)
        {
            Debug.LogError("TinyLlama: assign the UdonclTinyllamaChatBinding instance.");
            return;
        }

        textChunk = new int[256];
        busy = true;
        // Initialize creates the OpenCL objects, the model session and the
        // ArgMax kernel in the C Host; it spans a few frames.
        int status = binding.StartInitialize(this, nameof(OnChatInitialized));
        if (status != UdonclTinyllamaChatBinding.StatusSuccess)
            Fail("StartInitialize", status, OnnxSuccess);
    }

    public void OnChatInitialized()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetInitializeStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetInitializeResult(out onnxStatus);
        if (!Succeeded("Initialize", status, onnxStatus))
            return;
        initialized = true;
        busy = false;
    }

    public override void Interact()
    {
        Send();
    }

    // Sends userText as the next message of the conversation.
    public void Send()
    {
        if (!initialized || busy)
            return;

        busy = true;
        int onnxStatus;
        int status = binding.TryBeginMessage(userText, out onnxStatus);
        if (!Succeeded("BeginMessage", status, onnxStatus))
            return;
        Debug.Log("TinyLlama: you: " + userText);
        EncodeMessage();
    }

    // The C Host applies the chat template and tokenizes the message, a
    // bounded amount of work per frame: ContinueMessage returns 1 while work
    // is pending and 0 once the message is ready.
    public void EncodeMessage()
    {
        int pending;
        int status = binding.TryContinueMessage(out pending);
        if (!Succeeded("ContinueMessage", status, pending < 0 ? -pending : OnnxSuccess))
            return;
        if (pending != 0)
        {
            SendCustomEventDelayedFrames(nameof(EncodeMessage), 1);
            return;
        }

        reply = "";
        replyLength = 0;
        status = binding.StartGenerate(this, nameof(OnReplyGenerated));
        if (!Succeeded("StartGenerate", status, OnnxSuccess))
            return;
        SendCustomEventDelayedFrames(nameof(StreamReply), 1);
    }

    // While the reply is generated, read the part of it that is final.
    public void StreamReply()
    {
        if (!busy)
            return;
        ReadReplyText();
        SendCustomEventDelayedFrames(nameof(StreamReply), 1);
    }

    public void OnReplyGenerated()
    {
        busy = false;
        int onnxStatus = OnnxSuccess;
        int status = binding.GetGenerateStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetGenerateResult(out onnxStatus);
        if (!Succeeded("Generate", status, onnxStatus))
            return;

        // After the generation completed the streamed text is the whole reply.
        ReadReplyText();
        Debug.Log("TinyLlama: assistant: " + reply);
    }

    // Appends the reply text that became final since the last call. The
    // binding returns UTF-16 code units, one per int, at most textChunk's
    // length per call.
    private void ReadReplyText()
    {
        int count = textChunk.Length;
        while (count == textChunk.Length)
        {
            int status = binding.TryReadStreamedText(textChunk, replyLength, out count);
            if (status != UdonclTinyllamaChatBinding.StatusSuccess || count < 0)
                return;
            for (int i = 0; i < count; i++)
                reply += (char)textChunk[i];
            replyLength += count;
        }
    }

    private bool Succeeded(string step, int status, int onnxStatus)
    {
        if (status == UdonclTinyllamaChatBinding.StatusSuccess && onnxStatus == OnnxSuccess)
            return true;
        Fail(step, status, onnxStatus);
        return false;
    }

    private void Fail(string step, int status, int onnxStatus)
    {
        busy = false;
        Debug.LogError(
            "TinyLlama: "
                + step
                + " failed with status "
                + status
                + " and ONNX status "
                + onnxStatus
        );
    }
}
