using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class TinyLlamaChat : UdonSharpBehaviour
{
    // The status the C Host returns from every export (udononnxStatus_t).
    private const int OnnxSuccess = 0;

    [SerializeField]
    private UdonclTinyLlamaChatBinding binding;

    [SerializeField]
    private string systemText = "You are a friendly assistant in a VRChat world.";

    [SerializeField]
    private string userText = "What is your name?";

    // At most the generated-token capacity the artifact was built with
    // (UDONCL_TINYLLAMA_MAX_NEW_TOKENS, 32 by default).
    [SerializeField]
    private int maxNewTokens = 32;

    // Tokenizer work per frame while a message is encoded, about one UTF-16
    // code unit each; a long message is spread over several frames.
    [SerializeField]
    private int promptWorkPerFrame = 32;

    private bool initialized;
    private bool busy;
    private bool conversationStarted;
    private bool encodingTurn;
    private bool generating;
    private int[] textChunk;
    private int replyLength;
    private string reply;

    private void Start()
    {
        // Initialize one frame after start-up so every behaviour is ready.
        SendCustomEventDelayedFrames("InitializeChat", 1);
    }

    public void InitializeChat()
    {
        if (binding == null)
        {
            Debug.LogError("TinyLlama: assign the UdonclTinyLlamaChatBinding instance.");
            return;
        }

        textChunk = new int[256];
        busy = true;
        // Initialize creates the OpenCL context and the ONNX session in the
        // C Host and uploads the model weights; it spans many frames.
        int status = binding.StartInitialize(this, "OnChatInitialized");
        if (status != UdonclTinyLlamaChatBinding.StatusSuccess)
            Fail("StartInitialize", status, OnnxSuccess);
    }

    public void OnChatInitialized()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetInitializeStatus();
        if (status == UdonclTinyLlamaChatBinding.StatusSuccess)
            status = binding.TryGetInitializeResult(out onnxStatus);
        if (!Succeeded("Initialize", status, onnxStatus))
            return;

        initialized = true;
        busy = false;
        Send();
    }

    public override void Interact()
    {
        Send();
    }

    // Sends userText: the first message starts the conversation with the
    // system text, every later one is appended to it as the next turn.
    public void Send()
    {
        if (!initialized || busy)
            return;

        busy = true;
        encodingTurn = conversationStarted;
        int onnxStatus;
        int status;
        if (encodingTurn)
            status = binding.TryBeginChatTurn(out onnxStatus);
        else
            status = binding.TryBeginChatPrompt(out onnxStatus);
        if (!Succeeded("BeginChatPrompt", status, onnxStatus))
            return;
        Debug.Log("TinyLlama: you: " + userText);
        EncodeMessage();
    }

    // The C Host applies the chat template and tokenizes the message, a
    // bounded amount of work per frame, until nothing is pending.
    public void EncodeMessage()
    {
        int onnxStatus;
        int status;
        if (encodingTurn)
            status = binding.TryContinueChatTurn(userText, promptWorkPerFrame, out onnxStatus);
        else
            status = binding.TryContinueChatPrompt(
                systemText,
                userText,
                promptWorkPerFrame,
                out onnxStatus
            );
        if (!Succeeded("ContinueChatPrompt", status, onnxStatus))
            return;

        int pending;
        status = binding.TryChatPromptPending(out pending);
        if (!Succeeded("ChatPromptPending", status, OnnxSuccess))
            return;
        if (pending != 0)
        {
            SendCustomEventDelayedFrames("EncodeMessage", 1);
            return;
        }

        conversationStarted = true;
        reply = "";
        replyLength = 0;
        status = binding.StartGenerate(maxNewTokens, this, "OnReplyGenerated");
        if (!Succeeded("StartGenerate", status, OnnxSuccess))
            return;
        generating = true;
        SendCustomEventDelayedFrames("StreamReply", 1);
    }

    // While the reply is generated, read the part of it that is final.
    public void StreamReply()
    {
        if (!generating)
            return;
        ReadReplyText();
        SendCustomEventDelayedFrames("StreamReply", 1);
    }

    public void OnReplyGenerated()
    {
        generating = false;
        int onnxStatus = OnnxSuccess;
        int status = binding.GetGenerateStatus();
        if (status == UdonclTinyLlamaChatBinding.StatusSuccess)
            status = binding.TryGetGenerateResult(out onnxStatus);
        if (!Succeeded("Generate", status, onnxStatus))
            return;

        // After the generation completed the streamed text is the whole reply.
        ReadReplyText();
        Debug.Log("TinyLlama: assistant: " + reply);
        busy = false;
    }

    // Appends the reply text that became final since the last call. The
    // binding returns UTF-16 code units, one per int.
    private void ReadReplyText()
    {
        int length;
        // StatusBusy while the model runs a step: try again next frame.
        if (binding.TryStreamedTextLength(out length) != UdonclTinyLlamaChatBinding.StatusSuccess)
            return;
        while (replyLength < length)
        {
            int count = Mathf.Min(length - replyLength, textChunk.Length);
            int onnxStatus;
            int status = binding.TryReadStreamedText(textChunk, replyLength, count, out onnxStatus);
            if (status != UdonclTinyLlamaChatBinding.StatusSuccess || onnxStatus != OnnxSuccess)
                return;
            for (int i = 0; i < count; i++)
                reply += (char)textChunk[i];
            replyLength += count;
        }
    }

    private bool Succeeded(string step, int status, int onnxStatus)
    {
        if (status == UdonclTinyLlamaChatBinding.StatusSuccess && onnxStatus == OnnxSuccess)
            return true;
        Fail(step, status, onnxStatus);
        return false;
    }

    private void Fail(string step, int status, int onnxStatus)
    {
        generating = false;
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
