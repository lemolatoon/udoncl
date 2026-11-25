using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;

// One conversation shared by every player of the instance. The player who
// sends the next message generates its reply on their own GPU: they take
// ownership, restore the synced history into their Host, generate, and sync
// the new history and text log once. Everyone else only shows the log.
[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class TinyLlamaChat : UdonSharpBehaviour
{
    // The status the C Host returns from its exports (udononnxStatus_t).
    private const int OnnxSuccess = 0;
    private const int OnnxCapacityExceeded = 13;

    // UDONCL_TINYLLAMA_CONTEXT_CAPACITY of the artifact: the longest history.
    private const int Capacity = 2048;

    [SerializeField]
    private UdonclTinyllamaChatBinding binding;

    [SerializeField]
    private string userText = "What is your name?";

    // Optional: shows the conversation log.
    [SerializeField]
    private Text logText;

    // The committed history of the conversation (ReadConversationTokens),
    // its text, and whether a player is generating the next reply.
    [UdonSynced]
    private int[] conversationTokens = new int[0];

    [UdonSynced]
    private string conversationLog = "";

    [UdonSynced]
    private bool busy;

    private bool initialized;

    // This player's Host is initializing, restoring or replying.
    private bool working;
    private string message;
    private int[] textChunk;
    private int[] historyChunk;
    private int replyLength;
    private string reply;
    private int shownLength;

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
        // Dispose releases the model and this player's copy of the
        // conversation; the synced history stays.
        int onnxStatus;
        int status = binding.TryDispose(out onnxStatus);
        if (status != UdonclTinyllamaChatBinding.StatusSuccess || onnxStatus != OnnxSuccess)
            Debug.LogError(
                "TinyLlama: Dispose failed with status " + status + " and ONNX status " + onnxStatus
            );
    }

    public void InitializeChat()
    {
        if (initialized || working)
            return;
        if (binding == null)
        {
            Debug.LogError("TinyLlama: assign the UdonclTinyllamaChatBinding instance.");
            return;
        }

        textChunk = new int[256];
        historyChunk = new int[Capacity];
        working = true;
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
        working = false;
        ShowLog();
    }

    public override void Interact()
    {
        Send();
    }

    // Sends userText as the next message of the shared conversation.
    public void Send()
    {
        if (!initialized || working)
            return;
        if (busy)
        {
            Debug.Log("TinyLlama: another player's reply is being generated.");
            return;
        }

        // Take the conversation: the busy flag keeps the other players from
        // sending until this reply is synced.
        working = true;
        Networking.SetOwner(Networking.LocalPlayer, gameObject);
        busy = true;
        RequestSerialization();
        message = userText;

        // Rebuild this player's KV cache from the synced history. Only what
        // follows the tokens the Host already holds runs, so the player who
        // sent the last message restores almost nothing.
        int status = binding.StartRestoreConversation(
            conversationTokens,
            conversationTokens.Length,
            this,
            nameof(OnConversationRestored)
        );
        Succeeded("StartRestoreConversation", status, OnnxSuccess);
    }

    public void OnConversationRestored()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetRestoreConversationStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetRestoreConversationResult(out onnxStatus);
        if (!Succeeded("RestoreConversation", status, onnxStatus))
            return;

        status = binding.TryBeginMessage(message, out onnxStatus);
        if (!Succeeded("BeginMessage", status, onnxStatus))
            return;
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
        if (!working)
            return;
        ReadReplyText();
        if (logText != null)
            logText.text = conversationLog + Turn();
        SendCustomEventDelayedFrames(nameof(StreamReply), 1);
    }

    public void OnReplyGenerated()
    {
        int onnxStatus = OnnxSuccess;
        int status = binding.GetGenerateStatus();
        if (status == UdonclTinyllamaChatBinding.StatusSuccess)
            status = binding.TryGetGenerateResult(out onnxStatus);
        if (!Succeeded("Generate", status, onnxStatus))
            return;

        // After the generation completed the streamed text is the whole reply.
        ReadReplyText();

        // The new history: the old one, the message and the reply.
        int count;
        status = binding.TryReadConversationTokens(historyChunk, out count);
        if (status == UdonclTinyllamaChatBinding.StatusSuccess && count == -OnnxCapacityExceeded)
        {
            // The reply filled the conversation: no message can follow it,
            // so it is not synced.
            Debug.LogWarning(
                "TinyLlama: the conversation is full; Clear starts a new one." + Turn()
            );
            Release();
            return;
        }
        if (!Succeeded("ReadConversationTokens", status, count < 0 ? -count : OnnxSuccess))
            return;
        if (!Networking.IsOwner(gameObject))
        {
            // Another player sent a message at the same time and took the
            // conversation; theirs is synced.
            Debug.Log("TinyLlama: another player took the conversation." + Turn());
            working = false;
            return;
        }

        int[] tokens = new int[count];
        for (int i = 0; i < count; i++)
            tokens[i] = historyChunk[i];
        conversationTokens = tokens;
        conversationLog = conversationLog + Turn();
        busy = false;
        working = false;
        RequestSerialization();
        ShowLog();
    }

    // Starts a new conversation for everyone: an empty history and log.
    public void Clear()
    {
        if (busy || working)
            return;
        Networking.SetOwner(Networking.LocalPlayer, gameObject);
        conversationTokens = new int[0];
        conversationLog = "";
        RequestSerialization();
        ShowLog();
    }

    public override void OnDeserialization()
    {
        ShowLog();
    }

    // When the player generating a reply leaves, another player becomes the
    // owner with the busy flag still set: the reply is dropped and the last
    // synced history stays.
    public override void OnOwnershipTransferred(VRCPlayerApi player)
    {
        if (!Utilities.IsValid(player) || !player.isLocal || working || !busy)
            return;
        busy = false;
        RequestSerialization();
    }

    // The log lines of this player's message and the reply so far.
    private string Turn()
    {
        VRCPlayerApi player = Networking.LocalPlayer;
        string name = Utilities.IsValid(player) ? player.displayName : "you";
        return (conversationLog.Length == 0 ? "" : "\n")
            + name
            + ": "
            + message
            + "\nTinyLlama: "
            + reply;
    }

    // Logs the part of the synced log this player has not seen yet.
    private void ShowLog()
    {
        if (conversationLog.Length < shownLength)
            Debug.Log("TinyLlama: the conversation was cleared.");
        else if (conversationLog.Length > shownLength)
            Debug.Log("TinyLlama: " + conversationLog.Substring(shownLength));
        shownLength = conversationLog.Length;
        if (logText != null)
            logText.text = conversationLog;
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

    // A failed step drops the reply: the synced history stays as it was.
    private void Fail(string step, int status, int onnxStatus)
    {
        Release();
        Debug.LogError(
            "TinyLlama: "
                + step
                + " failed with status "
                + status
                + " and ONNX status "
                + onnxStatus
        );
    }

    // Lets the other players send again.
    private void Release()
    {
        working = false;
        if (busy && Networking.IsOwner(gameObject))
        {
            busy = false;
            RequestSerialization();
        }
    }
}
