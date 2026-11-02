using Lemolatoon.Udoncl;
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public sealed class UdonOnnxTinyLlamaExample : UdonSharpBehaviour
{
    [SerializeField]
    private UdonOnnxRuntimeBehaviour runtime;

    [SerializeField]
    [Range(0, 15)]
    private int tokenId = 3;

    private int predictedToken = -1;
    private int status;
    private float[] logits;

    public override void Interact()
    {
        Run();
    }

    public void Run()
    {
        int inputIndex = runtime.FindInput("token_ids");
        int outputIndex = runtime.FindOutput("logits");
        if (inputIndex < 0 || outputIndex < 0)
        {
            status = UdonOnnxRuntimeBehaviour.StatusInvalidArgument;
            return;
        }

        int[] tokens = new int[1];
        tokens[0] = tokenId;
        logits = new float[16];
        status = runtime.TryWriteInputInt(inputIndex, tokens);
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        status = runtime.TryRun();
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        status = runtime.StartReadOutputFloat(
            outputIndex,
            logits,
            this,
            nameof(OnReadbackComplete)
        );
    }

    public void OnReadbackComplete()
    {
        status = runtime.GetLastReadbackStatus();
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        predictedToken = 0;
        for (int index = 1; index < logits.Length; ++index)
            if (logits[index] > logits[predictedToken])
                predictedToken = index;
    }

    public int GetStatus()
    {
        return status;
    }

    public int GetPredictedToken()
    {
        return predictedToken;
    }
}
