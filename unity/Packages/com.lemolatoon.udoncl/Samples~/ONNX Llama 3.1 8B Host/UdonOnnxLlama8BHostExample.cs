using Lemolatoon.Udoncl;
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public sealed class UdonOnnxLlama8BHostExample : UdonSharpBehaviour
{
    [SerializeField]
    private UdonOnnxRuntimeBehaviour runtime;

    private float[] hiddenState;
    private float[] logits;
    private int predictedToken = -1;
    private int status = UdonOnnxRuntimeBehaviour.StatusNotInitialized;

    public int TryRunHiddenState(float[] value)
    {
        if (runtime == null)
            return SetStatus(UdonOnnxRuntimeBehaviour.StatusNotInitialized);

        int inputIndex = runtime.FindInput("hidden_state");
        int outputIndex = runtime.FindOutput("logits");
        if (inputIndex < 0 || outputIndex < 0 || value == null)
            return SetStatus(UdonOnnxRuntimeBehaviour.StatusInvalidArgument);

        int inputWords = runtime.GetInputWordCount(inputIndex);
        int outputWords = runtime.GetOutputWordCount(outputIndex);
        if (value.Length != inputWords || outputWords <= 0)
            return SetStatus(UdonOnnxRuntimeBehaviour.StatusInvalidArgument);

        hiddenState = value;
        logits = new float[outputWords];
        predictedToken = -1;
        status = runtime.TryWriteInputFloat(inputIndex, hiddenState);
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return status;
        status = runtime.TryRun();
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return status;
        status = runtime.StartReadOutputFloat(
            outputIndex,
            logits,
            this,
            nameof(OnReadbackComplete)
        );
        return status;
    }

    public void OnReadbackComplete()
    {
        status = runtime.GetLastReadbackStatus();
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess || logits.Length == 0)
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

    public float[] GetLogits()
    {
        return logits;
    }

    private int SetStatus(int value)
    {
        status = value;
        return value;
    }
}
