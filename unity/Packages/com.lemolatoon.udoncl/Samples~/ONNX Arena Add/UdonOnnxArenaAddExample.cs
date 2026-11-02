using Lemolatoon.Udoncl;
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public sealed class UdonOnnxArenaAddExample : UdonSharpBehaviour
{
    [SerializeField]
    private UdonOnnxRuntimeBehaviour runtime;

    [SerializeField]
    private float left = 2.0f;

    [SerializeField]
    private float right = 3.0f;

    private float result;
    private int status;
    private float[] output;

    public override void Interact()
    {
        Run();
    }

    public void Run()
    {
        if (runtime == null)
        {
            status = UdonOnnxRuntimeBehaviour.StatusNotInitialized;
            return;
        }
        int leftIndex = runtime.FindInput("left");
        int rightIndex = runtime.FindInput("right");
        int outputIndex = runtime.FindOutput("sum");
        if (leftIndex < 0 || rightIndex < 0 || outputIndex < 0)
        {
            status = UdonOnnxRuntimeBehaviour.StatusInvalidArgument;
            return;
        }

        int leftWords = runtime.GetInputWordCount(leftIndex);
        int rightWords = runtime.GetInputWordCount(rightIndex);
        int outputWords = runtime.GetOutputWordCount(outputIndex);
        if (leftWords <= 0 || leftWords != rightWords || leftWords != outputWords)
        {
            status = UdonOnnxRuntimeBehaviour.StatusInvalidArgument;
            return;
        }
        float[] leftValues = new float[leftWords];
        float[] rightValues = new float[rightWords];
        output = new float[outputWords];
        leftValues[0] = left;
        rightValues[0] = right;

        status = runtime.TryWriteInputFloat(leftIndex, leftValues);
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        status = runtime.TryWriteInputFloat(rightIndex, rightValues);
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        status = runtime.TryRun();
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        status = runtime.StartReadOutputFloat(
            outputIndex,
            output,
            this,
            nameof(OnReadbackComplete)
        );
    }

    public void OnReadbackComplete()
    {
        status = runtime.GetLastReadbackStatus();
        if (status == UdonOnnxRuntimeBehaviour.StatusSuccess)
            result = output[0];
    }

    public int GetStatus()
    {
        return status;
    }

    public float GetResult()
    {
        return result;
    }
}
