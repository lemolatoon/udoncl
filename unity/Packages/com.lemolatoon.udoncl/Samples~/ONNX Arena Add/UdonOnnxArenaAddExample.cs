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
    private int stage;
    private int leftWordCount;
    private int rightWordCount;
    private int outputWordCount;
    private float[] output;

    public override void Interact()
    {
        Run();
    }

    public void Run()
    {
        stage = 1;
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
        stage = 2;

        leftWordCount = runtime.GetInputWordCount(leftIndex);
        rightWordCount = runtime.GetInputWordCount(rightIndex);
        outputWordCount = runtime.GetOutputWordCount(outputIndex);
        if (
            leftWordCount <= 0
            || leftWordCount != rightWordCount
            || leftWordCount != outputWordCount
        )
        {
            status = UdonOnnxRuntimeBehaviour.StatusInvalidArgument;
            return;
        }
        stage = 3;
        float[] leftValues = new float[leftWordCount];
        float[] rightValues = new float[rightWordCount];
        output = new float[outputWordCount];
        leftValues[0] = left;
        rightValues[0] = right;

        status = runtime.TryWriteInputFloat(leftIndex, leftValues);
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        stage = 4;
        status = runtime.TryWriteInputFloat(rightIndex, rightValues);
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        stage = 5;
        status = runtime.TryRun();
        if (status != UdonOnnxRuntimeBehaviour.StatusSuccess)
            return;
        stage = 6;
        status = runtime.StartReadOutputFloat(
            outputIndex,
            output,
            this,
            nameof(OnReadbackComplete)
        );
        if (status == UdonOnnxRuntimeBehaviour.StatusSuccess)
            stage = 7;
    }

    public void OnReadbackComplete()
    {
        stage = 8;
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

    public int GetStage()
    {
        return stage;
    }

    public int GetLeftWordCount()
    {
        return leftWordCount;
    }

    public int GetRightWordCount()
    {
        return rightWordCount;
    }

    public int GetOutputWordCount()
    {
        return outputWordCount;
    }
}
