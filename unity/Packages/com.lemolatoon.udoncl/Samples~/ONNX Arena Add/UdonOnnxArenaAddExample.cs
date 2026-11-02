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
        int leftIndex = runtime.FindInput("left");
        int rightIndex = runtime.FindInput("right");
        int outputIndex = runtime.FindOutput("sum");
        if (leftIndex < 0 || rightIndex < 0 || outputIndex < 0)
        {
            status = UdonOnnxRuntimeBehaviour.StatusInvalidArgument;
            return;
        }

        float[] leftValues = new float[512];
        float[] rightValues = new float[512];
        output = new float[512];
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
