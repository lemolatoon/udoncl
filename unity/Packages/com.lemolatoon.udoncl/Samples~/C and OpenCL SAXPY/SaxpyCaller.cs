using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class SaxpyCaller : UdonSharpBehaviour
{
    [SerializeField]
    private UdonclSaxpyDemoBinding binding;

    private float[] x;
    private float[] y;
    private float[] result;

    private void Start()
    {
        // Run one frame after start-up so every behaviour is initialized.
        SendCustomEventDelayedFrames("RunSaxpyOnce", 1);
    }

    public void RunSaxpyOnce()
    {
        if (binding == null)
        {
            Debug.LogError("SAXPY: assign the UdonclSaxpyDemoBinding instance.");
            return;
        }

        x = new float[32];
        y = new float[32];
        result = new float[32];
        for (int i = 0; i < 32; i++)
        {
            x[i] = i + 1;
            y[i] = 100 + i;
        }

        int status = binding.TryWriteX(x);
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryWriteY(y);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: writing the inputs failed with status " + status);
            return;
        }

        status = binding.StartRunSaxpy(16, 2.0f, this, "OnSaxpyComplete");
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartRunSaxpy failed with status " + status);
    }

    public void OnSaxpyComplete()
    {
        int openclStatus = 0;
        int status = binding.GetRunSaxpyStatus();
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryGetRunSaxpyResult(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess || openclStatus != 0)
        {
            Debug.LogError(
                "SAXPY: RunSaxpy failed with status "
                    + status
                    + " and OpenCL status "
                    + openclStatus
            );
            return;
        }

        status = binding.StartReadResult(result, this, "OnSaxpyReadback");
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartReadResult failed with status " + status);
    }

    public void OnSaxpyReadback()
    {
        int status = binding.GetLastReadbackStatus();
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: readback failed with status " + status);
            return;
        }

        for (int i = 0; i < 16; i++)
        {
            float expected = 2.0f * x[i] + y[i];
            Debug.Log("SAXPY: result[" + i + "] = " + result[i] + " (expected " + expected + ")");
        }
    }
}
