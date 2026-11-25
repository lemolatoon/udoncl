using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class SaxpyCaller : UdonSharpBehaviour
{
    [SerializeField]
    private UdonclSaxpyDemoBinding binding;

    private bool initialized;
    private bool secondRun;
    private float[] x;
    private float[] y;
    private float[] result;

    private void OnEnable()
    {
        // One frame later every behaviour, the binding included, is ready.
        SendCustomEventDelayedFrames(nameof(InitializeSaxpy), 1);
    }

    private void OnDisable()
    {
        if (!initialized)
            return;
        initialized = false;
        int openclStatus;
        int status = binding.TryDispose(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: Dispose failed with status " + status);
    }

    public void InitializeSaxpy()
    {
        if (initialized)
            return;
        if (binding == null)
        {
            Debug.LogError("SAXPY: assign the UdonclSaxpyDemoBinding instance.");
            return;
        }
        // Initialize creates the buffers and the kernel and enqueues no Device
        // work, so it completes at once.
        int openclStatus;
        int status = binding.TryInitialize(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess || openclStatus != 0)
        {
            Debug.LogError(
                "SAXPY: Initialize failed with status "
                    + status
                    + " and OpenCL status "
                    + openclStatus
            );
            return;
        }
        initialized = true;
    }

    public override void Interact()
    {
        if (!initialized)
            return;
        x = new float[32];
        y = new float[32];
        result = new float[32];
        for (int i = 0; i < 32; i++)
        {
            x[i] = i + 1;
            y[i] = 100 + i;
        }
        secondRun = false;
        StartSaxpy(16, 2.0f);
    }

    private void StartSaxpy(int n, float alpha)
    {
        int status = binding.TryWriteX(x);
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryWriteY(y);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
        {
            Debug.LogError("SAXPY: writing the inputs failed with status " + status);
            return;
        }

        status = binding.StartRun(n, alpha, this, nameof(OnSaxpyComplete));
        if (status != UdonclSaxpyDemoBinding.StatusSuccess)
            Debug.LogError("SAXPY: StartRun failed with status " + status);
    }

    public void OnSaxpyComplete()
    {
        int openclStatus = 0;
        int status = binding.GetRunStatus();
        if (status == UdonclSaxpyDemoBinding.StatusSuccess)
            status = binding.TryGetRunResult(out openclStatus);
        if (status != UdonclSaxpyDemoBinding.StatusSuccess || openclStatus != 0)
        {
            Debug.LogError(
                "SAXPY: Run failed with status " + status + " and OpenCL status " + openclStatus
            );
            return;
        }

        status = binding.StartReadResult(result, this, nameof(OnSaxpyReadback));
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

        if (!secondRun)
        {
            // The first run: alpha 2 over the first 16 words.
            for (int i = 0; i < 16; i++)
            {
                float expected = 2.0f * x[i] + y[i];
                Debug.Log(
                    "SAXPY: result[" + i + "] = " + result[i] + " (expected " + expected + ")"
                );
            }

            // The second run reuses the kernel with new inputs: alpha 0.5 over
            // all 32 words.
            for (int i = 0; i < 32; i++)
            {
                x[i] = 10 * (i + 1);
                y[i] = -i;
            }
            secondRun = true;
            StartSaxpy(32, 0.5f);
            return;
        }

        for (int i = 0; i < 32; i++)
        {
            float expected = 0.5f * x[i] + y[i];
            Debug.Log(
                "SAXPY: second result[" + i + "] = " + result[i] + " (expected " + expected + ")"
            );
        }
    }
}
