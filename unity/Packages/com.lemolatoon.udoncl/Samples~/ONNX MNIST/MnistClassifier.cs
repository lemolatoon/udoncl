using Lemolatoon.Udoncl.Generated;
using UdonSharp;
using UnityEngine;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class MnistClassifier : UdonSharpBehaviour
{
    // A hand-drawn 28x28 digit, row by row. The model reads pixel
    // intensities in [0, 1]: '#' is ink (1.0) and '.' is background (0.0).
    private const string Digit =
        "............................"
        + "............................"
        + "............................"
        + "............................"
        + "............................"
        + ".......##############......."
        + ".......##############......."
        + "...................###......"
        + "..................###......."
        + "..................###......."
        + ".................###........"
        + ".................###........"
        + "................###........."
        + "................###........."
        + "...............###.........."
        + "...............###.........."
        + "..............###..........."
        + "..............###..........."
        + ".............###............"
        + ".............###............"
        + "............###............."
        + "............###............."
        + "............###............."
        + "............................"
        + "............................"
        + "............................"
        + "............................"
        + "............................";

    [SerializeField]
    private UdonclGenericOnnxMnistBinding binding;

    private float[] image;
    private float[] logits;
    private bool initialized;

    private void Start()
    {
        // Initialize one frame after start-up so every behaviour is ready.
        SendCustomEventDelayedFrames("InitializeModel", 1);
    }

    public void InitializeModel()
    {
        if (binding == null)
        {
            Debug.LogError("MNIST: assign the UdonclGenericOnnxMnistBinding instance.");
            return;
        }

        image = new float[28 * 28];
        logits = new float[10];
        for (int i = 0; i < image.Length; i++)
            image[i] = Digit[i] == '#' ? 1.0f : 0.0f;

        // Initialize creates the OpenCL context, the ONNX session and its
        // input and output buffers in the C Host, and uploads the weights.
        int status = binding.StartInitialize(this, "OnModelInitialized");
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
            Debug.LogError("MNIST: StartInitialize failed with status " + status);
    }

    public void OnModelInitialized()
    {
        int libraryStatus = 0;
        int status = binding.GetInitializeStatus();
        if (status == UdonclGenericOnnxMnistBinding.StatusSuccess)
            status = binding.TryGetInitializeResult(out libraryStatus);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess || libraryStatus != 0)
        {
            Debug.LogError(
                "MNIST: Initialize failed with status "
                    + status
                    + " and ONNX status "
                    + libraryStatus
            );
            return;
        }

        initialized = true;
        Classify();
    }

    public override void Interact()
    {
        Classify();
    }

    public void Classify()
    {
        if (!initialized)
            return;

        int status = binding.TryWriteImage(image);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
        {
            Debug.LogError("MNIST: writing the image failed with status " + status);
            return;
        }

        status = binding.StartRun(this, "OnRunComplete");
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
            Debug.LogError("MNIST: StartRun failed with status " + status);
    }

    public void OnRunComplete()
    {
        int libraryStatus = 0;
        int status = binding.GetRunStatus();
        if (status == UdonclGenericOnnxMnistBinding.StatusSuccess)
            status = binding.TryGetRunResult(out libraryStatus);
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess || libraryStatus != 0)
        {
            Debug.LogError(
                "MNIST: Run failed with status " + status + " and ONNX status " + libraryStatus
            );
            return;
        }

        status = binding.StartReadLogits(logits, this, "OnLogitsRead");
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
            Debug.LogError("MNIST: StartReadLogits failed with status " + status);
    }

    public void OnLogitsRead()
    {
        int status = binding.GetLastReadbackStatus();
        if (status != UdonclGenericOnnxMnistBinding.StatusSuccess)
        {
            Debug.LogError("MNIST: readback failed with status " + status);
            return;
        }

        int predicted = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[predicted])
                predicted = i;
        }
        Debug.Log("MNIST: predicted " + predicted + " (logit " + logits[predicted] + ")");
    }
}
