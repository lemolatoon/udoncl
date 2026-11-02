using UdonSharp;
using UnityEngine;
using VRC.SDK3.Rendering;
using VRC.SDKBase;
using VRC.Udon;

namespace Lemolatoon.Udoncl
{
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public sealed class UdonOnnxRuntimeBehaviour : UdonSharpBehaviour
    {
        public const int StatusSuccess = 0;
        public const int StatusNotInitialized = -1;
        public const int StatusBusy = -2;
        public const int StatusInvalidArgument = -3;
        public const int StatusTypeMismatch = -4;
        public const int StatusExecutionFailure = -5;
        public const int StatusDisposed = -6;

        [SerializeField]
        private UdonBehaviour program;

        [SerializeField]
        private string[] inputNames;

        [SerializeField]
        private string[] inputDataTypes;

        [SerializeField]
        private int[] inputWords;

        [SerializeField]
        private int[] inputSlots;

        [SerializeField]
        private string[] outputNames;

        [SerializeField]
        private string[] outputDataTypes;

        [SerializeField]
        private int[] outputWords;

        [SerializeField]
        private int[] outputSlots;

        [SerializeField]
        private RenderTexture[] arenaSlots;

        [SerializeField]
        private Material bufferCopyMaterial;

        private Texture2D[] uploadTextures;
        private bool busy;
        private bool readbackBusy;
        private bool disposed;
        private int pendingOutputIndex;
        private float[] pendingFloatReadback;
        private int[] pendingIntReadback;
        private UdonSharpBehaviour pendingReadbackReceiver;
        private string pendingReadbackEvent;
        private int lastReadbackStatus;
        private int generation = 1;
        private int pendingReadbackGeneration;

        public bool IsInitialized()
        {
            return program != null && arenaSlots != null && bufferCopyMaterial != null && !disposed;
        }

        public int GetInputCount()
        {
            return inputNames == null ? 0 : inputNames.Length;
        }

        public int GetOutputCount()
        {
            return outputNames == null ? 0 : outputNames.Length;
        }

        public string GetInputName(int index)
        {
            return inputNames != null && index >= 0 && index < inputNames.Length
                ? inputNames[index]
                : null;
        }

        public string GetOutputName(int index)
        {
            return outputNames != null && index >= 0 && index < outputNames.Length
                ? outputNames[index]
                : null;
        }

        public int FindInput(string name)
        {
            if (inputNames == null || string.IsNullOrEmpty(name))
                return -1;
            for (int index = 0; index < inputNames.Length; ++index)
                if (inputNames[index] == name)
                    return index;
            return -1;
        }

        public int FindOutput(string name)
        {
            if (outputNames == null || string.IsNullOrEmpty(name))
                return -1;
            for (int index = 0; index < outputNames.Length; ++index)
                if (outputNames[index] == name)
                    return index;
            return -1;
        }

        public int TryWriteInputFloat(int inputIndex, float[] source)
        {
            if (source == null)
                return StatusInvalidArgument;
            int status = ValidateInput(inputIndex, source.Length, "float32");
            if (status != StatusSuccess)
                return status;
            Color32[] pixels = NewPixelBuffer(inputSlots[inputIndex]);
            for (int word = 0; word < source.Length; ++word)
                pixels[word] = PackWord(System.BitConverter.SingleToInt32Bits(source[word]));
            return Upload(inputSlots[inputIndex], pixels);
        }

        public int TryWriteInputInt(int inputIndex, int[] source)
        {
            if (source == null)
                return StatusInvalidArgument;
            int status = ValidateInput(inputIndex, source.Length, "int32");
            if (status != StatusSuccess)
                return status;
            Color32[] pixels = NewPixelBuffer(inputSlots[inputIndex]);
            for (int word = 0; word < source.Length; ++word)
                pixels[word] = PackWord(source[word]);
            return Upload(inputSlots[inputIndex], pixels);
        }

        public int TryRun()
        {
            int status = ValidateReady();
            if (status != StatusSuccess)
                return status;
            busy = true;
            program.SendCustomEvent("_run");
            busy = false;
            return StatusSuccess;
        }

        public int StartReadOutputFloat(
            int outputIndex,
            float[] destination,
            UdonSharpBehaviour receiver,
            string completionEvent
        )
        {
            if (destination == null)
                return StatusInvalidArgument;
            pendingFloatReadback = destination;
            pendingIntReadback = null;
            int status = StartReadback(
                outputIndex,
                destination.Length,
                "float32",
                receiver,
                completionEvent
            );
            if (status != StatusSuccess)
                pendingFloatReadback = null;
            return status;
        }

        public int StartReadOutputInt(
            int outputIndex,
            int[] destination,
            UdonSharpBehaviour receiver,
            string completionEvent
        )
        {
            if (destination == null)
                return StatusInvalidArgument;
            pendingIntReadback = destination;
            pendingFloatReadback = null;
            int status = StartReadback(
                outputIndex,
                destination.Length,
                "int32",
                receiver,
                completionEvent
            );
            if (status != StatusSuccess)
                pendingIntReadback = null;
            return status;
        }

        public int GetLastReadbackStatus()
        {
            return lastReadbackStatus;
        }

        public int TryDispose()
        {
            if (disposed)
                return StatusSuccess;
            if (busy || readbackBusy)
                return StatusBusy;
            generation = generation == int.MaxValue ? 1 : generation + 1;
            disposed = true;
            program = null;
            return StatusSuccess;
        }

        public override void OnAsyncGpuReadbackComplete(VRCAsyncGPUReadbackRequest request)
        {
            if (!readbackBusy)
                return;
            if (pendingReadbackGeneration != generation)
            {
                readbackBusy = false;
                pendingFloatReadback = null;
                pendingIntReadback = null;
                pendingReadbackReceiver = null;
                pendingReadbackEvent = null;
                pendingReadbackGeneration = 0;
                return;
            }
            int slot = outputSlots[pendingOutputIndex];
            RenderTexture source = arenaSlots[slot];
            Color32[] pixels = new Color32[source.width * source.height];
            bool success = request.TryGetData(pixels, 0);
            if (success)
            {
                int words =
                    pendingFloatReadback != null
                        ? pendingFloatReadback.Length
                        : pendingIntReadback.Length;
                for (int word = 0; word < words; ++word)
                {
                    int bits = UnpackWord(pixels[word]);
                    if (pendingFloatReadback != null)
                        pendingFloatReadback[word] = System.BitConverter.Int32BitsToSingle(bits);
                    else
                        pendingIntReadback[word] = bits;
                }
            }
            lastReadbackStatus = success ? StatusSuccess : StatusExecutionFailure;
            readbackBusy = false;
            pendingFloatReadback = null;
            pendingIntReadback = null;
            UdonSharpBehaviour receiver = pendingReadbackReceiver;
            string completionEvent = pendingReadbackEvent;
            pendingReadbackReceiver = null;
            pendingReadbackEvent = null;
            pendingReadbackGeneration = 0;
            if (receiver != null)
                receiver.SendCustomEvent(completionEvent);
        }

        private int ValidateReady()
        {
            if (disposed)
                return StatusDisposed;
            if (busy || readbackBusy)
                return StatusBusy;
            return IsInitialized() ? StatusSuccess : StatusNotInitialized;
        }

        private int ValidateInput(int index, int words, string dataType)
        {
            int status = ValidateReady();
            if (status != StatusSuccess)
                return status;
            if (
                inputDataTypes == null
                || inputWords == null
                || inputSlots == null
                || index < 0
                || index >= inputDataTypes.Length
                || index >= inputWords.Length
                || index >= inputSlots.Length
                || inputSlots[index] < 0
                || inputSlots[index] >= arenaSlots.Length
                || arenaSlots[inputSlots[index]] == null
                || words != inputWords[index]
            )
                return StatusInvalidArgument;
            return inputDataTypes[index] == dataType ? StatusSuccess : StatusTypeMismatch;
        }

        private int StartReadback(
            int index,
            int words,
            string dataType,
            UdonSharpBehaviour receiver,
            string completionEvent
        )
        {
            int status = ValidateReady();
            if (status != StatusSuccess)
                return status;
            if (
                outputDataTypes == null
                || outputWords == null
                || outputSlots == null
                || index < 0
                || index >= outputDataTypes.Length
                || index >= outputWords.Length
                || index >= outputSlots.Length
                || outputSlots[index] < 0
                || outputSlots[index] >= arenaSlots.Length
                || arenaSlots[outputSlots[index]] == null
                || words != outputWords[index]
                || receiver == null
                || string.IsNullOrEmpty(completionEvent)
            )
                return StatusInvalidArgument;
            if (outputDataTypes[index] != dataType)
                return StatusTypeMismatch;
            readbackBusy = true;
            pendingOutputIndex = index;
            pendingReadbackReceiver = receiver;
            pendingReadbackEvent = completionEvent;
            pendingReadbackGeneration = generation;
            lastReadbackStatus = StatusBusy;
            VRCAsyncGPUReadback.Request(
                arenaSlots[outputSlots[index]],
                0,
                TextureFormat.RGBA32,
                this
            );
            return StatusSuccess;
        }

        private Color32[] NewPixelBuffer(int slot)
        {
            RenderTexture destination = arenaSlots[slot];
            return new Color32[destination.width * destination.height];
        }

        private int Upload(int slot, Color32[] pixels)
        {
            if (uploadTextures == null)
                uploadTextures = new Texture2D[arenaSlots.Length];
            RenderTexture destination = arenaSlots[slot];
            Texture2D staging = uploadTextures[slot];
            if (staging == null)
            {
                staging = new Texture2D(
                    destination.width,
                    destination.height,
                    TextureFormat.RGBA32,
                    false,
                    true
                );
                uploadTextures[slot] = staging;
            }
            staging.SetPixels32(pixels);
            staging.Apply(false, false);
            VRCGraphics.Blit(staging, destination, bufferCopyMaterial);
            return StatusSuccess;
        }

        private Color32 PackWord(int word)
        {
            return new Color32(
                (byte)word,
                (byte)(word >> 8),
                (byte)(word >> 16),
                (byte)(word >> 24)
            );
        }

        private int UnpackWord(Color32 pixel)
        {
            return pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
        }
    }
}
