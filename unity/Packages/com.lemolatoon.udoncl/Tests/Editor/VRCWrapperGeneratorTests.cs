using System;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class VRCWrapperGeneratorTests
    {
        private const string ScalarManifest =
            "{"
            + "\"physical_export_abi\":\"udonsharp-worlds-sdk-3.10.5\","
            + "\"exports\":[{"
            + "\"id\":1,"
            + "\"c_symbol\":\"add\","
            + "\"public_name\":\"Add\","
            + "\"udon_event\":\"__0_Add\","
            + "\"parameters\":["
            + "{\"index\":0,\"name\":\"left\",\"kind\":\"i32\",\"heap_slot\":\"__0_left__param\"},"
            + "{\"index\":1,\"name\":\"right\",\"kind\":\"i32\",\"heap_slot\":\"__0_right__param\"}],"
            + "\"result\":{\"kind\":\"i32\",\"heap_slot\":\"__0___0_Add__ret\"},"
            + "\"status_slot\":\"__udoncl_status_1\","
            + "\"async\":false"
            + "}]}";

        [Test]
        public void GeneratesPinnedUdonSharpSetSendGetWrapper()
        {
            string source = VRCWrapperGenerator.GenerateFromJson(
                ScalarManifest,
                "Lemolatoon.Generated",
                "AdderBinding"
            );

            StringAssert.Contains("public sealed class AdderBinding : UdonSharpBehaviour", source);
            StringAssert.Contains("public int TryAdd(int left, int right, out int result)", source);
            StringAssert.Contains("program.SetProgramVariable(\"__0_left__param\", left);", source);
            StringAssert.Contains("program.SendCustomEvent(\"__0_Add\");", source);
            StringAssert.Contains(
                "object rawResult = program.GetProgramVariable(\"__0___0_Add__ret\");",
                source
            );
            StringAssert.Contains(
                "object rawStatus = program.GetProgramVariable(\"__udoncl_status_1\");",
                source
            );
            StringAssert.DoesNotContain("GetProgramVariable<int>", source);
            StringAssert.Contains("public const int StatusSuccess = 0;", source);
            StringAssert.Contains("private bool busy;", source);
            StringAssert.Contains("private bool disposed;", source);
            StringAssert.Contains("private int generation = 1;", source);
            StringAssert.Contains(
                "generation = generation == int.MaxValue ? 1 : generation + 1;",
                source
            );
            StringAssert.Contains("public int TryBindProgram(UdonBehaviour value)", source);
            StringAssert.Contains("public bool IsInitialized()", source);
            StringAssert.Contains("public int TryDispose()", source);
            StringAssert.Contains("if (disposed)", source);
            StringAssert.Contains("return StatusDisposed;", source);
            StringAssert.Contains("if (busy)", source);
            StringAssert.Contains("return StatusBusy;", source);
            StringAssert.Contains("busy = true;", source);
            StringAssert.Contains("busy = false;", source);
            StringAssert.Contains("if (status != StatusSuccess)", source);
            StringAssert.Contains("if (rawStatus == null)", source);
            StringAssert.Contains("if (rawResult == null)", source);
            StringAssert.DoesNotContain(" is int", source);
            StringAssert.DoesNotContain("UdonClStatus", source);
            StringAssert.Contains("result = (int)rawResult;", source);
        }

        [Test]
        public void GeneratesVoidExportWhenJsonUtilityMaterializesNullResult()
        {
            string manifest = ScalarManifest
                .Replace("\"public_name\":\"Add\"", "\"public_name\":\"CopyFull\"")
                .Replace("\"udon_event\":\"__0_Add\"", "\"udon_event\":\"CopyFull\"")
                .Replace(
                    "\"parameters\":["
                        + "{\"index\":0,\"name\":\"left\",\"kind\":\"i32\",\"heap_slot\":\"__0_left__param\"},"
                        + "{\"index\":1,\"name\":\"right\",\"kind\":\"i32\",\"heap_slot\":\"__0_right__param\"}]",
                    "\"parameters\":[]"
                )
                .Replace(
                    "\"result\":{\"kind\":\"i32\",\"heap_slot\":\"__0___0_Add__ret\"}",
                    "\"result\":null"
                );
            string source = VRCWrapperGenerator.GenerateFromJson(
                manifest,
                "Lemolatoon.Generated",
                "CopyBinding"
            );

            StringAssert.Contains("public int TryCopyFull()", source);
            StringAssert.Contains("program.SendCustomEvent(\"CopyFull\");", source);
            StringAssert.DoesNotContain("out int result", source);
            StringAssert.DoesNotContain("rawResult", source);
        }

        [Test]
        public void GeneratesPrimitiveArrayParametersWithoutCopyingThem()
        {
            string manifest = ScalarManifest
                .Replace("\"public_name\":\"Add\"", "\"public_name\":\"Run\"")
                .Replace("\"udon_event\":\"__0_Add\"", "\"udon_event\":\"__0_Run\"")
                .Replace(
                    "\"parameters\":["
                        + "{\"index\":0,\"name\":\"left\",\"kind\":\"i32\",\"heap_slot\":\"__0_left__param\"},"
                        + "{\"index\":1,\"name\":\"right\",\"kind\":\"i32\",\"heap_slot\":\"__0_right__param\"}]",
                    "\"parameters\":["
                        + "{\"index\":0,\"name\":\"input\",\"kind\":\"f32_array\",\"heap_slot\":\"__0_input__param\"},"
                        + "{\"index\":1,\"name\":\"tokens\",\"kind\":\"i32_array\",\"heap_slot\":\"__0_tokens__param\"}]"
                )
                .Replace(
                    "\"result\":{\"kind\":\"i32\",\"heap_slot\":\"__0___0_Add__ret\"}",
                    "\"result\":null"
                );
            string source = VRCWrapperGenerator.GenerateFromJson(
                manifest,
                "Lemolatoon.Generated",
                "OnnxBinding"
            );

            StringAssert.Contains("public int TryRun(float[] input, int[] tokens)", source);
            StringAssert.Contains(
                "program.SetProgramVariable(\"__0_input__param\", input);",
                source
            );
            StringAssert.Contains(
                "program.SetProgramVariable(\"__0_tokens__param\", tokens);",
                source
            );
            StringAssert.DoesNotContain("new float", source);
            StringAssert.DoesNotContain("new int", source);
        }

        [Test]
        public void GeneratesByteExactDeviceBufferTransferApi()
        {
            string manifest = ScalarManifest.Replace(
                "]}",
                "],\"device\":{\"buffers\":[{"
                    + "\"id\":7,\"name\":\"input\",\"heap_slot\":\"__udoncl_buffer_7\","
                    + "\"logical_bytes\":16,\"width\":4,\"height\":1,"
                    + "\"format\":\"rgba8_unorm\",\"element_type\":\"float32\","
                    + "\"public_access\":\"input\"},{"
                    + "\"id\":8,\"name\":\"output\",\"heap_slot\":\"__udoncl_buffer_8\","
                    + "\"logical_bytes\":16,\"width\":4,\"height\":1,"
                    + "\"format\":\"rgba8_unorm\",\"element_type\":\"int32\","
                    + "\"public_access\":\"output\"}]}}"
            );
            string source = VRCWrapperGenerator.GenerateFromJson(
                manifest,
                "Lemolatoon.Generated",
                "TransferBinding"
            );

            StringAssert.Contains(
                "public int TryWriteFloatBuffer(int bufferId, float[] source)",
                source
            );
            StringAssert.Contains(
                "public int TryWriteIntBuffer(int bufferId, int[] source)",
                source
            );
            StringAssert.Contains("System.BitConverter.SingleToInt32Bits", source);
            StringAssert.Contains("staging.SetPixels32(pixels);", source);
            StringAssert.Contains(
                "VRCGraphics.Blit(staging, destination, bufferCopyMaterial);",
                source
            );
            StringAssert.Contains("public int StartReadFloatBuffer", source);
            StringAssert.Contains("public int StartReadIntBuffer", source);
            StringAssert.Contains("VRCAsyncGPUReadback.Request", source);
            StringAssert.Contains("public override void OnAsyncGpuReadbackComplete", source);
            StringAssert.Contains("request.TryGetData(pixels, 0);", source);
            StringAssert.Contains("pendingReadbackGeneration = generation;", source);
            StringAssert.Contains("if (pendingReadbackGeneration != generation)", source);
            StringAssert.Contains("pendingReadbackGeneration = 0;", source);
            StringAssert.Contains("System.BitConverter.Int32BitsToSingle", source);
            StringAssert.Contains("public int TryWriteInput(float[] source)", source);
            StringAssert.Contains("return TryWriteFloatBuffer(7, source);", source);
            StringAssert.Contains("public int StartReadOutput(int[] destination", source);
            StringAssert.Contains(
                "return StartReadIntBuffer(8, destination, receiver, completionEvent);",
                source
            );
        }

        [Test]
        public void GeneratesNamedOnnxWriteRunReadBinding()
        {
            const string manifest =
                "{"
                + "\"artifact_stage\":\"unity-udon-onnx-model\","
                + "\"runtime_abi_version\":1,"
                + "\"model_inputs\":["
                + "{\"index\":0,\"name\":\"left.input\",\"data_type\":\"float32\",\"words\":4,\"slot\":0},"
                + "{\"index\":1,\"name\":\"tokens\",\"data_type\":\"int32\",\"words\":2,\"slot\":1}],"
                + "\"model_outputs\":["
                + "{\"index\":0,\"name\":\"sum\",\"data_type\":\"float32\",\"words\":4,\"slot\":2}]"
                + "}";
            string source = VRCWrapperGenerator.GenerateOnnxFromJson(
                manifest,
                "Lemolatoon.Generated",
                "ArenaAddBinding"
            );

            StringAssert.Contains("public int TryWriteInputLeftInput(float[] source)", source);
            StringAssert.Contains("return TryWriteFloatBuffer(1, source);", source);
            StringAssert.Contains("public int TryWriteInputTokens(int[] source)", source);
            StringAssert.Contains("return TryWriteIntBuffer(2, source);", source);
            StringAssert.Contains("public int TryRun()", source);
            StringAssert.Contains("program.SendCustomEvent(\"_run\");", source);
            StringAssert.Contains("public int StartReadOutputSum(float[] destination", source);
            StringAssert.Contains(
                "return StartReadFloatBuffer(1073741825, destination, receiver, completionEvent);",
                source
            );
            StringAssert.Contains("private int TryWriteFloatBuffer", source);
            StringAssert.Contains("private int StartReadFloatBuffer", source);
            StringAssert.Contains(
                "new Color32((byte)(word & 255), (byte)((word >> 8) & 255), (byte)((word >> 16) & 255), (byte)((word >> 24) & 255))",
                source
            );
            StringAssert.Contains("public int TryDispose()", source);
            StringAssert.Contains("VRCAsyncGPUReadback.Request", source);
            StringAssert.Contains("pendingReadbackGeneration = generation;", source);
            StringAssert.Contains("if (pendingReadbackGeneration != generation)", source);
        }

        [Test]
        public void GeneratesAsyncContinuationAndTransferPump()
        {
            string manifest = ScalarManifest
                .Replace("\"async\":false", "\"async\":true")
                .Replace(
                    "\"exports\":[",
                    "\"program\":{\"async_commands\":{"
                        + "\"resume_state_slot\":\"__resume\","
                        + "\"completion_status_slot\":\"__completion\","
                        + "\"command_kind_slot\":\"__kind\","
                        + "\"command_buffer_slot\":\"__buffer\","
                        + "\"command_offset_slot\":\"__offset\","
                        + "\"command_size_slot\":\"__size\","
                        + "\"command_pointer_slot\":\"__pointer\","
                        + "\"raw_heap_slot\":\"__heap\","
                        + "\"upload_kind\":1,\"readback_kind\":2}},"
                        + "\"device\":{\"buffers\":[{"
                        + "\"id\":7,\"name\":\"scratch\",\"heap_slot\":\"__device_7\","
                        + "\"logical_bytes\":16,\"width\":4,\"height\":1,"
                        + "\"format\":\"rgba8_unorm\"}]},"
                        + "\"exports\":["
                );
            string source = VRCWrapperGenerator.GenerateFromJson(
                manifest,
                "Lemolatoon.Generated",
                "AdderBinding"
            );

            StringAssert.Contains(
                "public int StartAdd(int left, int right, UdonSharpBehaviour receiver, string completionEvent)",
                source
            );
            StringAssert.Contains("program.SetProgramVariable(\"__resume\", 0);", source);
            StringAssert.Contains("private void PumpAsyncCommand()", source);
            StringAssert.Contains("int[] heap = (int[])rawHeap;", source);
            StringAssert.Contains("CopyHeapToPixels(pixels);", source);
            StringAssert.Contains("pointer > heap.Length - size", source);
            StringAssert.Contains("asyncCommandPointerBytes = pointer;", source);
            StringAssert.Contains("int byteIndex = asyncCommandPointerBytes + word * 4;", source);
            StringAssert.Contains("asyncCommandHeap[byteIndex] = pixel.r;", source);
            StringAssert.Contains("private void CompleteAsyncCommandReadback", source);
            StringAssert.Contains(
                "SendCustomEventDelayedFrames(\"_ContinueAsyncCommand\", 1);",
                source
            );
            StringAssert.Contains("public void _ContinueAsyncCommand()", source);
            StringAssert.Contains("public int GetAddStatus()", source);
            StringAssert.Contains("public int TryGetAddResult(out int result)", source);
            StringAssert.DoesNotContain("public int TryAdd(", source);
        }

        [Test]
        public void RejectsUnknownPhysicalAbi()
        {
            string manifest = ScalarManifest.Replace("udonsharp-worlds-sdk-3.10.5", "unknown");
            Assert.Throws<ArgumentException>(() =>
                VRCWrapperGenerator.GenerateFromJson(
                    manifest,
                    "Lemolatoon.Generated",
                    "AdderBinding"
                )
            );
        }

        [Test]
        public void DerivesStableBindingClassNameFromProgramName()
        {
            Assert.AreEqual(
                "UdonclHelloKernelBinding",
                VRCWrapperGenerator.BindingClassName("hello-kernel")
            );
        }

        [Test]
        public void FinalizerWaitsUntilGeneratedWrapperTypeIsCompiled()
        {
            const string root = "Assets/udoncl-generated/__binding_type_pending";
            AssetDatabase.DeleteAsset(root);
            Directory.CreateDirectory(root);
            File.WriteAllText(
                root + "/manifest.json",
                "{"
                    + "\"artifact_stage\":\"unity-udoncl-library\","
                    + "\"physical_export_abi\":\"udonsharp-worlds-sdk-3.10.5\","
                    + "\"program\":{\"name\":\"type_pending\"}"
                    + "}"
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            try
            {
                GeneratedBindingFinalizer.FinalizeAvailableBindings();
                Assert.IsNull(
                    AssetDatabase.LoadAssetAtPath<GameObject>(
                        root + "/Prefabs/UdonclTypePendingBinding.prefab"
                    )
                );
            }
            finally
            {
                AssetDatabase.DeleteAsset(root);
                if (Directory.Exists(root))
                    Directory.Delete(root, true);
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
        }

        [Test]
        public void PendingBindingScenePlacementIsScopedAndConsumedOnce()
        {
            GeneratedBindingScenePlacement.ClearRequest();
            try
            {
                GeneratedBindingScenePlacement.Request("hello_dispatch");
                Assert.IsFalse(GeneratedBindingScenePlacement.ConsumeRequest("saxpy_demo"));
                Assert.IsTrue(GeneratedBindingScenePlacement.ConsumeRequest("hello_dispatch"));
                Assert.IsFalse(GeneratedBindingScenePlacement.ConsumeRequest("hello_dispatch"));
            }
            finally
            {
                GeneratedBindingScenePlacement.ClearRequest();
            }
        }
    }
}
