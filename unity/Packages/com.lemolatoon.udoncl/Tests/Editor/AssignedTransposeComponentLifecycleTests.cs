using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class AssignedTransposeComponentLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/onnx_transpose_component";
        private const int Sequence = 4;
        private const int Heads = 32;
        private const int Width = 64;
        private string artifactDirectory;
        private ValidatedArtifact artifact;

        [OneTimeSetUp]
        public void ImportArtifact()
        {
            string repository = PublicMnistOnnxRuntimeLifecycleTests.ResolveRepository();
            string configured = Environment.GetEnvironmentVariable(
                "UDONCL_ASSIGNED_TRANSPOSE_COMPONENT_UNITY_BUNDLE"
            );
            artifactDirectory = !string.IsNullOrWhiteSpace(configured)
                ? Path.GetFullPath(configured)
                : Path.Combine(
                    repository,
                    "build",
                    "udoncl",
                    "tests",
                    "onnx-transpose-component",
                    "consumer",
                    "udoncl",
                    "onnx-transpose-component",
                    "artifact"
                );
            artifact = ArtifactManifestValidator.Validate(artifactDirectory);
            Assert.AreEqual("onnx_transpose_component", artifact.ProgramName);
            if (!AssetDatabase.IsValidFolder(GeneratedRoot))
                Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));
        }

        [Test]
        public void AssignedBuffersMatchAttentionTransposeOracle()
        {
            UdonAssemblyProgramAsset asset = Load<UdonAssemblyProgramAsset>(
                "Programs/host_program.asset"
            );
            IUdonProgram program = asset.SerializedProgramAsset.RetrieveProgram();
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(program);
            Assert.NotNull(vm);
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();
            int[] rawHeap = PublicMnistOnnxRuntimeLifecycleTests.PackRawHeap(
                File.ReadAllBytes(Path.Combine(artifactDirectory, "raw-heap.bytes"))
            );
            PublicMnistOnnxRuntimeLifecycleTests.Set(heap, program, "__vrc_raw_heap", rawHeap);

            Dictionary<int, RenderTexture> buffers = new Dictionary<int, RenderTexture>();
            RenderTexture[] bufferPool = new RenderTexture[artifact.DeviceBuffers.Count];
            int inputBuffer = 0;
            int outputBuffer = 0;
            for (int index = 0; index < artifact.DeviceBuffers.Count; ++index)
            {
                ValidatedDeviceBuffer descriptor = artifact.DeviceBuffers[index];
                RenderTexture buffer = Load<RenderTexture>(
                    "RenderTextures/"
                        + UdonclArtifactImporter.DeviceBufferFileName(
                            descriptor.Id,
                            descriptor.Name
                        )
                );
                buffers.Add(descriptor.Id, buffer);
                bufferPool[index] = buffer;
                PublicMnistOnnxRuntimeLifecycleTests.Clear(buffer);
                if (descriptor.PublicAccess == "input")
                    inputBuffer = descriptor.Id;
                if (descriptor.PublicAccess == "output")
                    outputBuffer = descriptor.Id;
            }
            Assert.AreNotEqual(0, inputBuffer);
            Assert.AreNotEqual(0, outputBuffer);
            PublicMnistOnnxRuntimeLifecycleTests.Set(
                heap,
                program,
                "__udoncl_buffer_pool",
                bufferPool
            );

            Dictionary<int, Material> materials = new Dictionary<int, Material>();
            foreach (ValidatedKernel kernel in artifact.Kernels)
            {
                Material material = Load<Material>(
                    "Materials/kernel_" + kernel.Id + "_" + kernel.Name + ".mat"
                );
                materials.Add(kernel.Id, material);
            }
            // Every dispatch site Blits through its own material.
            foreach (ValidatedDispatchMaterial dispatch in artifact.DispatchMaterials)
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    dispatch.Slot,
                    Load<Material>("Materials/Dispatch/" + dispatch.AssetName + ".mat")
                );
            Material copyMaterial = Load<Material>("Materials/buffer-copy.mat");
            int initializeCommands = PublicMnistOnnxRuntimeLifecycleTests.RunAsyncExport(
                vm,
                program,
                heap,
                rawHeap,
                buffers,
                materials,
                copyMaterial,
                "Initialize"
            );
            Assert.AreEqual(
                0,
                PublicMnistOnnxRuntimeLifecycleTests.Get<int>(heap, program, "__0_Initialize__ret"),
                "Transpose initialization failed after " + initializeCommands + " commands"
            );

            float[] input = Input();
            PublicMnistOnnxRuntimeLifecycleTests.UploadFloats(
                buffers[inputBuffer],
                input,
                copyMaterial
            );
            int commands = PublicMnistOnnxRuntimeLifecycleTests.RunAsyncExport(
                vm,
                program,
                heap,
                rawHeap,
                buffers,
                materials,
                copyMaterial,
                "Run"
            );
            Assert.AreEqual(
                0,
                PublicMnistOnnxRuntimeLifecycleTests.Get<int>(heap, program, "__0_Run__ret"),
                "Transpose run failed after " + commands + " commands"
            );

            float[] expected = Oracle(input);
            Color32[] pixels = PublicMnistOnnxRuntimeLifecycleTests.ReadPixels(
                buffers[outputBuffer]
            );
            float maximumError = 0.0f;
            for (int index = 0; index < expected.Length; ++index)
            {
                Color32 pixel = pixels[index];
                int word = pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
                float actual = BitConverter.Int32BitsToSingle(word);
                maximumError = Mathf.Max(maximumError, Mathf.Abs(actual - expected[index]));
            }
            TestContext.Progress.WriteLine(
                "assigned_transpose_component commands={0} outputs={1} max_error={2:R}",
                commands,
                expected.Length,
                maximumError
            );
            Assert.That(maximumError, Is.LessThan(1.0e-6f));
        }

        [OneTimeTearDown]
        public void RemoveGeneratedUnityAssets()
        {
            AssignedQ4ComponentLifecycleTests.RemoveGeneratedUnityAssets(GeneratedRoot);
        }

        private static float[] Input()
        {
            float[] values = new float[Sequence * Heads * Width];
            for (int index = 0; index < values.Length; ++index)
                values[index] = ((index * 13) % 97 - 48) / 31.0f;
            return values;
        }

        private static float[] Oracle(float[] input)
        {
            float[] output = new float[input.Length];
            for (int head = 0; head < Heads; ++head)
            for (int sequence = 0; sequence < Sequence; ++sequence)
            for (int column = 0; column < Width; ++column)
                output[(head * Sequence + sequence) * Width + column] = input[
                    (sequence * Heads + head) * Width + column
                ];
            return output;
        }

        private static T Load<T>(string relativePath)
            where T : UnityEngine.Object
        {
            T asset = AssetDatabase.LoadAssetAtPath<T>(GeneratedRoot + "/" + relativePath);
            Assert.NotNull(asset, relativePath);
            return asset;
        }
    }
}
