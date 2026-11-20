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
    public sealed class AssignedQ4ComponentLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/onnx_q4_component";
        private const int M = 4;
        private const int N = 33;
        private const int K = 2048;
        private const int Block = 32;
        private string artifactDirectory;
        private ValidatedArtifact artifact;

        [OneTimeSetUp]
        public void ImportArtifact()
        {
            string repository = PublicMnistOnnxRuntimeLifecycleTests.ResolveRepository();
            string configured = Environment.GetEnvironmentVariable(
                "UDONCL_ASSIGNED_Q4_COMPONENT_UNITY_BUNDLE"
            );
            artifactDirectory = !string.IsNullOrWhiteSpace(configured)
                ? Path.GetFullPath(configured)
                : Path.Combine(
                    repository,
                    "build",
                    "udoncl",
                    "tests",
                    "onnx-q4-component",
                    "consumer",
                    "udoncl",
                    "onnx-q4-component",
                    "artifact"
                );
            artifact = ArtifactManifestValidator.Validate(artifactDirectory);
            Assert.AreEqual("onnx_q4_component", artifact.ProgramName);
            if (!AssetDatabase.IsValidFolder(GeneratedRoot))
                Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));
        }

        [Test]
        public void AssignedBuffersMatchTinyLlamaDepthQ4Oracle()
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

            Texture2D[] staticTensors = new Texture2D[artifact.StaticTensors.Count];
            foreach (ValidatedStaticTensor tensor in artifact.StaticTensors)
                staticTensors[tensor.Id] = Load<Texture2D>(
                    "Textures/"
                        + UdonclArtifactImporter.StaticTensorFileName(tensor.Id, tensor.Name)
                );
            PublicMnistOnnxRuntimeLifecycleTests.Set(
                heap,
                program,
                artifact.StaticTensorTableSlot,
                staticTensors
            );

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
                "Q4 component initialization failed after " + initializeCommands + " commands"
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
                "Q4 component run failed after " + commands + " commands"
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
                "assigned_q4_component commands={0} outputs={1} max_error={2:R}",
                commands,
                expected.Length,
                maximumError
            );
            Assert.That(maximumError, Is.LessThan(0.005f));
        }

        [OneTimeTearDown]
        public void RemoveGeneratedUnityAssets()
        {
            RemoveGeneratedUnityAssets(GeneratedRoot);
        }

        internal static void RemoveGeneratedUnityAssets(string generatedRoot)
        {
            RenderTexture.active = null;
            string programPath = generatedRoot + "/Programs/host_program.asset";
            string programGuid = AssetDatabase.AssetPathToGUID(programPath);
            AssetDatabase.DeleteAsset(generatedRoot);
            string serializedProgramPath = string.IsNullOrEmpty(programGuid)
                ? null
                : "Assets/SerializedUdonPrograms/" + programGuid + ".asset";
            if (serializedProgramPath != null)
                AssetDatabase.DeleteAsset(serializedProgramPath);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(generatedRoot));
            if (serializedProgramPath != null)
                Assert.IsNull(
                    AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(serializedProgramPath)
                );
        }

        private static float[] Input()
        {
            float[] values = new float[M * K];
            for (int row = 0; row < M; ++row)
            for (int inner = 0; inner < K; ++inner)
                values[row * K + inner] = (((row + 3) * 11 + inner * 5) % 29 - 14) / 17.0f;
            return values;
        }

        private static float[] Oracle(float[] input)
        {
            int blocks = K / Block;
            float[] output = new float[M * N];
            for (int row = 0; row < M; ++row)
            for (int column = 0; column < N; ++column)
            for (int inner = 0; inner < K; ++inner)
            {
                int block = inner / Block;
                int quantized = (column * 3 + inner * 7) & 15;
                float scale = ((column + block) % 7 + 1) / 23.0f;
                output[row * N + column] += (quantized - 8.0f) * scale * input[row * K + inner];
            }
            Assert.AreEqual(64, blocks);
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
