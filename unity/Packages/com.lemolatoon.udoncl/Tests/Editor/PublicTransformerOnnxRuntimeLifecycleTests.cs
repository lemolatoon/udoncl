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
    public sealed class PublicTransformerOnnxRuntimeLifecycleTests
    {
        private const string GeneratedRoot =
            "Assets/udoncl-generated/generic_onnx_transformer_tiny";
        private const string BindingClass = "UdonclGenericOnnxTransformerTinyBinding";
        private string artifactDirectory;
        private ValidatedArtifact artifact;

        [OneTimeSetUp]
        public void ImportArtifact()
        {
            string repository = PublicMnistOnnxRuntimeLifecycleTests.ResolveRepository();
            string configured = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TRANSFORMER_UNITY_BUNDLE"
            );
            artifactDirectory = !string.IsNullOrWhiteSpace(configured)
                ? Path.GetFullPath(configured)
                : Path.Combine(
                    repository,
                    "build",
                    "udoncl",
                    "tests",
                    "cmake-package",
                    "onnx-transformer-consumer",
                    "udoncl",
                    "onnx-transformer-tiny",
                    "artifact"
                );
            artifact = ArtifactManifestValidator.Validate(artifactDirectory);
            Assert.AreEqual("generic_onnx_transformer_tiny", artifact.ProgramName);
            if (!AssetDatabase.IsValidFolder(GeneratedRoot))
                Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));
            CollectionAssert.AreEqual(
                File.ReadAllBytes(Path.Combine(artifactDirectory, "binding.cs")),
                File.ReadAllBytes(GeneratedRoot + "/Bindings/" + BindingClass + ".cs"),
                "Importer must install compiler-generated binding source byte-for-byte."
            );
            string programPath = GeneratedRoot + "/Programs/host_program.asset";
            Assert.IsFalse(string.IsNullOrEmpty(AssetDatabase.AssetPathToGUID(programPath)));
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                GeneratedRoot + "/Bindings/" + BindingClass + ".cs",
                GeneratedRoot + "/Bindings/" + BindingClass + ".asset"
            );
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
        }

        [Test]
        public void DynamicTransformerMatchesOracleWithSeparateKvOutputs()
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
            int[] rawHeap = ReadRawHeap();
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
            Assert.AreEqual(10, artifact.DeviceBuffers.Count);
            foreach (ValidatedDeviceBuffer descriptor in artifact.DeviceBuffers)
            {
                RenderTexture buffer = Load<RenderTexture>(
                    "RenderTextures/"
                        + UdonclArtifactImporter.DeviceBufferFileName(
                            descriptor.Id,
                            descriptor.Name
                        )
                );
                buffers.Add(descriptor.Id, buffer);
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    descriptor.HeapSlot,
                    buffer
                );
                PublicMnistOnnxRuntimeLifecycleTests.Clear(buffer);
            }
            Dictionary<int, Material> materials = new Dictionary<int, Material>();
            foreach (ValidatedKernel kernel in artifact.Kernels)
            {
                Material material = Load<Material>(
                    "Materials/kernel_" + kernel.Id + "_" + kernel.Name + ".mat"
                );
                materials.Add(kernel.Id, material);
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__udoncl_kernel_material_" + kernel.Id,
                    material
                );
            }
            CollectionAssert.AreEquivalent(
                new[]
                {
                    1,
                    2,
                    207,
                    217,
                    218,
                    219,
                    220,
                    221,
                    222,
                    231,
                    232,
                    233,
                    234,
                    235,
                    236,
                    237,
                    238,
                    239,
                    240,
                    241,
                    242,
                    243,
                    244,
                    245,
                    246,
                },
                materials.Keys
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
                "Transformer initialization failed after " + initializeCommands + " commands"
            );

            int[] tokens = { 1, 2, 3, 4 };
            PublicMnistOnnxRuntimeLifecycleTests.UploadLongs(
                buffers[1],
                Array.ConvertAll(tokens, value => (long)value),
                copyMaterial
            );
            float[] emptyCache = new float[32];
            PublicMnistOnnxRuntimeLifecycleTests.UploadFloats(buffers[2], emptyCache, copyMaterial);
            PublicMnistOnnxRuntimeLifecycleTests.UploadFloats(buffers[3], emptyCache, copyMaterial);
            float[][] firstExpected = Oracle(tokens, emptyCache, emptyCache);
            int firstCommands = Run(vm, program, heap, rawHeap, buffers, materials, copyMaterial);
            AssertOutputs(buffers, firstExpected);
            TestContext.Progress.WriteLine(
                "public_transformer commands={0} separate_kv_outputs=passed",
                firstCommands
            );
        }

        [OneTimeTearDown]
        public void RemoveGeneratedAssets()
        {
            RenderTexture.active = null;
            // Keep generated U# sources valid until Unity exits; precommit restores
            // generated and serialized asset snapshots after the Editor has stopped.
        }

        private int[] ReadRawHeap()
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(artifactDirectory, "raw-heap.bytes"));
            return PublicMnistOnnxRuntimeLifecycleTests.PackRawHeap(bytes);
        }

        private static int Run(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            int[] rawHeap,
            IReadOnlyDictionary<int, RenderTexture> buffers,
            IReadOnlyDictionary<int, Material> materials,
            Material copyMaterial
        )
        {
            PublicMnistOnnxRuntimeLifecycleTests.Set(
                heap,
                program,
                "__udoncl_async_command_kind",
                0
            );
            int commands = PublicMnistOnnxRuntimeLifecycleTests.RunAsyncExport(
                vm,
                program,
                heap,
                rawHeap,
                buffers,
                materials,
                copyMaterial
            );
            Assert.AreEqual(
                0,
                PublicMnistOnnxRuntimeLifecycleTests.Get<int>(heap, program, "__0_Run__ret"),
                "Run failed after " + commands + " async commands"
            );
            return commands;
        }

        private static void AssertOutputs(
            IReadOnlyDictionary<int, RenderTexture> buffers,
            float[][] expected
        )
        {
            int[] outputBuffers = { 4, 5, 6 };
            for (int output = 0; output < expected.Length; ++output)
            {
                float maximumError = 0.0f;
                Color32[] pixels = PublicMnistOnnxRuntimeLifecycleTests.ReadPixels(
                    buffers[outputBuffers[output]]
                );
                for (int index = 0; index < expected[output].Length; ++index)
                {
                    Color32 pixel = pixels[index];
                    int word = pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
                    float actual = BitConverter.Int32BitsToSingle(word);
                    maximumError = Mathf.Max(
                        maximumError,
                        Mathf.Abs(actual - expected[output][index])
                    );
                }
                TestContext.Progress.WriteLine(
                    "public_transformer output={0} max_error={1:R}",
                    outputBuffers[output],
                    maximumError
                );
                Assert.That(
                    maximumError,
                    Is.LessThan(0.002f),
                    "Transformer output buffer " + outputBuffers[output] + " mismatched."
                );
            }
        }

        private static float[][] Oracle(int[] tokens, float[] pastKey, float[] pastValue)
        {
            float[] hidden = new float[32];
            for (int token = 0; token < tokens.Length; ++token)
            for (int column = 0; column < 8; ++column)
            {
                int index = tokens[token] * 8 + column;
                hidden[token * 8 + column] = ((index * 7) % 23 - 11) / 32.0f;
            }
            float[] query = MatMul(hidden, 1);
            float[] value = MatMul(hidden, 2);
            float[] presentKey = new float[32];
            float[] presentValue = new float[32];
            float[] attention = new float[32];
            for (int index = 0; index < 32; ++index)
            {
                presentKey[index] = query[index] + pastKey[index];
                presentValue[index] = value[index] + pastValue[index];
                attention[index] = (float)Math.Tanh(presentKey[index] + presentValue[index]);
            }
            return new[] { MatMul(attention, 3, true), presentKey, presentValue };
        }

        private static float[] MatMul(float[] input, int seed, bool quantized = false)
        {
            float[] output = new float[32];
            for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 8; ++column)
            for (int inner = 0; inner < 8; ++inner)
            {
                int signedWeight = (inner * 5 + column * 3 + seed) % 17 - 8;
                if (quantized)
                    signedWeight = Mathf.Clamp(signedWeight, -8, 7);
                float weight = signedWeight / 32.0f;
                output[row * 8 + column] += input[row * 8 + inner] * weight;
            }
            return output;
        }

        private static T Load<T>(string relativePath)
            where T : UnityEngine.Object
        {
            T asset = AssetDatabase.LoadAssetAtPath<T>(GeneratedRoot + "/" + relativePath);
            Assert.NotNull(asset, "Missing generated asset " + relativePath);
            return asset;
        }
    }
}
