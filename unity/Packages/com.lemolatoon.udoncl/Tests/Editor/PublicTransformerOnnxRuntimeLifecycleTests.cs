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
        private string serializedProgramPath;
        private HashSet<string> serializedProgramsBefore;
        private ValidatedArtifact artifact;

        [OneTimeSetUp]
        public void ImportArtifact()
        {
            string repository = PublicMnistOnnxRuntimeLifecycleTests.ResolveRepository();
            serializedProgramsBefore =
                PublicMnistOnnxRuntimeLifecycleTests.SerializedProgramAssets();
            AssetDatabase.DeleteAsset(GeneratedRoot);
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
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));
            CollectionAssert.AreEqual(
                File.ReadAllBytes(Path.Combine(artifactDirectory, "binding.cs")),
                File.ReadAllBytes(GeneratedRoot + "/Bindings/" + BindingClass + ".cs"),
                "Importer must install compiler-generated binding source byte-for-byte."
            );
            string programPath = GeneratedRoot + "/Programs/host_program.asset";
            string guid = AssetDatabase.AssetPathToGUID(programPath);
            Assert.IsFalse(string.IsNullOrEmpty(guid));
            serializedProgramPath = "Assets/SerializedUdonPrograms/" + guid + ".asset";
        }

        [Test]
        public void DynamicTransformerMatchesOracleAndReusesKvOutputs()
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

            Dictionary<int, RenderTexture> buffers = new Dictionary<int, RenderTexture>();
            Assert.AreEqual(20, artifact.DeviceBuffers.Count);
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
                new[] { 1, 2, 207, 219, 220, 231, 232, 233, 234 },
                materials.Keys
            );
            Material copyMaterial = Load<Material>("Materials/buffer-copy.mat");

            int[] tokens = { 1, 2, 3, 4 };
            WriteTokens(vm, program, heap, tokens);
            float[] emptyCache = new float[32];
            WriteFloatInput(vm, program, heap, 1, emptyCache);
            WriteFloatInput(vm, program, heap, 2, emptyCache);
            float[][] firstExpected = Oracle(tokens, emptyCache, emptyCache);
            int firstCommands = Run(vm, program, heap, rawHeap, buffers, materials, copyMaterial);
            AssertOutputs(vm, program, heap, firstExpected);

            Bind(vm, program, heap, 1, 1);
            Bind(vm, program, heap, 2, 2);
            float[][] secondExpected = Oracle(tokens, firstExpected[1], firstExpected[2]);
            int secondCommands = Run(vm, program, heap, rawHeap, buffers, materials, copyMaterial);
            AssertOutputs(vm, program, heap, secondExpected);
            TestContext.Progress.WriteLine(
                "public_transformer first_commands={0} second_commands={1} kv_binding=passed",
                firstCommands,
                secondCommands
            );
        }

        [OneTimeTearDown]
        public void RemoveGeneratedAssets()
        {
            RenderTexture.active = null;
            AssetDatabase.DeleteAsset(GeneratedRoot);
            if (!string.IsNullOrEmpty(serializedProgramPath))
                AssetDatabase.DeleteAsset(serializedProgramPath);
            if (serializedProgramsBefore != null)
                foreach (
                    string path in PublicMnistOnnxRuntimeLifecycleTests.SerializedProgramAssets()
                )
                    if (!serializedProgramsBefore.Contains(path))
                        AssetDatabase.DeleteAsset(path);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
            if (serializedProgramsBefore != null)
                CollectionAssert.AreEquivalent(
                    serializedProgramsBefore,
                    PublicMnistOnnxRuntimeLifecycleTests.SerializedProgramAssets(),
                    "Transformer test leaked a serialized Udon program."
                );
        }

        private int[] ReadRawHeap()
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(artifactDirectory, "raw-heap.bytes"));
            return PublicMnistOnnxRuntimeLifecycleTests.PackRawHeap(bytes);
        }

        private static void WriteTokens(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            int[] values
        )
        {
            for (int index = 0; index < values.Length; ++index)
            {
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__1_input_index__param",
                    0
                );
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__1_element_index__param",
                    index
                );
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__1_value__param",
                    values[index]
                );
                PublicMnistOnnxRuntimeLifecycleTests.Execute(vm, program, "__0_TryWriteInputInt32");
                Assert.AreEqual(
                    0,
                    PublicMnistOnnxRuntimeLifecycleTests.Get<int>(
                        heap,
                        program,
                        "__0___0_TryWriteInputInt32__ret"
                    ),
                    "TryWriteInputInt32 failed for token element " + index
                );
            }
        }

        private static void WriteFloatInput(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            int input,
            float[] values
        )
        {
            for (int index = 0; index < values.Length; ++index)
            {
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__0_input_index__param",
                    input
                );
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__0_element_index__param",
                    index
                );
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__0_value__param",
                    values[index]
                );
                PublicMnistOnnxRuntimeLifecycleTests.Execute(vm, program, "__0_TryWriteInputFloat");
                Assert.AreEqual(
                    0,
                    PublicMnistOnnxRuntimeLifecycleTests.Get<int>(
                        heap,
                        program,
                        "__0___0_TryWriteInputFloat__ret"
                    ),
                    "TryWriteInputFloat failed for input " + input + " element " + index
                );
            }
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
                "__udoncl_async_resume_state",
                0
            );
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
                PublicMnistOnnxRuntimeLifecycleTests.Get<int>(heap, program, "__0_StartRun__ret"),
                "StartRun failed after " + commands + " async commands"
            );
            return commands;
        }

        private static void Bind(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            int output,
            int input
        )
        {
            PublicMnistOnnxRuntimeLifecycleTests.Set(
                heap,
                program,
                "__1_output_index__param",
                output
            );
            PublicMnistOnnxRuntimeLifecycleTests.Set(
                heap,
                program,
                "__2_input_index__param",
                input
            );
            PublicMnistOnnxRuntimeLifecycleTests.Execute(vm, program, "__0_TryBindOutputToInput");
            Assert.AreEqual(
                0,
                PublicMnistOnnxRuntimeLifecycleTests.Get<int>(
                    heap,
                    program,
                    "__0___0_TryBindOutputToInput__ret"
                ),
                "TryBindOutputToInput failed for output " + output + " input " + input
            );
        }

        private static void AssertOutputs(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            float[][] expected
        )
        {
            float maximumError = 0.0f;
            for (int output = 0; output < expected.Length; ++output)
            for (int index = 0; index < expected[output].Length; ++index)
            {
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__0_output_index__param",
                    output
                );
                PublicMnistOnnxRuntimeLifecycleTests.Set(
                    heap,
                    program,
                    "__2_element_index__param",
                    index
                );
                PublicMnistOnnxRuntimeLifecycleTests.Execute(vm, program, "__0_ReadOutputFloat");
                float actual = PublicMnistOnnxRuntimeLifecycleTests.Get<float>(
                    heap,
                    program,
                    "__0___0_ReadOutputFloat__ret"
                );
                maximumError = Mathf.Max(maximumError, Mathf.Abs(actual - expected[output][index]));
            }
            Assert.That(maximumError, Is.LessThan(0.002f));
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
