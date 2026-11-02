using System;
using System.Collections;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;
using UnityEngine.TestTools;
using VRC.Udon;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class UdonOnnxAotLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/onnx_int4_fixture";
        private const string GeneratedMixedRoot = "Assets/udoncl-generated/onnx_mixed_int4_fixture";
        private const string GeneratedArenaRoot = "Assets/udoncl-generated/onnx_arena_add_fixture";
        private const string GeneratedRlRoot = "Assets/udoncl-generated/onnx_rl_policy_fixture";
        private const string GeneratedTinyStandardRoot =
            "Assets/udoncl-generated/onnx_tiny_llama_standard_fixture";
        private const string GeneratedRuntimeRoot = "Assets/udoncl-generated/Runtime";
        private const string ArenaSampleScript =
            "Assets/UdonclSampleValidation/UdonOnnxArenaAddExample.cs";
        private const string ArenaSampleProgram =
            "Assets/UdonclSampleValidation/UdonOnnxArenaAddExample.asset";
        private bool arenaSampleProgramAssetExisted;

        private static string ResolveFixture(string environmentName, string relativeFallback)
        {
            string fixture = Environment.GetEnvironmentVariable(environmentName);
            if (!string.IsNullOrWhiteSpace(fixture))
                return Path.GetFullPath(fixture);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonOnnxAotLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            return Path.GetFullPath(Path.Combine(repository, relativeFallback));
        }

        [OneTimeSetUp]
        public void ImportUnityInt4ModelBundle()
        {
            arenaSampleProgramAssetExisted =
                AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(ArenaSampleProgram) != null;
            AssetDatabase.DeleteAsset(GeneratedRoot);
            string bundle = ResolveFixture(
                "UDONCL_ONNX_INT4_UNITY_BUNDLE",
                Path.Combine("build", "precommit", "onnx-aot", "unity-int4-model-bundle")
            );
            Assert.That(File.Exists(Path.Combine(bundle, "manifest.json")));
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
            string mixedBundle = ResolveFixture(
                "UDONCL_ONNX_MIXED_INT4_UNITY_BUNDLE",
                Path.Combine("build", "precommit", "onnx-aot", "unity-mixed-int4-model-bundle")
            );
            Assert.That(File.Exists(Path.Combine(mixedBundle, "manifest.json")));
            Assert.AreEqual(GeneratedMixedRoot, UdonclArtifactImporter.Import(mixedBundle));
            string arenaBundle = ResolveFixture(
                "UDONCL_ONNX_ARENA_UNITY_BUNDLE",
                Path.Combine("build", "precommit", "onnx-aot", "unity-arena-add-model-bundle")
            );
            Assert.That(File.Exists(Path.Combine(arenaBundle, "manifest.json")));
            Assert.AreEqual(GeneratedArenaRoot, UdonclArtifactImporter.Import(arenaBundle));
            string rlBundle = ResolveFixture(
                "UDONCL_ONNX_RL_UNITY_BUNDLE",
                Path.Combine("build", "precommit", "onnx-aot", "unity-rl-policy-model-bundle")
            );
            Assert.That(File.Exists(Path.Combine(rlBundle, "manifest.json")));
            Assert.AreEqual(GeneratedRlRoot, UdonclArtifactImporter.Import(rlBundle));
            string tinyStandardBundle = ResolveFixture(
                "UDONCL_ONNX_TINY_STANDARD_UNITY_BUNDLE",
                Path.Combine(
                    "build",
                    "precommit",
                    "onnx-aot",
                    "unity-tiny-llama-standard-model-bundle"
                )
            );
            Assert.That(File.Exists(Path.Combine(tinyStandardBundle, "manifest.json")));
            Assert.AreEqual(
                GeneratedTinyStandardRoot,
                UdonclArtifactImporter.Import(tinyStandardBundle)
            );
            EnsureArenaSampleProgram();
        }

        [OneTimeTearDown]
        public void RemoveImportedUnityInt4ModelBundle()
        {
            string modelProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedRoot + "/Programs/host_program.asset"
            );
            string runnerProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedRoot + "/Int4Runners/call-0000/Programs/host_program.asset"
            );
            string mixedModelProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedMixedRoot + "/Programs/host_program.asset"
            );
            string mixedRunner0ProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedMixedRoot + "/Int4Runners/call-0000/Programs/host_program.asset"
            );
            string mixedRunner1ProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedMixedRoot + "/Int4Runners/call-0001/Programs/host_program.asset"
            );
            string arenaProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedArenaRoot + "/Programs/host_program.asset"
            );
            string rlProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedRlRoot + "/Programs/host_program.asset"
            );
            string tinyStandardProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedTinyStandardRoot + "/Programs/host_program.asset"
            );
            string runtimeProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedRuntimeRoot + "/UdonOnnxRuntimeBehaviour.asset"
            );
            string arenaSampleProgramGuid = arenaSampleProgramAssetExisted
                ? null
                : AssetDatabase.AssetPathToGUID(ArenaSampleProgram);
            AssetDatabase.DeleteAsset(GeneratedRoot);
            AssetDatabase.DeleteAsset(GeneratedMixedRoot);
            AssetDatabase.DeleteAsset(GeneratedArenaRoot);
            AssetDatabase.DeleteAsset(GeneratedRlRoot);
            AssetDatabase.DeleteAsset(GeneratedTinyStandardRoot);
            AssetDatabase.DeleteAsset(GeneratedRuntimeRoot);
            if (!arenaSampleProgramAssetExisted)
                AssetDatabase.DeleteAsset(ArenaSampleProgram);
            foreach (
                string guid in new[]
                {
                    modelProgramGuid,
                    runnerProgramGuid,
                    mixedModelProgramGuid,
                    mixedRunner0ProgramGuid,
                    mixedRunner1ProgramGuid,
                    arenaProgramGuid,
                    rlProgramGuid,
                    tinyStandardProgramGuid,
                    runtimeProgramGuid,
                    arenaSampleProgramGuid,
                }
            )
                if (!string.IsNullOrEmpty(guid))
                    AssetDatabase.DeleteAsset("Assets/SerializedUdonPrograms/" + guid + ".asset");
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedMixedRoot));
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedArenaRoot));
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRlRoot));
            Assert.IsNull(
                AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedTinyStandardRoot)
            );
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRuntimeRoot));
            if (!arenaSampleProgramAssetExisted)
                Assert.IsNull(
                    AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(ArenaSampleProgram)
                );
        }

        private static void EnsureArenaSampleProgram()
        {
            Assert.NotNull(
                AssetDatabase.LoadAssetAtPath<MonoScript>(ArenaSampleScript),
                "The precommit driver did not stage the Arena Add sample."
            );
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                ArenaSampleScript,
                ArenaSampleProgram
            );
        }

#if false
        // Disabled while the arena AveragePool Unity/D3D11 zero-output defect remains unresolved.
        // See docs/investigations/2026-11-02-unity-mnist-averagepool-zero.md.
        [Test]
        public void UnityGenericMnistModelExecutesAllOperatorsNumericallyOnSdkVm()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedMnistRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.AreEqual("UdonOperatorExecutionProvider", modelAsset.ExecutionProvider);
            Assert.AreEqual(2, modelAsset.TextureArenaSlots.Length);
            Assert.AreEqual(6, modelAsset.PassMaterials.Length);
            foreach (Material material in modelAsset.PassMaterials)
            {
                Assert.IsTrue(material.shader.isSupported, material.shader.name);
                Assert.Greater(material.passCount, 0, material.shader.name);
            }
            Assert.AreEqual(
                "UdonCL/device_gpu__udondnn_arena_pooling_forward_225",
                modelAsset.PassMaterials[3].shader.name
            );
            Assert.AreEqual(
                "UdonCL/device_gpu__udonblas_arena_sgemm_221",
                modelAsset.PassMaterials[4].shader.name
            );
            Assert.AreEqual(
                "UdonCL/device_gpu__udonaten_arena_pointwise_215",
                modelAsset.PassMaterials[5].shader.name
            );
            int[] poolingArguments = { 144, 16, 24, 24, 16, 3, 3, 8, 8, 8, 8 };
            for (int index = 0; index < poolingArguments.Length; ++index)
                Assert.AreEqual(
                    poolingArguments[index],
                    modelAsset.PassMaterials[3].GetInteger("_VRC_Arg" + (index + 2))
                );
            Assert.AreEqual(1, modelAsset.PassMaterials[4].GetInteger("_VRC_Arg3"));
            Assert.AreEqual(10, modelAsset.PassMaterials[4].GetInteger("_VRC_Arg4"));
            Assert.AreEqual(144, modelAsset.PassMaterials[4].GetInteger("_VRC_Arg5"));
            Assert.AreEqual(1.0f, modelAsset.PassMaterials[4].GetFloat("_VRC_Arg8"));
            Assert.NotNull(modelAsset.PassMaterials[4].GetTexture("_VRC_Input1"));
            Assert.NotNull(modelAsset.PassMaterials[5].GetTexture("_VRC_Input1"));

            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedMnistRoot + "/Programs/host_program.asset"
                );
            Assert.NotNull(programAsset);
            IUdonProgram program = programAsset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();

            byte[] inputBytes = File.ReadAllBytes(
                ResolveFixture(
                    "UDONCL_ONNX_MNIST_INPUT",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-input.bytes")
                )
            );
            byte[] reference = File.ReadAllBytes(
                ResolveFixture(
                    "UDONCL_ONNX_MNIST_REFERENCE",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-reference.bytes")
                )
            );
            Color32[] pixels = new Color32[8192 * 2];
            for (int index = 0; index < inputBytes.Length / 4; ++index)
                pixels[index] = Encode(BitConverter.ToSingle(inputBytes, index * 4));
            Texture2D input = new Texture2D(8192, 2, TextureFormat.RGBA32, false, true);
            try
            {
                input.filterMode = FilterMode.Point;
                input.wrapMode = TextureWrapMode.Clamp;
                input.SetPixels32(pixels);
                input.Apply(false, false);
                Graphics.Blit(input, modelAsset.TextureArenaSlots[0]);
                int[] stageWords = { 9216, 9216, 9216, 144, 10, 10 };
                float[] stageTolerances = { 0.0002f, 0.0002f, 0.0002f, 0.0002f, 0.002f, 0.002f };
                for (int index = 0; index < modelAsset.PassMaterials.Length; ++index)
                {
                    int source = index % 2;
                    int destination = 1 - source;
                    Material material = modelAsset.PassMaterials[index];
                    material.SetTexture("_VRC_Input", modelAsset.TextureArenaSlots[source]);
                    Graphics.Blit(
                        modelAsset.TextureArenaSlots[source],
                        modelAsset.TextureArenaSlots[destination],
                        material
                    );
                    byte[] stageReference = File.ReadAllBytes(
                        ResolveFixture(
                            "UDONCL_ONNX_MNIST_STAGE_" + index,
                            Path.Combine(
                                "build",
                                "precommit",
                                "onnx-aot",
                                "unity-mnist-stage-" + index + ".bytes"
                            )
                        )
                    );
                    AssertTextureMatches(
                        modelAsset.TextureArenaSlots[destination],
                        stageReference,
                        stageWords[index],
                        stageTolerances[index],
                        "MNIST pass " + index
                    );
                }

                Graphics.Blit(input, modelAsset.TextureArenaSlots[0]);
                for (int index = 0; index < modelAsset.TextureArenaSlots.Length; ++index)
                    Set(heap, program, "arenaSlot" + index, modelAsset.TextureArenaSlots[index]);
                Set(heap, program, "int4Runners", Array.Empty<IUdonEventReceiver>());
                Set(heap, program, "passMaterials", modelAsset.PassMaterials);
                Set(heap, program, "resultState", modelAsset.TextureArenaSlots[0]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(0u, vm.Interpret());

                Color32[] actual = ReadPixels(modelAsset.TextureArenaSlots[0], 10);
                float maximumError = 0.0f;
                for (int index = 0; index < 10; ++index)
                    maximumError = Mathf.Max(
                        maximumError,
                        Mathf.Abs(
                            Decode(actual[index]) - BitConverter.ToSingle(reference, index * 4)
                        )
                    );
                Assert.That(maximumError, Is.LessThan(0.002f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }
#endif

        [Test]
        public void UnityRlPolicyModelExecutesAllOperatorsNumericallyOnSdkVm()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedRlRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.AreEqual("UdonOperatorExecutionProvider", modelAsset.ExecutionProvider);
            Assert.AreEqual(3, modelAsset.TextureArenaSlots.Length);
            Assert.AreEqual(10, modelAsset.PassMaterials.Length);

            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedRlRoot + "/Programs/host_program.asset"
                );
            Assert.NotNull(programAsset);
            IUdonProgram program = programAsset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();

            float[] observation = { 0.25f, -0.5f, 0.75f, -1.0f };
            float[] expected = EvaluateRlPolicy(observation);
            Color32[] pixels = new Color32[8];
            for (int index = 0; index < observation.Length; ++index)
                pixels[index] = Encode(observation[index]);
            Texture2D input = new Texture2D(8, 1, TextureFormat.RGBA32, false, true);
            try
            {
                input.filterMode = FilterMode.Point;
                input.wrapMode = TextureWrapMode.Clamp;
                input.SetPixels32(pixels);
                input.Apply(false, false);
                Graphics.Blit(input, modelAsset.TextureArenaSlots[0]);
                for (int index = 0; index < modelAsset.TextureArenaSlots.Length; ++index)
                    Set(heap, program, "arenaSlot" + index, modelAsset.TextureArenaSlots[index]);
                Set(heap, program, "int4Runners", Array.Empty<IUdonEventReceiver>());
                Set(heap, program, "passMaterials", modelAsset.PassMaterials);
                Set(heap, program, "resultState", modelAsset.TextureArenaSlots[0]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(0u, vm.Interpret());

                Color32[] actual = ReadPixels(modelAsset.TextureArenaSlots[1], 3);
                float maximumError = 0.0f;
                for (int index = 0; index < expected.Length; ++index)
                    maximumError = Mathf.Max(
                        maximumError,
                        Mathf.Abs(Decode(actual[index]) - expected[index])
                    );
                Assert.That(maximumError, Is.LessThan(0.0005f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        private static float[] EvaluateRlPolicy(float[] observation)
        {
            float[] preactivation = new float[8];
            float[] hidden = new float[8];
            for (int column = 0; column < 8; ++column)
            {
                float value = 0.05f * (column - 3);
                for (int inner = 0; inner < 4; ++inner)
                    value += observation[inner] * (((inner * 8 + column) % 9) - 4) / 16.0f;
                preactivation[column] = value;
                float tanh = (float)Math.Tanh(value);
                float sigmoid = 1.0f / (1.0f + (float)Math.Exp(-value));
                hidden[column] = tanh * sigmoid;
            }
            float[] logits = new float[3];
            float[] bias = { 0.1f, -0.05f, 0.025f };
            float maximum = float.NegativeInfinity;
            for (int column = 0; column < 3; ++column)
            {
                float value = bias[column] - 0.02f;
                for (int inner = 0; inner < 8; ++inner)
                    value += hidden[inner] * (((inner * 3 + column) % 7) - 3) / 12.0f;
                logits[column] = value * 0.75f;
                maximum = Mathf.Max(maximum, logits[column]);
            }
            float denominator = 0.0f;
            for (int index = 0; index < logits.Length; ++index)
            {
                logits[index] = (float)Math.Exp(logits[index] - maximum);
                denominator += logits[index];
            }
            for (int index = 0; index < logits.Length; ++index)
                logits[index] /= denominator;
            return logits;
        }

        [Test]
        public void UnityStandardOperatorModelExecutesArenaAddNumericallyOnSdkVm()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedArenaRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.AreEqual("UdonOperatorExecutionProvider", modelAsset.ExecutionProvider);
            Assert.AreEqual(3, modelAsset.TextureArenaSlots.Length);
            Assert.AreEqual(1, modelAsset.PassMaterials.Length);
            CollectionAssert.AreEqual(new[] { "left", "right" }, modelAsset.InputNames);
            CollectionAssert.AreEqual(new[] { "float32", "float32" }, modelAsset.InputDataTypes);
            CollectionAssert.AreEqual(new[] { 16384, 16384 }, modelAsset.InputWords);
            CollectionAssert.AreEqual(new[] { 0, 1 }, modelAsset.InputSlots);
            CollectionAssert.AreEqual(new[] { "sum" }, modelAsset.OutputNames);
            CollectionAssert.AreEqual(new[] { "float32" }, modelAsset.OutputDataTypes);
            CollectionAssert.AreEqual(new[] { 16384 }, modelAsset.OutputWords);
            CollectionAssert.AreEqual(new[] { 2 }, modelAsset.OutputSlots);
            Assert.NotNull(modelAsset.RuntimePrefab);
            Assert.NotNull(modelAsset.RuntimePrefab.GetComponent<UdonBehaviour>());
            Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour runtime =
                modelAsset.RuntimePrefab.GetComponent<Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour>();
            Assert.NotNull(runtime);
            Assert.IsTrue(runtime.IsInitialized());
            Assert.AreEqual(2, runtime.GetInputCount());
            Assert.AreEqual(1, runtime.GetOutputCount());
            Assert.AreEqual(0, runtime.FindInput("left"));
            Assert.AreEqual(1, runtime.FindInput("right"));
            Assert.AreEqual(0, runtime.FindOutput("sum"));
            Assert.AreEqual("float32", runtime.GetInputDataType(0));
            Assert.AreEqual("float32", runtime.GetOutputDataType(0));
            Assert.AreEqual(16384, runtime.GetInputWordCount(0));
            Assert.AreEqual(16384, runtime.GetInputWordCount(1));
            Assert.AreEqual(16384, runtime.GetOutputWordCount(0));
            Assert.AreEqual(-1, runtime.GetInputWordCount(-1));
            Assert.AreEqual(-1, runtime.GetOutputWordCount(1));

            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedArenaRoot + "/Programs/host_program.asset"
                );
            Assert.NotNull(programAsset);
            IUdonProgram program = programAsset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();

            const int width = 8192;
            const int height = 2;
            Color32[] leftPixels = new Color32[width * height];
            Color32[] rightPixels = new Color32[width * height];
            float[] expected = new float[width * height];
            for (int index = 0; index < expected.Length; ++index)
            {
                float left = ((index % 97) - 48) * 0.03125f;
                float right = ((index % 31) - 15) * 0.0625f;
                leftPixels[index] = Encode(left);
                rightPixels[index] = Encode(right);
                expected[index] = left + right;
            }
            Texture2D leftTexture = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
            Texture2D rightTexture = new Texture2D(
                width,
                height,
                TextureFormat.RGBA32,
                false,
                true
            );
            try
            {
                foreach (Texture2D texture in new[] { leftTexture, rightTexture })
                {
                    texture.filterMode = FilterMode.Point;
                    texture.wrapMode = TextureWrapMode.Clamp;
                }
                leftTexture.SetPixels32(leftPixels);
                leftTexture.Apply(false, false);
                rightTexture.SetPixels32(rightPixels);
                rightTexture.Apply(false, false);
                Graphics.Blit(leftTexture, modelAsset.TextureArenaSlots[0]);
                Graphics.Blit(rightTexture, modelAsset.TextureArenaSlots[1]);

                for (int index = 0; index < modelAsset.TextureArenaSlots.Length; ++index)
                    Set(heap, program, "arenaSlot" + index, modelAsset.TextureArenaSlots[index]);
                Set(heap, program, "int4Runners", Array.Empty<IUdonEventReceiver>());
                Set(heap, program, "passMaterials", modelAsset.PassMaterials);
                Set(heap, program, "resultState", modelAsset.TextureArenaSlots[0]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(0u, vm.Interpret());
                Assert.AreSame(
                    modelAsset.TextureArenaSlots[2],
                    heap.GetHeapVariable<RenderTexture>(
                        program.SymbolTable.GetAddressFromSymbol("resultState")
                    )
                );

                Color32[] actual = ReadPixels(modelAsset.TextureArenaSlots[2], width, height);
                float maximumError = 0.0f;
                for (int index = 0; index < actual.Length; ++index)
                    maximumError = Mathf.Max(
                        maximumError,
                        Mathf.Abs(Decode(actual[index]) - expected[index])
                    );
                Assert.That(maximumError, Is.EqualTo(0.0f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(leftTexture);
                UnityEngine.Object.DestroyImmediate(rightTexture);
            }
        }

        [UnityTest]
        public IEnumerator ArenaAddSampleExecutesPublicWriteRunAndAsyncReadPath()
        {
            yield return new EnterPlayMode();

            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedArenaRoot + "/Models/UdonOnnxModel.asset"
                );
            GameObject runtimeInstance = UnityEngine.Object.Instantiate(modelAsset.RuntimePrefab);
            GameObject sampleObject = new GameObject("udoncl Arena Add public API test");
            MonoScript sampleScript = AssetDatabase.LoadAssetAtPath<MonoScript>(ArenaSampleScript);
            UdonBehaviour sample = UdonclArtifactImporter.AddUdonSharpComponentAndGetBacking(
                sampleObject,
                sampleScript.GetClass()
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(runtimeInstance);
            UdonclArtifactImporter.InitializeUdonBehaviour(sample);
            yield return null;
            UdonBehaviour runtime = UdonclArtifactImporter.FindUdonSharpBacking(runtimeInstance);
            Assert.NotNull(runtime, "The instantiated runtime Udon heap was not initialized.");
            int[] runtimeInputWords =
                UdonclArtifactImporter.GetUdonProgramVariable(runtime, "inputWords") as int[];
            int[] runtimeOutputWords =
                UdonclArtifactImporter.GetUdonProgramVariable(runtime, "outputWords") as int[];
            UdonclArtifactImporter.SetUdonProgramVariable(sample, "runtime", runtime);
            UdonclArtifactImporter.SetUdonProgramVariable(sample, "left", 2.0f);
            UdonclArtifactImporter.SetUdonProgramVariable(sample, "right", 3.0f);
            UdonclArtifactImporter.SendUdonCustomEvent(sample, "Run");

            const int timeoutFrames = 300;
            int frame = 0;
            float result = 0.0f;
            while (frame < timeoutFrames && Math.Abs(result - 5.0f) > 0.0001f)
            {
                ++frame;
                yield return null;
                result = (float)UdonclArtifactImporter.GetUdonProgramVariable(sample, "result");
            }
            int status = (int)UdonclArtifactImporter.GetUdonProgramVariable(sample, "status");
            int stage = (int)UdonclArtifactImporter.GetUdonProgramVariable(sample, "stage");
            int leftWords = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(sample, "leftWordCount");
            int rightWords = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(sample, "rightWordCount");
            int outputWords = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(sample, "outputWordCount");

            yield return new ExitPlayMode();

            Assert.Less(
                frame,
                timeoutFrames,
                "Arena Add async readback did not complete; status="
                    + status
                    + " stage="
                    + stage
                    + " words="
                    + leftWords
                    + "/"
                    + rightWords
                    + "/"
                    + outputWords
                    + " runtimeWords="
                    + (runtimeInputWords == null ? "null" : string.Join(",", runtimeInputWords))
                    + "/"
                    + (runtimeOutputWords == null ? "null" : string.Join(",", runtimeOutputWords))
            );
            Assert.AreEqual(Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour.StatusSuccess, status);
            Assert.AreEqual(5.0f, result, 0.0001f);
        }

        [Test]
        public void ImportedOnnxModelCanAddItsRuntimePrefabToTheActiveScene()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedArenaRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.NotNull(modelAsset.RuntimePrefab);
            GameObject instance = null;
            try
            {
                instance = UdonOnnxModelEditorUtility.InstantiateRuntime(modelAsset);
                Assert.NotNull(instance);
                Assert.AreSame(instance, Selection.activeGameObject);
                Assert.AreEqual(
                    modelAsset.RuntimePrefab,
                    PrefabUtility.GetCorrespondingObjectFromSource(instance)
                );
                Assert.NotNull(instance.GetComponent<UdonBehaviour>());
                Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour runtime =
                    instance.GetComponent<Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour>();
                Assert.NotNull(runtime);
                Assert.IsTrue(runtime.IsInitialized());
            }
            finally
            {
                Selection.activeObject = null;
                if (instance != null)
                    UnityEngine.Object.DestroyImmediate(instance);
            }
        }

        [Test]
        public void UnityStandardOperatorTinyLlamaExecutesNumericallyOnSdkVm()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedTinyStandardRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.AreEqual("UdonOperatorExecutionProvider", modelAsset.ExecutionProvider);
            Assert.AreEqual(5, modelAsset.TextureArenaSlots.Length);
            Assert.AreEqual(18, modelAsset.PassMaterials.Length);
            CollectionAssert.AreEqual(new[] { "token_ids" }, modelAsset.InputNames);
            CollectionAssert.AreEqual(new[] { "int32" }, modelAsset.InputDataTypes);
            CollectionAssert.AreEqual(new[] { 1 }, modelAsset.InputWords);
            CollectionAssert.AreEqual(new[] { "logits" }, modelAsset.OutputNames);
            CollectionAssert.AreEqual(new[] { "float32" }, modelAsset.OutputDataTypes);
            CollectionAssert.AreEqual(new[] { 16 }, modelAsset.OutputWords);
            foreach (Material material in modelAsset.PassMaterials)
            {
                Assert.IsTrue(material.shader.isSupported, material.shader.name);
                Assert.Greater(material.passCount, 0, material.shader.name);
            }
            Assert.AreEqual(
                "UdonCL/device_gpu__udonaten_arena_embedding_226",
                modelAsset.PassMaterials[0].shader.name
            );

            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedTinyStandardRoot + "/Programs/host_program.asset"
                );
            Assert.NotNull(programAsset);
            IUdonProgram program = programAsset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();

            RenderTexture inputSlot = modelAsset.TextureArenaSlots[modelAsset.InputSlots[0]];
            Color32[] inputPixels = new Color32[inputSlot.width * inputSlot.height];
            inputPixels[0] = new Color32(3, 0, 0, 0);
            Texture2D input = new Texture2D(
                inputSlot.width,
                inputSlot.height,
                TextureFormat.RGBA32,
                false,
                true
            );
            try
            {
                input.filterMode = FilterMode.Point;
                input.wrapMode = TextureWrapMode.Clamp;
                input.SetPixels32(inputPixels);
                input.Apply(false, false);
                Graphics.Blit(input, inputSlot);
                for (int index = 0; index < modelAsset.TextureArenaSlots.Length; ++index)
                    Set(heap, program, "arenaSlot" + index, modelAsset.TextureArenaSlots[index]);
                Set(heap, program, "int4Runners", Array.Empty<IUdonEventReceiver>());
                Set(heap, program, "passMaterials", modelAsset.PassMaterials);
                Set(heap, program, "resultState", modelAsset.TextureArenaSlots[0]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(0u, vm.Interpret());

                byte[] reference = File.ReadAllBytes(
                    ResolveFixture(
                        "UDONCL_ONNX_TINY_STANDARD_REFERENCE",
                        Path.Combine(
                            "build",
                            "precommit",
                            "onnx-aot",
                            "unity-tiny-llama-standard-reference.bytes"
                        )
                    )
                );
                RenderTexture outputSlot = modelAsset.TextureArenaSlots[modelAsset.OutputSlots[0]];
                Color32[] actual = ReadPixels(outputSlot, 16);
                float maximumError = 0.0f;
                for (int index = 0; index < 16; ++index)
                    maximumError = Mathf.Max(
                        maximumError,
                        Mathf.Abs(
                            Decode(actual[index]) - BitConverter.ToSingle(reference, index * 4)
                        )
                    );
                Assert.That(maximumError, Is.LessThan(0.0005f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(input);
            }
        }

        [Test]
        public void UnityInt4ModelPrefabBindsChildRunnerResources()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.NotNull(modelAsset.RuntimePrefab);
            Assert.AreEqual(3, modelAsset.TextureArenaSlots.Length);
            UdonBehaviour model = modelAsset.RuntimePrefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(model);
            Assert.IsTrue(
                model.publicVariables.TryGetVariableValue(
                    "int4Runners",
                    out IUdonEventReceiver[] runners
                )
            );
            Assert.AreEqual(1, runners.Length);
            Assert.IsTrue(
                model.publicVariables.TryGetVariableValue(
                    "passMaterials",
                    out Material[] passMaterials
                )
            );
            Assert.IsEmpty(passMaterials);
            Transform child = modelAsset.RuntimePrefab.transform.Find("call-0000");
            Assert.NotNull(child);
            UdonBehaviour runner = child.GetComponent<UdonBehaviour>();
            Assert.NotNull(runner);
            Assert.AreSame(runner, runners[0]);
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue("pageAtlas0", out Texture atlas)
            );
            Assert.NotNull(atlas);
            Assert.AreEqual(1280, atlas.width);
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "composeState",
                    out RenderTexture composeState
                )
            );
            Assert.AreEqual(8192, composeState.width);
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "int4Material",
                    out Material int4Material
                )
            );
            Assert.NotNull(int4Material);
        }

        [Test]
        public void UnityMixedInt4ModelBindsPagedAndDirectSchedule()
        {
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                    GeneratedMixedRoot + "/Models/UdonOnnxModel.asset"
                );
            Assert.NotNull(modelAsset);
            Assert.AreEqual("UdonOperatorExecutionProvider", modelAsset.ExecutionProvider);
            CollectionAssert.AreEqual(new[] { "x" }, modelAsset.InputNames);
            CollectionAssert.AreEqual(new[] { 160 }, modelAsset.InputWords);
            CollectionAssert.AreEqual(new[] { "merged" }, modelAsset.OutputNames);
            CollectionAssert.AreEqual(new[] { 70 }, modelAsset.OutputWords);
            Assert.AreEqual(3, modelAsset.TextureArenaSlots.Length);
            Assert.AreEqual(1, modelAsset.PassMaterials.Length);
            Assert.AreEqual(
                "UdonCL/device_gpu__udonaten_arena_pointwise_215",
                modelAsset.PassMaterials[0].shader.name
            );
            Assert.NotNull(modelAsset.RuntimePrefab);
            Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour runtime =
                modelAsset.RuntimePrefab.GetComponent<Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour>();
            Assert.NotNull(runtime);
            Assert.IsTrue(runtime.IsInitialized());
            Assert.AreEqual(0, runtime.FindInput("x"));
            Assert.AreEqual(0, runtime.FindOutput("merged"));

            UdonBehaviour model = modelAsset.RuntimePrefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(model);
            Assert.IsTrue(
                model.publicVariables.TryGetVariableValue(
                    "int4Runners",
                    out IUdonEventReceiver[] runners
                )
            );
            Assert.AreEqual(2, runners.Length);
            Assert.IsTrue(
                model.publicVariables.TryGetVariableValue(
                    "passMaterials",
                    out Material[] passMaterials
                )
            );
            Assert.AreEqual(1, passMaterials.Length);
            Assert.AreSame(modelAsset.PassMaterials[0], passMaterials[0]);
            for (int index = 0; index < runners.Length; ++index)
            {
                Transform child = modelAsset.RuntimePrefab.transform.Find(
                    "call-" + index.ToString("D4")
                );
                Assert.NotNull(child);
                Assert.AreSame(child.GetComponent<UdonBehaviour>(), runners[index]);
            }

            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedMixedRoot + "/Programs/host_program.asset"
                );
            Assert.NotNull(programAsset);
            IUdonProgram program = programAsset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            Assert.DoesNotThrow(() => program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("int4Runners"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("passMaterials"));
        }

        [Test]
        public void UnityInt4ChildRunnerExecutesNumericallyOnSdkVm()
        {
            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedRoot + "/Int4Runners/call-0000/Programs/host_program.asset"
                );
            Assert.NotNull(programAsset);
            IUdonProgram program = programAsset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonOnnxModel.prefab"
            );
            UdonBehaviour runner = prefab.transform.Find("call-0000").GetComponent<UdonBehaviour>();
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue("pageAtlas0", out Texture atlas)
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "composeState",
                    out RenderTexture composeState
                )
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "accumulatorState",
                    out RenderTexture accumulatorState
                )
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue("resultA", out RenderTexture resultA)
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue("resultB", out RenderTexture resultB)
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "composeMaterial",
                    out Material composeMaterial
                )
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "int4Material",
                    out Material int4Material
                )
            );
            Assert.IsTrue(
                runner.publicVariables.TryGetVariableValue(
                    "scatterMaterial",
                    out Material scatterMaterial
                )
            );

            byte[] activationBytes = File.ReadAllBytes(
                ResolveFixture(
                    "UDONCL_ONNX_INT4_ACTIVATION",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-int4-activation.bytes")
                )
            );
            Texture2D activation = new Texture2D(160, 1, TextureFormat.RGBA32, false, true);
            try
            {
                activation.filterMode = FilterMode.Point;
                activation.wrapMode = TextureWrapMode.Clamp;
                activation.LoadRawTextureData(activationBytes);
                activation.Apply(false, false);
                Set(heap, program, "activationState", (Texture)activation);
                Set(heap, program, "composeState", composeState);
                Set(heap, program, "accumulatorState", accumulatorState);
                Set(heap, program, "resultA", resultA);
                Set(heap, program, "resultB", resultB);
                Set(heap, program, "composeMaterial", composeMaterial);
                Set(heap, program, "int4Material", int4Material);
                Set(heap, program, "scatterMaterial", scatterMaterial);
                Set(heap, program, "pageAtlas0", atlas);
                Assert.AreSame(
                    composeState,
                    heap.GetHeapVariable<RenderTexture>(
                        program.SymbolTable.GetAddressFromSymbol("composeState")
                    )
                );
                Assert.AreSame(
                    accumulatorState,
                    heap.GetHeapVariable<RenderTexture>(
                        program.SymbolTable.GetAddressFromSymbol("accumulatorState")
                    )
                );
                Assert.AreSame(
                    resultA,
                    heap.GetHeapVariable<RenderTexture>(
                        program.SymbolTable.GetAddressFromSymbol("resultA")
                    )
                );
                Assert.AreSame(
                    resultB,
                    heap.GetHeapVariable<RenderTexture>(
                        program.SymbolTable.GetAddressFromSymbol("resultB")
                    )
                );
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(0u, vm.Interpret());

                Color32[] actual = ReadPixels(resultB, 70);
                byte[] reference = File.ReadAllBytes(
                    ResolveFixture(
                        "UDONCL_ONNX_INT4_REFERENCE",
                        Path.Combine("build", "precommit", "onnx-aot", "unity-int4-reference.bytes")
                    )
                );
                float maximumError = 0.0f;
                for (int index = 0; index < 70; ++index)
                    maximumError = Mathf.Max(
                        maximumError,
                        Mathf.Abs(
                            Decode(actual[index]) - BitConverter.ToSingle(reference, index * 4)
                        )
                    );
                Assert.That(maximumError, Is.LessThan(0.01f));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(activation);
            }
        }

        [TestCase("UDONCL_ONNX_AOT_UASM", "generic-host.uasm", "pass202Material")]
        [TestCase("UDONCL_ONNX_TILED_AOT_UASM", "tiled-gemm-host.uasm", "pass200Material")]
        [TestCase("UDONCL_ONNX_RL_OPERATOR_AOT_UASM", "rl-policy-host.uasm", "pass209Material")]
        [TestCase(
            "UDONCL_ONNX_TINY_LLAMA_OPERATOR_AOT_UASM",
            "tiny-llama-standard-host.uasm",
            "pass214Material"
        )]
        public void GeneratedOperatorPlanHostAssemblesWithSdk(
            string environmentName,
            string fallbackName,
            string finalMaterial
        )
        {
            string fixture = ResolveFixture(
                environmentName,
                Path.Combine("build", "precommit", "onnx-aot", fallbackName)
            );
            Assert.That(File.Exists(fixture), "Missing compiler-emitted ONNX AOT Host fixture.");
            IUdonProgram program = UdonEditorManager.Instance.Assemble(File.ReadAllText(fixture));
            Assert.NotNull(program, "SDK assembler rejected the generated ONNX AOT Host.");
            Assert.DoesNotThrow(() => program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("initialState"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("stateA"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("stateB"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("resultState"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("pass200Material"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol(finalMaterial));
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(vm);
            Assert.IsTrue(vm.LoadProgram(program), "SDK VM rejected the generated ONNX AOT Host.");
        }

        [Test]
        public void GeneratedLlama8BModelHostAssemblesWithSdk()
        {
            string fixture = ResolveFixture(
                "UDONCL_ONNX_LLAMA_8B_MODEL_UASM",
                Path.Combine("build", "precommit", "llama-8b-onnx", "model-host.uasm")
            );
            Assert.That(File.Exists(fixture), "Missing Llama 8B whole-model Host fixture.");
            IUdonProgram program = UdonEditorManager.Instance.Assemble(File.ReadAllText(fixture));
            Assert.NotNull(program, "SDK assembler rejected the Llama 8B whole-model Host.");
            Assert.DoesNotThrow(() => program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("arenaSlot0"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("arenaSlot4"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("int4Runners"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("passMaterials"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("resultState"));
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(vm);
            Assert.IsTrue(
                vm.LoadProgram(program),
                "SDK VM rejected the Llama 8B whole-model Host."
            );
        }

        private static Color32[] ReadPixels(RenderTexture source, int width, int height = 1)
        {
            RenderTexture previous = RenderTexture.active;
            Texture2D pixels = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
            try
            {
                RenderTexture.active = source;
                pixels.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                pixels.Apply(false, false);
                return pixels.GetPixels32();
            }
            finally
            {
                RenderTexture.active = previous;
                UnityEngine.Object.DestroyImmediate(pixels);
            }
        }

        private static float Decode(Color32 bytes)
        {
            int bits = bytes.r | (bytes.g << 8) | (bytes.b << 16) | (bytes.a << 24);
            return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
        }

        private static void AssertTextureMatches(
            RenderTexture actualTexture,
            byte[] reference,
            int words,
            float tolerance,
            string label
        )
        {
            Assert.AreEqual(words * 4, reference.Length, label + " reference size");
            int width = Math.Min(words, 8192);
            int height = (words + 8191) / 8192;
            Color32[] actual = ReadPixels(actualTexture, width, height);
            float maximumError = 0.0f;
            int maximumIndex = 0;
            float maximumActual = 0.0f;
            float maximumReference = 0.0f;
            for (int index = 0; index < words; ++index)
            {
                float actualValue = Decode(actual[index]);
                float referenceValue = BitConverter.ToSingle(reference, index * 4);
                if (float.IsNaN(actualValue) || float.IsInfinity(actualValue))
                    Assert.Fail(
                        label
                            + " non-finite value at word "
                            + index
                            + ": actual="
                            + actualValue.ToString("R")
                    );
                float error = Mathf.Abs(actualValue - referenceValue);
                if (error > maximumError)
                {
                    maximumError = error;
                    maximumIndex = index;
                    maximumActual = actualValue;
                    maximumReference = referenceValue;
                }
            }
            Assert.That(
                maximumError,
                Is.LessThan(tolerance),
                label
                    + " maximum error at word "
                    + maximumIndex
                    + ": actual="
                    + maximumActual.ToString("R")
                    + " reference="
                    + maximumReference.ToString("R")
            );
        }

        private static Color32 Encode(float value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            return new Color32(bytes[0], bytes[1], bytes[2], bytes[3]);
        }

        private static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
