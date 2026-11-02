using System;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;
using VRC.Udon;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class UdonOnnxAotLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/onnx_int4_fixture";
        private const string GeneratedArenaRoot = "Assets/udoncl-generated/onnx_arena_add_fixture";
        private const string GeneratedRlRoot = "Assets/udoncl-generated/onnx_rl_policy_fixture";

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
            AssetDatabase.DeleteAsset(GeneratedRoot);
            string bundle = ResolveFixture(
                "UDONCL_ONNX_INT4_UNITY_BUNDLE",
                Path.Combine("build", "precommit", "onnx-aot", "unity-int4-model-bundle")
            );
            Assert.That(File.Exists(Path.Combine(bundle, "manifest.json")));
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
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
            string arenaProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedArenaRoot + "/Programs/host_program.asset"
            );
            string rlProgramGuid = AssetDatabase.AssetPathToGUID(
                GeneratedRlRoot + "/Programs/host_program.asset"
            );
            AssetDatabase.DeleteAsset(GeneratedRoot);
            AssetDatabase.DeleteAsset(GeneratedArenaRoot);
            AssetDatabase.DeleteAsset(GeneratedRlRoot);
            foreach (
                string guid in new[]
                {
                    modelProgramGuid,
                    runnerProgramGuid,
                    arenaProgramGuid,
                    rlProgramGuid,
                }
            )
                if (!string.IsNullOrEmpty(guid))
                    AssetDatabase.DeleteAsset("Assets/SerializedUdonPrograms/" + guid + ".asset");
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedArenaRoot));
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRlRoot));
        }

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
        [TestCase(
            "UDONCL_ONNX_MNIST_OPERATOR_AOT_UASM",
            "mnist-operators-host.uasm",
            "pass205Material"
        )]
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
