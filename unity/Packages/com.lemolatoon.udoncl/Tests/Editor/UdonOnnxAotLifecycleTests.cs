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
            AssetDatabase.DeleteAsset(GeneratedRoot);
            foreach (string guid in new[] { modelProgramGuid, runnerProgramGuid })
                if (!string.IsNullOrEmpty(guid))
                    AssetDatabase.DeleteAsset("Assets/SerializedUdonPrograms/" + guid + ".asset");
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
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
        [TestCase("UDONCL_ONNX_TILED_AOT_UASM", "tiled-gemm-host.uasm", "pass211Material")]
        [TestCase(
            "UDONCL_ONNX_MNIST_OPERATOR_AOT_UASM",
            "mnist-operators-host.uasm",
            "pass213Material"
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

        private static Color32[] ReadPixels(RenderTexture source, int width)
        {
            RenderTexture previous = RenderTexture.active;
            Texture2D pixels = new Texture2D(width, 1, TextureFormat.RGBA32, false, true);
            try
            {
                RenderTexture.active = source;
                pixels.ReadPixels(new Rect(0, 0, width, 1), 0, 0);
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

        private static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
