using System;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using VRC.Udon;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    /// <summary>
    /// MCP-for-Unity setup, SDK IUdonVM execution, and cleanup for reusable
    /// fragment-parallel UdonDNN OpTensor broadcasting.
    /// </summary>
    public sealed class UdonDnnOpTensorArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udondnn_op_tensor";

        private static readonly int[] AShapes = { 8753, 8721, 8465, 4369, 8753, 8753 };
        private static readonly int[] BShapes = { 4401, 8753, 4625, 8753, 8753, 8753 };
        private static readonly int[] Operations = { 0, 1, 2, 3, 2, 3 };
        private static readonly int[] NanPropagation = { 0, 0, 0, 1, 1, 0 };
        private static readonly float[] AlphaA = { 1.0f, -1.0f, 1.25f, 0.5f, 0.75f, 1.0f };
        private static readonly float[] AlphaB = { 0.75f, 0.5f, -0.75f, 1.5f, 1.25f, -1.0f };
        private static readonly float[] Betas = { -0.25f, 0.25f, 0.0f, -0.5f, 0.125f, 0.5f };

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonDnnOpTensorArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "udondnn-device-run",
                "unity-op-tensor-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udondnn/run_device.py before this MCP lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesReusableThreeDimensionalOpTensorPrefab()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonDnnOpTensor.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(56.0f, 1.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonDNN OpTensor result");
            Assert.NotNull(display);
            Assert.That(display.localScale.z, Is.GreaterThan(0.0f));
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "aShape", 8753);
            AssertPublic(behaviour, "bShape", 4401);
            AssertPublic(behaviour, "cShape", 8753);
            AssertPublic(behaviour, "operation", 0);
            AssertPublic(behaviour, "nanPropagation", 0);
            AssertPublic(behaviour, "alphaA", 1.0f);
            AssertPublic(behaviour, "alphaB", 0.75f);
            AssertPublic(behaviour, "beta", -0.25f);
        }

        [Test]
        public void SdkUdonVmExecutesBroadcastOperationsAndNanPolicies()
        {
            UdonAssemblyProgramAsset asset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(
                    GeneratedRoot + "/Programs/host_program.asset"
                );
            Assert.NotNull(asset);
            IUdonProgram program = asset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();
            Set(
                heap,
                program,
                "packedState",
                Load<Texture2D>("Textures/udondnn_op_tensor_input.asset")
            );
            RenderTexture output = Load<RenderTexture>(
                "RenderTextures/udondnn_op_tensor_output.asset"
            );
            Set(heap, program, "outputState", output);
            Material opTensorMaterial = Load<Material>(
                "Materials/kernel_205_udondnn_op_tensor.mat"
            );
            Set(heap, program, "opTensorMaterial", opTensorMaterial);
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "cShape", 8753);
            Color32[] expected = Load<Texture2D>("Textures/udondnn_op_tensor_reference.asset")
                .GetPixels32();

            float maximumError = 0.0f;
            int comparisons = 0;
            for (int caseIndex = 0; caseIndex < Operations.Length; ++caseIndex)
            {
                Set(heap, program, "aShape", AShapes[caseIndex]);
                Set(heap, program, "bShape", BShapes[caseIndex]);
                Set(heap, program, "operation", Operations[caseIndex]);
                Set(heap, program, "nanPropagation", NanPropagation[caseIndex]);
                Set(heap, program, "alphaA", AlphaA[caseIndex]);
                Set(heap, program, "alphaB", AlphaB[caseIndex]);
                Set(heap, program, "beta", Betas[caseIndex]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(
                    0u,
                    vm.Interpret(),
                    "SDK IUdonVM rejected UdonDNN OpTensor case " + caseIndex + "."
                );
                Assert.AreEqual(AShapes[caseIndex], opTensorMaterial.GetInteger("_VRC_Arg2"));
                Assert.AreEqual(BShapes[caseIndex], opTensorMaterial.GetInteger("_VRC_Arg3"));
                Assert.AreEqual(8753, opTensorMaterial.GetInteger("_VRC_Arg4"));
                Assert.AreEqual(Operations[caseIndex], opTensorMaterial.GetInteger("_VRC_Arg5"));
                Assert.AreEqual(
                    NanPropagation[caseIndex],
                    opTensorMaterial.GetInteger("_VRC_Arg6")
                );

                Color32[] actual = ReadPixels(output, 2048);
                for (int element = 0; element < 12; ++element)
                {
                    float expectedValue = Decode(expected[caseIndex * 12 + element]);
                    float actualValue = Decode(actual[1536 + element]);
                    if (float.IsNaN(expectedValue) && float.IsNaN(actualValue))
                    {
                        ++comparisons;
                        continue;
                    }
                    maximumError = Mathf.Max(maximumError, Mathf.Abs(expectedValue - actualValue));
                    ++comparisons;
                }
            }
            Assert.AreEqual(72, comparisons);
            Assert.That(maximumError, Is.LessThan(0.0005f));
        }

        [OneTimeTearDown]
        public void RemoveGeneratedUnityAssets()
        {
            RenderTexture.active = null;
            string programPath = GeneratedRoot + "/Programs/host_program.asset";
            string programGuid = AssetDatabase.AssetPathToGUID(programPath);
            AssetDatabase.DeleteAsset(GeneratedRoot);
            string serializedProgramPath = string.IsNullOrEmpty(programGuid)
                ? null
                : "Assets/SerializedUdonPrograms/" + programGuid + ".asset";
            if (serializedProgramPath != null)
                AssetDatabase.DeleteAsset(serializedProgramPath);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
            if (serializedProgramPath != null)
                Assert.IsNull(
                    AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(serializedProgramPath)
                );
        }

        private static void AssertPublic<T>(UdonBehaviour behaviour, string symbol, T expected)
        {
            Assert.IsTrue(behaviour.publicVariables.TryGetVariableValue(symbol, out T actual));
            Assert.AreEqual(expected, actual);
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

        private static T Load<T>(string relativePath)
            where T : UnityEngine.Object
        {
            string path = GeneratedRoot + "/" + relativePath;
            T asset = AssetDatabase.LoadAssetAtPath<T>(path);
            Assert.NotNull(asset, "Missing generated asset " + path);
            return asset;
        }

        private static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
