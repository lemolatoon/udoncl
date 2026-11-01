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
    /// MCP-for-Unity setup, SDK IUdonVM execution, and cleanup for the reusable
    /// fragment-parallel UdonDNN softmax artifact.
    /// </summary>
    public sealed class UdonDnnSoftmaxArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udondnn_softmax";

        private static readonly int[] Algorithms = { 0, 1, 2, 0, 1, 2 };
        private static readonly int[] Modes = { 0, 0, 0, 1, 1, 1 };
        private static readonly float[] Alphas = { 1.0f, 0.75f, -1.0f, 1.25f, 0.5f, -0.75f };
        private static readonly float[] Betas = { 0.0f, -0.25f, 0.5f, 0.125f, -0.5f, 1.0f };

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonDnnSoftmaxArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "udondnn-device-run",
                "unity-softmax-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udondnn/run_device.py before this MCP lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesReusableThreeDimensionalSoftmaxPrefab()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonDnnSoftmax.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(44.0f, 1.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonDNN softmax result");
            Assert.NotNull(display);
            Assert.That(display.localScale.z, Is.GreaterThan(0.0f));
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "inputShape", 8753);
            AssertPublic(behaviour, "algorithm", 1);
            AssertPublic(behaviour, "mode", 1);
            AssertPublic(behaviour, "alpha", 1.0f);
            AssertPublic(behaviour, "beta", 0.0f);
        }

        [Test]
        public void SdkUdonVmExecutesAllAlgorithmsAndModes()
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
                Load<Texture2D>("Textures/udondnn_softmax_input.asset")
            );
            RenderTexture output = Load<RenderTexture>(
                "RenderTextures/udondnn_softmax_output.asset"
            );
            Set(heap, program, "outputState", output);
            Material softmaxMaterial = Load<Material>(
                "Materials/kernel_203_udondnn_softmax_forward.mat"
            );
            Set(heap, program, "softmaxMaterial", softmaxMaterial);
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "inputShape", 8753);
            Color32[] expected = Load<Texture2D>("Textures/udondnn_softmax_reference.asset")
                .GetPixels32();

            float maximumError = 0.0f;
            int comparisons = 0;
            for (int caseIndex = 0; caseIndex < Algorithms.Length; ++caseIndex)
            {
                Set(heap, program, "algorithm", Algorithms[caseIndex]);
                Set(heap, program, "mode", Modes[caseIndex]);
                Set(heap, program, "alpha", Alphas[caseIndex]);
                Set(heap, program, "beta", Betas[caseIndex]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(
                    0u,
                    vm.Interpret(),
                    "SDK IUdonVM rejected UdonDNN softmax case " + caseIndex + "."
                );
                Assert.AreEqual(8753, softmaxMaterial.GetInt("_VRC_Arg2"));
                Assert.AreEqual(Algorithms[caseIndex], softmaxMaterial.GetInt("_VRC_Arg3"));
                Assert.AreEqual(Modes[caseIndex], softmaxMaterial.GetInt("_VRC_Arg4"));

                Color32[] actual = ReadPixels(output, 2048);
                for (int element = 0; element < 12; ++element)
                {
                    float expectedValue = Decode(expected[caseIndex * 12 + element]);
                    float actualValue = Decode(actual[1024 + element]);
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
