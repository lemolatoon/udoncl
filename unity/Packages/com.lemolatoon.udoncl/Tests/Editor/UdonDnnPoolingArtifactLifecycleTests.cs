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
    /// fragment-parallel UdonDNN pooling artifact.
    /// </summary>
    public sealed class UdonDnnPoolingArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udondnn_pooling";

        private static readonly int[] Modes = { 0, 3, 1, 2, 0, 0 };
        private static readonly int[] NanPropagation = { 0, 0, 0, 0, 1, 0 };
        private static readonly float[] Alphas = { 1.0f, 0.75f, -1.0f, 1.25f, 1.0f, 0.5f };
        private static readonly float[] Betas = { 0.0f, -0.25f, 0.5f, 0.125f, 0.0f, -0.5f };

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonDnnPoolingArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "udondnn-device-run",
                "unity-pooling-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udondnn/run_device.py before this MCP lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesReusableThreeDimensionalPoolingPrefab()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonDnnPooling.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(38.0f, 1.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonDNN pooling result");
            Assert.NotNull(display);
            Assert.That(display.localScale.z, Is.GreaterThan(0.0f));
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "inputShape", 17441);
            AssertPublic(behaviour, "poolingShape", 2232627);
            AssertPublic(behaviour, "poolingMode", 0);
            AssertPublic(behaviour, "nanPropagation", 0);
            AssertPublic(behaviour, "outputShape", 8737);
            AssertPublic(behaviour, "alpha", 1.0f);
            AssertPublic(behaviour, "beta", 0.0f);
        }

        [Test]
        public void SdkUdonVmExecutesAllModesAndNanPolicies()
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
                Load<Texture2D>("Textures/udondnn_pooling_input.asset")
            );
            RenderTexture output = Load<RenderTexture>(
                "RenderTextures/udondnn_pooling_output.asset"
            );
            Set(heap, program, "outputState", output);
            Material poolingMaterial = Load<Material>(
                "Materials/kernel_202_udondnn_pooling_forward.mat"
            );
            Set(heap, program, "poolingMaterial", poolingMaterial);
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "inputShape", 17441);
            Set(heap, program, "poolingShape", 2232627);
            Set(heap, program, "outputShape", 8737);
            Color32[] expected = Load<Texture2D>("Textures/udondnn_pooling_reference.asset")
                .GetPixels32();

            float maximumError = 0.0f;
            int comparisons = 0;
            for (int caseIndex = 0; caseIndex < Modes.Length; ++caseIndex)
            {
                Set(heap, program, "poolingMode", Modes[caseIndex]);
                Set(heap, program, "nanPropagation", NanPropagation[caseIndex]);
                Set(heap, program, "alpha", Alphas[caseIndex]);
                Set(heap, program, "beta", Betas[caseIndex]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(
                    0u,
                    vm.Interpret(),
                    "SDK IUdonVM rejected UdonDNN pooling case " + caseIndex + "."
                );
                Assert.AreEqual(17441, poolingMaterial.GetInt("_VRC_Arg2"));
                Assert.AreEqual(2232627, poolingMaterial.GetInt("_VRC_Arg3"));
                Assert.AreEqual(Modes[caseIndex], poolingMaterial.GetInt("_VRC_Arg4"));
                Assert.AreEqual(NanPropagation[caseIndex], poolingMaterial.GetInt("_VRC_Arg5"));
                Assert.AreEqual(8737, poolingMaterial.GetInt("_VRC_Arg6"));

                Color32[] actual = ReadPixels(output, 2048);
                for (int element = 0; element < 8; ++element)
                {
                    float expectedValue = Decode(expected[caseIndex * 8 + element]);
                    float actualValue = Decode(actual[1024 + element]);
                    if (float.IsNaN(expectedValue) && float.IsNaN(actualValue))
                    {
                        ++comparisons;
                        continue;
                    }
                    Assert.AreEqual(
                        float.IsNaN(expectedValue),
                        float.IsNaN(actualValue),
                        "One-sided NaN at case " + caseIndex + ", element " + element + "."
                    );
                    maximumError = Mathf.Max(maximumError, Mathf.Abs(expectedValue - actualValue));
                    ++comparisons;
                }
            }
            Assert.AreEqual(48, comparisons);
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
