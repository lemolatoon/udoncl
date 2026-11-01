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
    /// fragment-parallel UdonDNN activation artifact.
    /// </summary>
    public sealed class UdonDnnActivationArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udondnn_activation";

        private static readonly int[] Modes = { 0, 1, 2, 3, 4, 5, 1, 1 };
        private static readonly int[] NanPropagation = { 0, 0, 0, 0, 0, 0, 0, 1 };
        private static readonly float[] Coefficients =
        {
            0.0f,
            0.0f,
            0.0f,
            0.8f,
            1.2f,
            0.0f,
            0.0f,
            0.0f,
        };
        private static readonly float[] Alphas =
        {
            1.0f,
            0.75f,
            -1.25f,
            1.0f,
            0.5f,
            -0.75f,
            1.0f,
            1.0f,
        };
        private static readonly float[] Betas =
        {
            0.0f,
            -0.25f,
            0.5f,
            0.125f,
            -0.5f,
            1.0f,
            0.0f,
            0.0f,
        };

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonDnnActivationArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "udondnn-device-run",
                "unity-activation-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udondnn/run_device.py before this MCP lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesReusableThreeDimensionalActivationPrefab()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonDnnActivation.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(32.0f, 1.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonDNN activation result");
            Assert.NotNull(display);
            Assert.That(display.localScale.z, Is.GreaterThan(0.0f));
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "words", 64);
            AssertPublic(behaviour, "mode", 4);
            AssertPublic(behaviour, "nanPropagation", 0);
            AssertPublic(behaviour, "coefficient", 1.2f);
            AssertPublic(behaviour, "alpha", 0.5f);
            AssertPublic(behaviour, "beta", -0.5f);
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
                Load<Texture2D>("Textures/udondnn_activation_input.asset")
            );
            RenderTexture output = Load<RenderTexture>(
                "RenderTextures/udondnn_activation_output.asset"
            );
            Set(heap, program, "outputState", output);
            Material activationMaterial = Load<Material>(
                "Materials/kernel_201_udondnn_activation_forward.mat"
            );
            Set(heap, program, "activationMaterial", activationMaterial);
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "words", 64);
            Color32[] expected = Load<Texture2D>("Textures/udondnn_activation_reference.asset")
                .GetPixels32();

            float maximumError = 0.0f;
            int comparisons = 0;
            for (int caseIndex = 0; caseIndex < Modes.Length; ++caseIndex)
            {
                Set(heap, program, "mode", Modes[caseIndex]);
                Set(heap, program, "nanPropagation", NanPropagation[caseIndex]);
                Set(heap, program, "coefficient", Coefficients[caseIndex]);
                Set(heap, program, "alpha", Alphas[caseIndex]);
                Set(heap, program, "beta", Betas[caseIndex]);
                vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
                Assert.AreEqual(
                    0u,
                    vm.Interpret(),
                    "SDK IUdonVM rejected UdonDNN activation case " + caseIndex + "."
                );
                Assert.AreEqual(64, activationMaterial.GetInt("_VRC_Arg2"));
                Assert.AreEqual(Modes[caseIndex], activationMaterial.GetInt("_VRC_Arg3"));
                Assert.AreEqual(NanPropagation[caseIndex], activationMaterial.GetInt("_VRC_Arg4"));

                Color32[] actual = ReadPixels(output, 2048);
                for (int element = 0; element < 64; ++element)
                {
                    float expectedValue = Decode(expected[caseIndex * 64 + element]);
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
            Assert.AreEqual(512, comparisons);
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
