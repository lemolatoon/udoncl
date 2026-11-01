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
    /// Unity import, SDK IUdonVM execution, and cleanup for signed-Q4 GEMV.
    /// </summary>
    public sealed class UdonBlasQ4GemvArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udonblas_q4_gemv";

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonBlasQ4GemvArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "udonblas-device-run",
                "unity-q4-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udonblas/run_device.py before this lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesReusableQuantizedGemvPrefab()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonBlasQ4Gemv.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(20.0f, 4.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonBLAS Q4 vector result");
            Assert.NotNull(display);
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "m", 5);
            AssertPublic(behaviour, "k", 64);
            AssertPublic(behaviour, "lda", 36);
            AssertPublic(behaviour, "incx", 2);
            AssertPublic(behaviour, "incy", 3);
            AssertPublic(behaviour, "alpha", 0.75f);
            AssertPublic(behaviour, "beta", -0.5f);
        }

        [Test]
        public void SdkUdonVmExecutesQ4GemvAndMatchesReferenceVector()
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
                "inputState",
                Load<Texture2D>("Textures/udonblas_q4_gemv_input.asset")
            );
            RenderTexture output = Load<RenderTexture>(
                "RenderTextures/udonblas_q4_gemv_output.asset"
            );
            Set(heap, program, "outputState", output);
            Set(
                heap,
                program,
                "q4Material",
                Load<Material>("Materials/kernel_101_udonblas_sgemv_q4.mat")
            );
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "m", 5);
            Set(heap, program, "k", 64);
            Set(heap, program, "lda", 36);
            Set(heap, program, "incx", 2);
            Set(heap, program, "incy", 3);
            Set(heap, program, "alpha", 0.75f);
            Set(heap, program, "beta", -0.5f);
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM rejected UdonBLAS Q4 GEMV.");

            Color32[] actual = ReadPixels(output, 8192);
            Color32[] expected = Load<Texture2D>("Textures/udonblas_q4_gemv_reference.asset")
                .GetPixels32();
            float maximumError = 0.0f;
            for (int row = 0; row < 5; ++row)
                maximumError = Mathf.Max(
                    maximumError,
                    Mathf.Abs(Decode(actual[7168 + row]) - Decode(expected[row]))
                );
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
