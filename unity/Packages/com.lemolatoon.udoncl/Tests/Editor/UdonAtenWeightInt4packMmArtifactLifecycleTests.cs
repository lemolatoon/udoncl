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
    /// Unity import, SDK IUdonVM execution, and cleanup for ATen INT4 MM.
    /// </summary>
    public sealed class UdonAtenWeightInt4packMmArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udonaten_weight_int4pack_mm";
        private string bundle;

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonAtenWeightInt4packMmArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            bundle = Path.Combine(
                repository,
                "build",
                "udondnn-device-run",
                "unity-udonaten-int4-mm-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udondnn/run_device.py before this lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ManifestKeepsThePublicAtenOperatorOwnershipAndShapes()
        {
            string manifest = File.ReadAllText(Path.Combine(bundle, "manifest.json"));
            StringAssert.Contains("\"artifact_stage\": \"unity-udonaten-int4-mm\"", manifest);
            StringAssert.Contains("\"c_symbol\": \"udonatenWeightInt4packMm\"", manifest);
            StringAssert.Contains("\"packed_weight_shape\"", manifest);
            StringAssert.Contains("\"scales_and_zeros_shape\"", manifest);
            StringAssert.DoesNotContain("udonblasExt", manifest);
            StringAssert.DoesNotContain("udondnnExt", manifest);
        }

        [Test]
        public void ImportCreatesReusableAtenInt4Prefab()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonAtenWeightInt4packMm.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(24.0f, 4.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonATen INT4 matrix result");
            Assert.NotNull(display);
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "n", 8);
            AssertPublic(behaviour, "k", 128);
            AssertPublic(behaviour, "rowStride", 72);
        }

        [Test]
        public void SdkUdonVmExecutesAtenInt4MmAndMatchesReferenceVector()
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
                "weightPage0",
                Load<Texture2D>("Textures/udonaten_weight_int4pack_mm_weight_page_0.asset")
            );
            Set(
                heap,
                program,
                "weightPage1",
                Load<Texture2D>("Textures/udonaten_weight_int4pack_mm_weight_page_1.asset")
            );
            Set(
                heap,
                program,
                "activationState",
                Load<Texture2D>("Textures/udonaten_weight_int4pack_mm_activation.asset")
            );
            Set(
                heap,
                program,
                "assemblyState",
                Load<RenderTexture>("RenderTextures/udonaten_weight_int4pack_mm_assembly.asset")
            );
            Set(
                heap,
                program,
                "partialState",
                Load<RenderTexture>("RenderTextures/udonaten_weight_int4pack_mm_partial.asset")
            );
            RenderTexture output = Load<RenderTexture>(
                "RenderTextures/udonaten_weight_int4pack_mm_output.asset"
            );
            Set(heap, program, "outputState", output);
            Set(heap, program, "composeMaterial", Load<Material>("Materials/page-compose.mat"));
            Set(
                heap,
                program,
                "int4Material",
                Load<Material>("Materials/kernel_208_udonaten_weight_int4pack_mm.mat")
            );
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "n", 8);
            Set(heap, program, "k", 128);
            Set(heap, program, "rowStride", 72);
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM rejected UdonATen INT4 MM.");

            Color32[] actual = ReadPixels(output, 8192);
            Color32[] expected = Load<Texture2D>(
                    "Textures/udonaten_weight_int4pack_mm_reference.asset"
                )
                .GetPixels32();
            float maximumError = 0.0f;
            for (int row = 0; row < 8; ++row)
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
