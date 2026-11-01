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
    public sealed class UdonAtenTileArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/udonaten_tile";

        [OneTimeSetUp]
        public void ImportGeneratedBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonAtenTileArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "udondnn-device-run",
                "unity-udonaten-tile-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run libraries/udondnn/run_device.py before this lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ManifestAndPrefabExposeAtenTileContract()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/UdonAtenTile.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(68.0f, 1.5f, 0.0f), prefab.transform.position);
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Transform display = prefab.transform.Find("UdonATen Tile result");
            Assert.NotNull(display);
            Assert.IsNull(display.GetComponent<Collider>());
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            AssertPublic(behaviour, "inputN", 1);
            AssertPublic(behaviour, "inputC", 2);
            AssertPublic(behaviour, "inputH", 2);
            AssertPublic(behaviour, "inputW", 3);
            AssertPublic(behaviour, "repeatN", 1);
            AssertPublic(behaviour, "repeatC", 1);
            AssertPublic(behaviour, "repeatH", 2);
            AssertPublic(behaviour, "repeatW", 3);
            AssertPublic(behaviour, "outputWords", 72);
        }

        [Test]
        public void SdkUdonVmExecutesTileNumerically()
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
            Set(heap, program, "inputState", Load<Texture2D>("Textures/udonaten_tile_input.asset"));
            RenderTexture output = Load<RenderTexture>("RenderTextures/udonaten_tile_output.asset");
            Set(heap, program, "outputState", output);
            Material material = Load<Material>("Materials/kernel_209_udonaten_tile.mat");
            Set(heap, program, "tileMaterial", material);
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            Set(heap, program, "inputN", 1);
            Set(heap, program, "inputC", 2);
            Set(heap, program, "inputH", 2);
            Set(heap, program, "inputW", 3);
            Set(heap, program, "repeatN", 1);
            Set(heap, program, "repeatC", 1);
            Set(heap, program, "repeatH", 2);
            Set(heap, program, "repeatW", 3);
            Set(heap, program, "outputWords", 72);
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM rejected UdonATen Tile.");
            Assert.AreEqual(1, material.GetInt("_VRC_Arg2"));
            Assert.AreEqual(2, material.GetInt("_VRC_Arg3"));
            Assert.AreEqual(2, material.GetInt("_VRC_Arg4"));
            Assert.AreEqual(3, material.GetInt("_VRC_Arg5"));
            Assert.AreEqual(1, material.GetInt("_VRC_Arg6"));
            Assert.AreEqual(1, material.GetInt("_VRC_Arg7"));
            Assert.AreEqual(2, material.GetInt("_VRC_Arg8"));
            Assert.AreEqual(3, material.GetInt("_VRC_Arg9"));
            Assert.AreEqual(72, material.GetInt("_VRC_Arg10"));

            Color32[] expected = Load<Texture2D>("Textures/udonaten_tile_reference.asset")
                .GetPixels32();
            Color32[] actual = ReadPixels(output, 8192);
            float maximumError = 0.0f;
            for (int element = 0; element < 72; ++element)
                maximumError = Mathf.Max(
                    maximumError,
                    Mathf.Abs(Decode(actual[4096 + element]) - Decode(expected[element]))
                );
            Assert.That(maximumError, Is.EqualTo(0.0f));
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
            T asset = AssetDatabase.LoadAssetAtPath<T>(GeneratedRoot + "/" + relativePath);
            Assert.NotNull(asset, "Missing generated asset " + relativePath);
            return asset;
        }

        private static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
