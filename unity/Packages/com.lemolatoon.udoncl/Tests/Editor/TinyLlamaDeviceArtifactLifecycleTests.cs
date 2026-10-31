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
    /// MCP-for-Unity setup, numeric IUdonVM execution, and cleanup for the
    /// seven-pass Tiny LLaMA Device artifact.
    /// </summary>
    public sealed class TinyLlamaDeviceArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/tiny_llama";

        [OneTimeSetUp]
        public void ImportGeneratedDeviceBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(TinyLlamaDeviceArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "tiny-llama-device-run",
                "unity-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run examples/tiny-llama/run_device.py before this MCP lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesSeparateThreeDimensionalDeviceDemo()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/TinyLlamaDeviceDemo.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(14.0f, 1.5f, 0.0f), prefab.transform.position);
            Transform display = prefab.transform.Find("Tiny LLaMA logits and predicted token");
            Assert.NotNull(display);
            Assert.That(display.localScale.z, Is.GreaterThan(0.0f));
            Assert.IsNull(display.GetComponent<Collider>());
            Assert.NotNull(prefab.GetComponent<UdonBehaviour>());
        }

        [Test]
        public void SdkUdonVmExecutesDevicePassesAndMatchesReferenceLogits()
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
                "initialState",
                Load<Texture2D>("Textures/tiny_llama_initial.asset")
            );
            Set(
                heap,
                program,
                "stateA",
                Load<RenderTexture>("RenderTextures/tiny_llama_state_a.asset")
            );
            RenderTexture stateB = Load<RenderTexture>("RenderTextures/tiny_llama_state_b.asset");
            Set(heap, program, "stateB", stateB);
            string[] symbols =
            {
                "embedMaterial",
                "qkvMaterial",
                "attentionMaterial",
                "projectMaterial",
                "ffnUpMaterial",
                "ffnDownMaterial",
                "logitsMaterial",
            };
            string[] kernels =
            {
                "kernel_30_tiny_llama_embed.mat",
                "kernel_31_tiny_llama_qkv.mat",
                "kernel_32_tiny_llama_attention.mat",
                "kernel_33_tiny_llama_project.mat",
                "kernel_34_tiny_llama_ffn_up.mat",
                "kernel_35_tiny_llama_ffn_down.mat",
                "kernel_36_tiny_llama_logits.mat",
            };
            for (int index = 0; index < symbols.Length; ++index)
                Set(heap, program, symbols[index], Load<Material>("Materials/" + kernels[index]));
            Set(heap, program, "displayMaterial", Load<Material>("Materials/display.mat"));
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_start"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM rejected the Device dispatch host.");

            Texture2D reference = Load<Texture2D>("Textures/tiny_llama_reference_logits.asset");
            Color32[] expected = reference.GetPixels32();
            Color32[] actual = ReadPixels(stateB);
            float maximumError = 0.0f;
            int expectedToken = 0;
            int actualToken = 0;
            for (int token = 0; token < 16; ++token)
            {
                float expectedValue = Decode(expected[token]);
                float actualValue = Decode(actual[352 + token]);
                maximumError = Mathf.Max(maximumError, Mathf.Abs(expectedValue - actualValue));
                if (expectedValue > Decode(expected[expectedToken]))
                    expectedToken = token;
                if (actualValue > Decode(actual[352 + actualToken]))
                    actualToken = token;
            }
            Assert.That(maximumError, Is.LessThan(0.002f));
            Assert.AreEqual(expectedToken, actualToken);
            Assert.AreEqual(2, actualToken);
        }

        [OneTimeTearDown]
        public void RemoveGeneratedUnityAssets()
        {
            RenderTexture.active = null;
            AssetDatabase.DeleteAsset(GeneratedRoot);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
        }

        private static Color32[] ReadPixels(RenderTexture source)
        {
            RenderTexture previous = RenderTexture.active;
            Texture2D pixels = new Texture2D(2048, 1, TextureFormat.RGBA32, false, true);
            try
            {
                RenderTexture.active = source;
                pixels.ReadPixels(new Rect(0, 0, 2048, 1), 0, 0);
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
