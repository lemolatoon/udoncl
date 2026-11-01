using System;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
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
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            object syncMethod = typeof(UdonBehaviour)
                .GetProperty("SyncMethod")
                ?.GetValue(behaviour);
            Assert.NotNull(syncMethod);
            Assert.AreEqual("Manual", syncMethod.ToString());
            Assert.NotNull(prefab.GetComponent<BoxCollider>());
            Assert.NotNull(prefab.GetComponentInChildren<Canvas>(true));
            Assert.NotNull(prefab.GetComponentInChildren<InputField>(true));
            Assert.NotNull(
                prefab.transform.Find("Shared world chat/Chat panel/Shared conversation history")
            );
            Assert.IsTrue(
                behaviour.publicVariables.TryGetVariableValue("promptInput", out InputField input)
            );
            Assert.NotNull(input);
            Assert.IsTrue(
                behaviour.publicVariables.TryGetVariableValue("historyText", out Text history)
            );
            Assert.NotNull(history);
            Assert.IsTrue(
                behaviour.publicVariables.TryGetVariableValue(
                    "responseWords",
                    out string[] responseWords
                )
            );
            Assert.AreEqual(16, responseWords.Length);
            Assert.IsTrue(
                behaviour.publicVariables.TryGetVariableValue(
                    "readbackPixels",
                    out Color32[] readbackPixels
                )
            );
            Assert.AreEqual(1, readbackPixels.Length);
        }

        [Test]
        public void ImportIncludesOnnxModelAndUdonLlamaExecutionPlan()
        {
            TextAsset model = AssetDatabase.LoadAssetAtPath<TextAsset>(
                GeneratedRoot + "/Models/model.onnx.bytes"
            );
            Assert.NotNull(model);
            Assert.That(model.bytes.Length, Is.GreaterThan(5000));
            TextAsset plan = AssetDatabase.LoadAssetAtPath<TextAsset>(
                GeneratedRoot + "/Models/execution-plan.json.bytes"
            );
            Assert.NotNull(plan);
            StringAssert.Contains("\"runtime\": \"udon-onnx\"", plan.text);
            StringAssert.Contains("\"profile\": \"udon-tiny-llama-f32\"", plan.text);
            StringAssert.Contains("UdonLlamaExecutionProvider", plan.text);
            StringAssert.Contains("udonatenRmsNorm", plan.text);
            StringAssert.Contains("udonatenAdd", plan.text);
            StringAssert.Contains("udonatenMul", plan.text);
            StringAssert.Contains("udonatenSigmoid", plan.text);
            StringAssert.Contains("udonblasSgemm", plan.text);
            string modelAssetPath = GeneratedRoot + "/Models/UdonOnnxModel.asset";
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(modelAssetPath);
            Assert.NotNull(modelAsset);
            Resources.UnloadAsset(modelAsset);
            AssetDatabase.ImportAsset(
                modelAssetPath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            modelAsset = AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                modelAssetPath
            );
            Assert.NotNull(modelAsset);
            Assert.AreSame(model, modelAsset.Model);
            Assert.AreSame(plan, modelAsset.ExecutionPlan);
            Assert.AreEqual("UdonLlamaExecutionProvider", modelAsset.ExecutionProvider);
            Assert.AreEqual("udon-tiny-llama-f32", modelAsset.Profile);
            Assert.NotNull(modelAsset.HostProgram);
            Assert.AreEqual(7, modelAsset.PassMaterials.Length);
            CollectionAssert.AllItemsAreNotNull(modelAsset.PassMaterials);
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
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_runDeviceFixture"));
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

            Material argmaxMaterial = Load<Material>("Materials/tiny_llama_argmax.mat");
            RenderTexture argmax = Load<RenderTexture>("RenderTextures/tiny_llama_argmax.asset");
            argmaxMaterial.SetTexture("_Logits", stateB);
            Graphics.Blit(stateB, argmax, argmaxMaterial);
            Color32 encodedPrediction = ReadPixels(argmax, 1)[0];
            Assert.AreEqual(actualToken, encodedPrediction.r);
        }

        [Test]
        public void HostAssemblyMakesTheInputterOwnerAndRemoteClientsOnlyRefreshHistory()
        {
            string sourcePath = Path.GetFullPath(GeneratedRoot + "/Sources/host.uasm.txt");
            string assembly = File.ReadAllText(sourcePath);
            StringAssert.Contains(".sync sharedHistory, none", assembly);
            StringAssert.Contains(".sync sharedToken3, none", assembly);
            int ownership = assembly.IndexOf(
                "VRCSDKBaseNetworking.__SetOwner__VRCSDKBaseVRCPlayerApi_UnityEngineGameObject__SystemVoid",
                StringComparison.Ordinal
            );
            int dispatch = assembly.IndexOf("JUMP, __dispatchDevice", StringComparison.Ordinal);
            Assert.That(ownership, Is.GreaterThanOrEqualTo(0));
            Assert.That(dispatch, Is.GreaterThan(ownership));
            StringAssert.Contains("VRCSDK3RenderingVRCAsyncGPUReadback.__Request__", assembly);
            StringAssert.Contains(
                "VRCUdonCommonInterfacesIUdonEventReceiver.__RequestSerialization__SystemVoid",
                assembly
            );
            int deserialization = assembly.IndexOf(
                ".export _onDeserialization",
                StringComparison.Ordinal
            );
            string remotePath = assembly.Substring(deserialization);
            StringAssert.Contains("JUMP, __refreshHistory", remotePath);
            StringAssert.DoesNotContain("JUMP, __dispatchDevice", remotePath);
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

        private static Color32[] ReadPixels(RenderTexture source, int width = 2048)
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
