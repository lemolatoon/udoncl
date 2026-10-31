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
using Object = UnityEngine.Object;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    /// <summary>
    /// MCP-for-Unity lifecycle fixture. The native setup generates the bundle;
    /// setup imports it, the SDK VM executes its real Blit host, and teardown
    /// removes every generated Unity asset even when an assertion fails.
    /// </summary>
    public sealed class MnistDeviceArtifactLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/mnist_cnn";

        [OneTimeSetUp]
        public void ImportGeneratedDeviceBundle()
        {
            AssetDatabase.DeleteAsset(GeneratedRoot);
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(MnistDeviceArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            string bundle = Path.Combine(
                repository,
                "build",
                "mnist-cnn-device-run",
                "unity-bundle"
            );
            Assert.That(
                File.Exists(Path.Combine(bundle, "manifest.json")),
                "Run examples/mnist-cnn/run_device.py before this MCP lifecycle test."
            );
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(bundle));
        }

        [Test]
        public void ImportCreatesSeparateThreeDimensionalDeviceDemo()
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/MnistDeviceDemo.prefab"
            );
            Assert.NotNull(prefab);
            Assert.AreEqual(new Vector3(8.0f, 1.5f, 0.0f), prefab.transform.position);
            Transform board = prefab.transform.Find("3D digit drawing board and prediction");
            Assert.NotNull(board);
            Assert.That(board.localScale.z, Is.GreaterThan(0.0f));
            Assert.IsNull(board.GetComponent<Collider>());
            BoxCollider trigger = prefab.GetComponent<BoxCollider>();
            Assert.NotNull(trigger);
            Assert.IsTrue(trigger.isTrigger);
            UdonBehaviour behaviour = prefab.GetComponent<UdonBehaviour>();
            Assert.NotNull(behaviour);
            Assert.IsTrue(
                behaviour.publicVariables.TryGetVariableValue(
                    "rightFinger",
                    out HumanBodyBones finger
                )
            );
            Assert.AreEqual(HumanBodyBones.RightIndexDistal, finger);
            Assert.IsTrue(
                behaviour.publicVariables.TryGetVariableValue(
                    "boardTransform",
                    out Transform boundBoard
                )
            );
            Assert.AreEqual(board, boundBoard);
        }

        [Test]
        public void BrushFragmentWritesTheDrawingRenderTexture()
        {
            RenderTexture drawing = Load<RenderTexture>("mnist_drawing.asset");
            RenderTexture scratch = Load<RenderTexture>("mnist_drawing_scratch.asset");
            Material brush = Load<Material>("../Materials/mnist_brush.mat");
            brush.SetFloat("_Clear", 1.0f);
            Graphics.Blit(drawing, scratch, brush);
            brush.SetFloat("_Clear", 0.0f);
            brush.SetFloat("_BrushX", 0.5f);
            brush.SetFloat("_BrushY", 0.5f);
            Graphics.Blit(scratch, drawing, brush);

            RenderTexture previous = RenderTexture.active;
            Texture2D pixels = new Texture2D(28, 28, TextureFormat.RGBA32, false, true);
            try
            {
                RenderTexture.active = drawing;
                pixels.ReadPixels(new Rect(0, 0, 28, 28), 0, 0);
                pixels.Apply(false, false);
                Assert.That(pixels.GetPixel(14, 14).r, Is.GreaterThan(0.5f));
                Assert.That(pixels.GetPixel(0, 0).r, Is.LessThan(0.1f));
            }
            finally
            {
                RenderTexture.active = previous;
                Object.DestroyImmediate(pixels);
            }
        }

        [Test]
        public void SdkUdonVmExecutesDeviceDispatchHost()
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
            Set(heap, program, "drawingTexture", Load<RenderTexture>("mnist_drawing.asset"));
            Set(
                heap,
                program,
                "drawingScratch",
                Load<RenderTexture>("mnist_drawing_scratch.asset")
            );
            Set(heap, program, "packedState", Load<RenderTexture>("mnist_packed.asset"));
            RenderTexture logits = Load<RenderTexture>("mnist_logits.asset");
            Set(heap, program, "logitsState", logits);
            Set(heap, program, "packMaterial", Load<Material>("../Materials/mnist_pack.mat"));
            Set(heap, program, "brushMaterial", Load<Material>("../Materials/mnist_brush.mat"));
            Set(
                heap,
                program,
                "convMaterial",
                Load<Material>("../Materials/kernel_20_mnist_conv.mat")
            );
            Set(
                heap,
                program,
                "poolMaterial",
                Load<Material>("../Materials/kernel_21_mnist_pool.mat")
            );
            Set(
                heap,
                program,
                "denseMaterial",
                Load<Material>("../Materials/kernel_22_mnist_dense.mat")
            );
            Set(heap, program, "displayMaterial", Load<Material>("../Materials/display.mat"));
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/MnistDeviceDemo.prefab"
            );
            Assert.NotNull(prefab);
            Transform board = prefab.transform.Find("3D digit drawing board and prediction");
            Assert.NotNull(board);
            Set(heap, program, "boardTransform", board);
            Set(heap, program, "rightFinger", HumanBodyBones.RightIndexDistal);
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_start"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM rejected the Device dispatch host.");

            Texture2D reference = AssetDatabase.LoadAssetAtPath<Texture2D>(
                GeneratedRoot + "/Textures/mnist_reference_blank_logits.asset"
            );
            Assert.NotNull(reference);
            Color32[] expected = reference.GetPixels32();
            Color32[] actual = ReadPixels(logits);
            float maximumError = 0.0f;
            int expectedDigit = 0;
            int actualDigit = 0;
            for (int digit = 0; digit < 10; ++digit)
            {
                float expectedValue = Decode(expected[digit]);
                float actualValue = Decode(actual[digit]);
                maximumError = Mathf.Max(maximumError, Mathf.Abs(expectedValue - actualValue));
                if (expectedValue > Decode(expected[expectedDigit]))
                    expectedDigit = digit;
                if (actualValue > Decode(actual[actualDigit]))
                    actualDigit = digit;
            }
            Assert.That(maximumError, Is.LessThan(0.02f));
            Assert.AreEqual(expectedDigit, actualDigit);
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
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<Object>(GeneratedRoot));
            if (serializedProgramPath != null)
                Assert.IsNull(AssetDatabase.LoadAssetAtPath<Object>(serializedProgramPath));
        }

        private static Color32[] ReadPixels(RenderTexture source)
        {
            RenderTexture previous = RenderTexture.active;
            Texture2D pixels = new Texture2D(10800, 1, TextureFormat.RGBA32, false, true);
            try
            {
                RenderTexture.active = source;
                pixels.ReadPixels(new Rect(0, 0, 10800, 1), 0, 0);
                pixels.Apply(false, false);
                return pixels.GetPixels32();
            }
            finally
            {
                RenderTexture.active = previous;
                Object.DestroyImmediate(pixels);
            }
        }

        private static float Decode(Color32 bytes)
        {
            int bits = bytes.r | (bytes.g << 8) | (bytes.b << 16) | (bytes.a << 24);
            return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
        }

        private static T Load<T>(string path)
            where T : Object
        {
            string assetPath = path.StartsWith("../")
                ? GeneratedRoot + "/RenderTextures/" + path
                : GeneratedRoot + "/RenderTextures/" + path;
            assetPath = assetPath.Replace("/RenderTextures/../", "/");
            T asset = AssetDatabase.LoadAssetAtPath<T>(assetPath);
            Assert.NotNull(asset, "Missing generated asset " + assetPath);
            return asset;
        }

        private static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
