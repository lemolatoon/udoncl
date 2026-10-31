using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEngine;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;

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
            Set<Texture>(
                heap,
                program,
                "drawingTexture",
                Load<RenderTexture>("mnist_drawing.asset")
            );
            Set(heap, program, "packedState", Load<RenderTexture>("mnist_packed.asset"));
            Set(heap, program, "logitsState", Load<RenderTexture>("mnist_logits.asset"));
            Set(heap, program, "packMaterial", Load<Material>("../Materials/mnist_pack.mat"));
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
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("_start"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM rejected the Device dispatch host.");
        }

        [OneTimeTearDown]
        public void RemoveGeneratedUnityAssets()
        {
            RenderTexture.active = null;
            AssetDatabase.DeleteAsset(GeneratedRoot);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<Object>(GeneratedRoot));
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
