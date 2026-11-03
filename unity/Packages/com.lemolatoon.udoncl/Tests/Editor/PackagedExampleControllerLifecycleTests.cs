using System;
using System.Collections;
using NUnit.Framework;
using UdonSharp;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using VRC.Udon;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class PackagedExampleControllerLifecycleTests
    {
        private const string HelloRoot = "Assets/udoncl-generated/hello_dispatch";
        private const string SaxpyRoot = "Assets/udoncl-generated/saxpy_demo";
        private const string GeneratedNamespace = "Lemolatoon.Udoncl.Generated.";

        [OneTimeSetUp]
        public void FinalizePreparedExamples()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_PACKAGED_EXAMPLE_VALIDATION") != "1")
                return;
            EnsureCompiledPrograms(
                HelloRoot,
                "UdonclHelloDispatchBinding",
                "UdonclHelloDispatchExample"
            );
            EnsureCompiledPrograms(SaxpyRoot, "UdonclSaxpyDemoBinding", "UdonclSaxpyDemoExample");
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
            Assert.NotNull(LoadPrefab(HelloRoot, "UdonclHelloDispatchBinding"));
            Assert.NotNull(LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding"));
        }

        [Test]
        public void ImportedPrefabsContainBoundTypedControllers()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_PACKAGED_EXAMPLE_VALIDATION") != "1")
                return;
            AssertControllerBinding(
                LoadPrefab(HelloRoot, "UdonclHelloDispatchBinding"),
                "UdonclHelloDispatchBinding",
                "UdonclHelloDispatchExample"
            );
            AssertControllerBinding(
                LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding"),
                "UdonclSaxpyDemoBinding",
                "UdonclSaxpyDemoExample"
            );
        }

        [UnityTest]
        public IEnumerator ImportedControllersExecuteNumericallyThroughPublicApi()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_PACKAGED_EXAMPLE_VALIDATION") != "1")
                yield break;

            yield return new EnterPlayMode();
            GameObject hello = UnityEngine.Object.Instantiate(
                LoadPrefab(HelloRoot, "UdonclHelloDispatchBinding")
            );
            GameObject saxpy = UnityEngine.Object.Instantiate(
                LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding")
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(hello);
            UdonclArtifactImporter.InitializeUdonBehaviours(saxpy);
            yield return null;

            UdonBehaviour helloController = UdonclArtifactImporter.FindUdonSharpBacking(
                hello,
                GeneratedNamespace + "UdonclHelloDispatchExample"
            );
            UdonBehaviour saxpyController = UdonclArtifactImporter.FindUdonSharpBacking(
                saxpy,
                GeneratedNamespace + "UdonclSaxpyDemoExample"
            );
            Assert.NotNull(helloController);
            Assert.NotNull(saxpyController);
            AssertRuntimeBinding(hello, helloController, "UdonclHelloDispatchBinding");
            AssertRuntimeBinding(saxpy, saxpyController, "UdonclSaxpyDemoBinding");
            UdonclArtifactImporter.SendUdonCustomEvent(helloController, "Run");
            UdonclArtifactImporter.SendUdonCustomEvent(saxpyController, "Run");

            const int timeoutFrames = 300;
            int frame = 0;
            float helloResult = 0.0f;
            float saxpyResult = 0.0f;
            while (
                frame < timeoutFrames
                && (
                    Math.Abs(helloResult - 10.0f) > 0.0001f
                    || Math.Abs(saxpyResult - 102.0f) > 0.0001f
                )
            )
            {
                ++frame;
                yield return null;
                helloResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(helloController, "result");
                saxpyResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "firstResult");
            }
            int helloStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(helloController, "status");
            int saxpyStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "status");
            yield return new ExitPlayMode();

            Assert.Less(
                frame,
                timeoutFrames,
                "Packaged controller readback timed out; status=" + helloStatus + "/" + saxpyStatus
            );
            Assert.AreEqual(0, helloStatus);
            Assert.AreEqual(0, saxpyStatus);
            Assert.AreEqual(10.0f, helloResult, 0.0001f);
            Assert.AreEqual(102.0f, saxpyResult, 0.0001f);
        }

        private static GameObject LoadPrefab(string root, string bindingClass)
        {
            return AssetDatabase.LoadAssetAtPath<GameObject>(
                root + "/Prefabs/" + bindingClass + ".prefab"
            );
        }

        private static void AssertRuntimeBinding(
            GameObject instance,
            UdonBehaviour controller,
            string bindingClass
        )
        {
            UdonBehaviour binding = UdonclArtifactImporter.FindUdonSharpBacking(
                instance,
                GeneratedNamespace + bindingClass
            );
            Assert.NotNull(binding, bindingClass + " backing is missing.");
            Assert.AreSame(
                binding,
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "binding"),
                bindingClass + " was not restored into the controller Udon heap."
            );
            Assert.NotNull(
                UdonclArtifactImporter.GetUdonProgramVariable(binding, "program"),
                bindingClass + " raw host program was not restored into the binding Udon heap."
            );
            Assert.NotNull(
                UdonclArtifactImporter.GetUdonProgramVariable(binding, "bufferCopyMaterial"),
                bindingClass + " buffer copy material was not restored into the binding Udon heap."
            );
        }

        private static void EnsureCompiledPrograms(
            string root,
            string bindingClass,
            string controllerClass
        )
        {
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                root + "/Bindings/" + bindingClass + ".cs",
                root + "/Bindings/" + bindingClass + ".asset"
            );
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                root + "/Examples/" + controllerClass + ".cs",
                root + "/Examples/" + controllerClass + ".asset"
            );
        }

        private static void AssertControllerBinding(
            GameObject prefab,
            string bindingClass,
            string controllerClass
        )
        {
            Assert.NotNull(prefab);
            UdonSharpBehaviour binding = null;
            UdonSharpBehaviour controller = null;
            foreach (UdonSharpBehaviour proxy in prefab.GetComponents<UdonSharpBehaviour>())
            {
                if (proxy.GetType().FullName == GeneratedNamespace + bindingClass)
                    binding = proxy;
                if (proxy.GetType().FullName == GeneratedNamespace + controllerClass)
                    controller = proxy;
            }
            Assert.NotNull(binding);
            Assert.NotNull(controller);
            SerializedObject serialized = new SerializedObject(controller);
            Assert.AreSame(binding, serialized.FindProperty("binding").objectReferenceValue);
            UdonBehaviour bindingBacking = UdonclArtifactImporter.FindUdonSharpBacking(
                prefab,
                GeneratedNamespace + bindingClass
            );
            UdonBehaviour controllerBacking = UdonclArtifactImporter.FindUdonSharpBacking(
                prefab,
                GeneratedNamespace + controllerClass
            );
            Assert.IsTrue(
                bindingBacking.publicVariables.TryGetVariableValue(
                    "program",
                    out UdonBehaviour hostProgram
                )
            );
            Assert.NotNull(hostProgram);
            Assert.IsTrue(
                bindingBacking.publicVariables.TryGetVariableValue(
                    "deviceBufferIds",
                    out int[] deviceBufferIds
                )
            );
            Assert.IsNotEmpty(deviceBufferIds);
            Assert.IsTrue(
                bindingBacking.publicVariables.TryGetVariableValue(
                    "deviceBuffers",
                    out RenderTexture[] deviceBuffers
                )
            );
            Assert.AreEqual(deviceBufferIds.Length, deviceBuffers.Length);
            Assert.IsTrue(
                bindingBacking.publicVariables.TryGetVariableValue(
                    "bufferCopyMaterial",
                    out Material bufferCopyMaterial
                )
            );
            Assert.NotNull(bufferCopyMaterial);
            Assert.IsTrue(
                controllerBacking.publicVariables.TryGetVariableValue(
                    "binding",
                    out UdonBehaviour bindingBehaviour
                )
            );
            Assert.AreSame(bindingBacking, bindingBehaviour);
        }
    }
}
