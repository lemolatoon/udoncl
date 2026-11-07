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
        private const string PoolingRoot = "Assets/udoncl-generated/average_pooling";
        private const string UdonBlasRoot = "Assets/udoncl-generated/udonblas_sgemm_m1n10k144";
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
            EnsureCompiledPrograms(
                PoolingRoot,
                "UdonclAveragePoolingBinding",
                "UdonclAveragePoolingExample"
            );
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
            Assert.NotNull(LoadPrefab(HelloRoot, "UdonclHelloDispatchBinding"));
            Assert.NotNull(LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding"));
            Assert.NotNull(LoadPrefab(PoolingRoot, "UdonclAveragePoolingBinding"));
        }

        [OneTimeSetUp]
        public void FinalizePreparedUdonBlasExample()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_UDONBLAS_HOST_VALIDATION") != "1")
                return;
            EnsureCompiledPrograms(
                UdonBlasRoot,
                "UdonclUdonblasSgemmM1n10k144Binding",
                "UdonclUdonBlasSgemmExample"
            );
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
            Assert.NotNull(LoadPrefab(UdonBlasRoot, "UdonclUdonblasSgemmM1n10k144Binding"));
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
            AssertControllerBinding(
                LoadPrefab(PoolingRoot, "UdonclAveragePoolingBinding"),
                "UdonclAveragePoolingBinding",
                "UdonclAveragePoolingExample"
            );
        }

        [Test]
        public void ImportedUdonBlasPrefabContainsBoundTypedController()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_UDONBLAS_HOST_VALIDATION") != "1")
                return;
            AssertControllerBinding(
                LoadPrefab(UdonBlasRoot, "UdonclUdonblasSgemmM1n10k144Binding"),
                "UdonclUdonblasSgemmM1n10k144Binding",
                "UdonclUdonBlasSgemmExample"
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
            GameObject pooling = UnityEngine.Object.Instantiate(
                LoadPrefab(PoolingRoot, "UdonclAveragePoolingBinding")
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(hello);
            UdonclArtifactImporter.InitializeUdonBehaviours(saxpy);
            UdonclArtifactImporter.InitializeUdonBehaviours(pooling);
            yield return null;

            UdonBehaviour helloController = UdonclArtifactImporter.FindUdonSharpBacking(
                hello,
                GeneratedNamespace + "UdonclHelloDispatchExample"
            );
            UdonBehaviour saxpyController = UdonclArtifactImporter.FindUdonSharpBacking(
                saxpy,
                GeneratedNamespace + "UdonclSaxpyDemoExample"
            );
            UdonBehaviour poolingController = UdonclArtifactImporter.FindUdonSharpBacking(
                pooling,
                GeneratedNamespace + "UdonclAveragePoolingExample"
            );
            Assert.NotNull(helloController);
            Assert.NotNull(saxpyController);
            Assert.NotNull(poolingController);
            AssertRuntimeBinding(hello, helloController, "UdonclHelloDispatchBinding");
            AssertRuntimeBinding(saxpy, saxpyController, "UdonclSaxpyDemoBinding");
            AssertRuntimeBinding(pooling, poolingController, "UdonclAveragePoolingBinding");

            const int timeoutFrames = 300;
            UdonclArtifactImporter.SendUdonCustomEvent(helloController, "Run");
            int helloFrame = 0;
            float helloResult = 0.0f;
            while (helloFrame < timeoutFrames && Math.Abs(helloResult - 10.0f) > 0.0001f)
            {
                ++helloFrame;
                yield return null;
                helloResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(helloController, "result");
            }

            UdonclArtifactImporter.SendUdonCustomEvent(saxpyController, "Run");
            int saxpyFrame = 0;
            float saxpyResult = 0.0f;
            while (saxpyFrame < timeoutFrames && Math.Abs(saxpyResult - 102.0f) > 0.0001f)
            {
                ++saxpyFrame;
                yield return null;
                saxpyResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "firstResult");
            }

            UdonclArtifactImporter.SendUdonCustomEvent(poolingController, "Run");
            int poolingFrame = 0;
            float poolingResult = 0.0f;
            while (poolingFrame < timeoutFrames && Math.Abs(poolingResult - 87.5f) > 0.0001f)
            {
                ++poolingFrame;
                yield return null;
                poolingResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(poolingController, "firstResult");
            }
            int helloStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(helloController, "status");
            int saxpyStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "status");
            int poolingStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(poolingController, "status");
            yield return new ExitPlayMode();

            Assert.Less(
                helloFrame,
                timeoutFrames,
                "Hello readback timed out; status=" + helloStatus + "; result=" + helloResult
            );
            Assert.Less(
                saxpyFrame,
                timeoutFrames,
                "SAXPY readback timed out; status=" + saxpyStatus + "; result=" + saxpyResult
            );
            Assert.Less(
                poolingFrame,
                timeoutFrames,
                "Average pooling readback timed out; status="
                    + poolingStatus
                    + "; result="
                    + poolingResult
            );
            Assert.AreEqual(0, helloStatus);
            Assert.AreEqual(0, saxpyStatus);
            Assert.AreEqual(0, poolingStatus);
            Assert.AreEqual(10.0f, helloResult, 0.0001f);
            Assert.AreEqual(102.0f, saxpyResult, 0.0001f);
            Assert.AreEqual(87.5f, poolingResult, 0.0001f);
        }

        [UnityTest]
        public IEnumerator ImportedUdonBlasControllerExecutesNumericallyThroughPublicApi()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_UDONBLAS_HOST_VALIDATION") != "1")
                yield break;

            yield return new EnterPlayMode();
            GameObject instance = UnityEngine.Object.Instantiate(
                LoadPrefab(UdonBlasRoot, "UdonclUdonblasSgemmM1n10k144Binding")
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(instance);
            yield return null;

            UdonBehaviour controller = UdonclArtifactImporter.FindUdonSharpBacking(
                instance,
                GeneratedNamespace + "UdonclUdonBlasSgemmExample"
            );
            Assert.NotNull(controller);
            AssertRuntimeBinding(instance, controller, "UdonclUdonblasSgemmM1n10k144Binding");
            UdonclArtifactImporter.SendUdonCustomEvent(controller, "Run");

            const int timeoutFrames = 900;
            int frame = 0;
            bool completed = false;
            while (frame < timeoutFrames && !completed)
            {
                ++frame;
                yield return null;
                completed = (bool)
                    UdonclArtifactImporter.GetUdonProgramVariable(controller, "completed");
            }
            int status = (int)UdonclArtifactImporter.GetUdonProgramVariable(controller, "status");
            int libraryStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "libraryStatus");
            int libraryPhase = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "libraryPhase");
            float maximumError = (float)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "maximumError");
            float[] results = (float[])
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "results");
            yield return new ExitPlayMode();

            Assert.IsTrue(
                completed,
                "UdonBLAS async execution timed out after "
                    + frame
                    + " frames; status="
                    + status
                    + "; libraryStatus="
                    + libraryStatus
                    + "; libraryPhase="
                    + libraryPhase
            );
            Assert.AreEqual(0, status);
            Assert.AreEqual(0, libraryStatus);
            Assert.That(maximumError, Is.LessThan(0.0005f));
            Assert.AreEqual(10, results.Length);
            for (int column = 0; column < results.Length; ++column)
                Assert.AreEqual(1224.0f * (column + 1), results[column], 0.0005f);
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
