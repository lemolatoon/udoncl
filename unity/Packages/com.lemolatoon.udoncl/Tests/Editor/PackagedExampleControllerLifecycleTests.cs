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
        private const string SaxpyRoot = "Assets/udoncl-generated/saxpy_demo";
        private const string MnistRoot = "Assets/udoncl-generated/generic_onnx_mnist";
        private const string UdonBlasRoot = "Assets/udoncl-generated/udonblas_sgemm_m1n10k144";
        private const string GeneratedNamespace = "Lemolatoon.Udoncl.Generated.";

        [OneTimeSetUp]
        public void FinalizePreparedExamples()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_PACKAGED_EXAMPLE_VALIDATION") != "1")
                return;
            EnsureCompiledPrograms(SaxpyRoot, "UdonclSaxpyDemoBinding", "UdonclSaxpyDemoExample");
            EnsureCompiledPrograms(
                MnistRoot,
                "UdonclGenericOnnxMnistBinding",
                "UdonclGenericOnnxMnistExample"
            );
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
            Assert.NotNull(LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding"));
            Assert.NotNull(LoadPrefab(MnistRoot, "UdonclGenericOnnxMnistBinding"));
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
                LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding"),
                "UdonclSaxpyDemoBinding",
                "UdonclSaxpyDemoExample"
            );
            AssertControllerBinding(
                LoadPrefab(MnistRoot, "UdonclGenericOnnxMnistBinding"),
                "UdonclGenericOnnxMnistBinding",
                "UdonclGenericOnnxMnistExample"
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
            GameObject saxpy = UnityEngine.Object.Instantiate(
                LoadPrefab(SaxpyRoot, "UdonclSaxpyDemoBinding")
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(saxpy);
            yield return null;

            UdonBehaviour saxpyController = UdonclArtifactImporter.FindUdonSharpBacking(
                saxpy,
                GeneratedNamespace + "UdonclSaxpyDemoExample"
            );
            Assert.NotNull(saxpyController);
            AssertRuntimeBinding(saxpy, saxpyController, "UdonclSaxpyDemoBinding");

            // Run is an asynchronous export of a Host that includes the
            // OpenCL runtime, so it spans frames like the UdonBLAS example.
            // The controller initializes the library, runs SAXPY over the
            // first 16 words, then again with new inputs over all 32 words,
            // reading the result after each run.
            const int timeoutFrames = 900;
            UdonclArtifactImporter.SendUdonCustomEvent(saxpyController, "Run");
            int saxpyFrame = 0;
            float firstResult = 0.0f;
            float secondResult = 0.0f;
            while (saxpyFrame < timeoutFrames && Math.Abs(secondResult - 129.0f) > 0.0001f)
            {
                ++saxpyFrame;
                yield return null;
                firstResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "firstResult");
                secondResult = (float)
                    UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "secondResult");
            }
            int saxpyStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "status");
            // Disabling the controller disposes the library.
            saxpy.SetActive(false);
            yield return null;
            int disposeStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(saxpyController, "disposeStatus");
            yield return new ExitPlayMode();

            Assert.Less(
                saxpyFrame,
                timeoutFrames,
                "SAXPY readback timed out; status="
                    + saxpyStatus
                    + "; first="
                    + firstResult
                    + "; second="
                    + secondResult
            );
            Assert.AreEqual(0, saxpyStatus);
            // 2 * x[0] + y[0] = 2 * 1 + 100.
            Assert.AreEqual(102.0f, firstResult, 0.0001f);
            // 0.5 * x[31] + y[31] = 0.5 * 320 - 31.
            Assert.AreEqual(129.0f, secondResult, 0.0001f);
            Assert.AreEqual(0, disposeStatus);
        }

        [UnityTest]
        public IEnumerator ImportedMnistControllerClassifiesThroughPublicApi()
        {
            if (Environment.GetEnvironmentVariable("UDONCL_PACKAGED_EXAMPLE_VALIDATION") != "1")
                yield break;

            yield return new EnterPlayMode();
            GameObject instance = UnityEngine.Object.Instantiate(
                LoadPrefab(MnistRoot, "UdonclGenericOnnxMnistBinding")
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(instance);
            yield return null;

            UdonBehaviour controller = UdonclArtifactImporter.FindUdonSharpBacking(
                instance,
                GeneratedNamespace + "UdonclGenericOnnxMnistExample"
            );
            Assert.NotNull(controller);
            AssertRuntimeBinding(instance, controller, "UdonclGenericOnnxMnistBinding");

            // The packaged controller classifies its hand-drawn seven: it
            // initializes the session, uploads the image, runs the model and
            // reads the logits back through the generated binding.
            const int timeoutFrames = 1200;
            UdonclArtifactImporter.SendUdonCustomEvent(controller, "Run");
            int frame = 0;
            int predicted = -1;
            int status = 0;
            while (frame < timeoutFrames && predicted < 0 && status == 0)
            {
                ++frame;
                yield return null;
                predicted = (int)
                    UdonclArtifactImporter.GetUdonProgramVariable(controller, "predictedDigit");
                status = (int)UdonclArtifactImporter.GetUdonProgramVariable(controller, "status");
            }
            int libraryStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "libraryStatus");
            // Disabling the controller disposes the session.
            instance.SetActive(false);
            yield return null;
            int disposeStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "disposeStatus");
            int disposeResult = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "disposeResult");
            yield return new ExitPlayMode();

            Assert.Less(
                frame,
                timeoutFrames,
                "MNIST classification timed out; status="
                    + status
                    + "; libraryStatus="
                    + libraryStatus
            );
            Assert.AreEqual(0, status);
            Assert.AreEqual(0, libraryStatus);
            Assert.AreEqual(7, predicted);
            Assert.AreEqual(0, disposeStatus);
            Assert.AreEqual(0, disposeResult);
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
            // The named transfer methods index deviceBuffers by constant;
            // the binding has no buffer id table to fill.
            Assert.IsFalse(
                bindingBacking.publicVariables.TryGetVariableType("deviceBufferIds", out Type _)
            );
            Assert.IsTrue(
                bindingBacking.publicVariables.TryGetVariableValue(
                    "deviceBufferLogicalBytes",
                    out int[] deviceBufferLogicalBytes
                )
            );
            Assert.IsNotEmpty(deviceBufferLogicalBytes);
            Assert.IsTrue(
                bindingBacking.publicVariables.TryGetVariableValue(
                    "deviceBuffers",
                    out RenderTexture[] deviceBuffers
                )
            );
            Assert.AreEqual(deviceBufferLogicalBytes.Length, deviceBuffers.Length);
            Assert.IsTrue(
                hostProgram.publicVariables.TryGetVariableValue(
                    "__udoncl_buffer_pool",
                    out RenderTexture[] programBufferPool
                )
            );
            CollectionAssert.AreEqual(deviceBuffers, programBufferPool);
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
