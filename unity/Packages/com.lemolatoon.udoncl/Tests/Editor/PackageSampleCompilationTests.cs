using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEditor;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    // Compiles every Package Sample to an Udon program against the bindings
    // of the imported example artifacts. Prepare Package Sample Validation
    // copies the samples into SampleRoot after SAXPY and MNIST are imported;
    // the TinyLlama Chat sample uses the public API of the binding of the
    // TinyLlama chat example (examples/tinyllama-chat).
    public sealed class PackageSampleCompilationTests
    {
        private const string SampleRoot = "Assets/UdonclSampleValidation";
        private const string GeneratedRoot = "Assets/udoncl-generated";
        private readonly List<string> createdPrograms = new List<string>();

        private static bool Enabled()
        {
            return Environment.GetEnvironmentVariable("UDONCL_PACKAGED_EXAMPLE_VALIDATION") == "1";
        }

        [OneTimeSetUp]
        public void CompileSampleBindings()
        {
            if (!Enabled())
                return;
            EnsureCompiled(GeneratedRoot + "/saxpy_demo/Bindings", "UdonclSaxpyDemoBinding");
            EnsureCompiled(
                GeneratedRoot + "/generic_onnx_mnist/Bindings",
                "UdonclGenericOnnxMnistBinding"
            );
            if (
                AssetDatabase.LoadAssetAtPath<MonoScript>(
                    SampleRoot + "/UdonclTinyllamaChatBinding.cs"
                ) != null
            )
                EnsureCompiled(SampleRoot, "UdonclTinyllamaChatBinding");
        }

        [TestCase("SaxpyCaller")]
        [TestCase("MnistClassifier")]
        [TestCase("TinyLlamaChat")]
        public void PackageSampleCompilesToUdonProgram(string className)
        {
            if (!Enabled())
                return;
            // Throws unless UdonSharp compiled the sample at its current version.
            EnsureCompiled(SampleRoot, className);
        }

        [OneTimeTearDown]
        public void RemoveSamplePrograms()
        {
            foreach (string path in createdPrograms)
            {
                string guid = AssetDatabase.AssetPathToGUID(path);
                AssetDatabase.DeleteAsset(path);
                if (!string.IsNullOrEmpty(guid))
                    AssetDatabase.DeleteAsset("Assets/SerializedUdonPrograms/" + guid + ".asset");
            }
            createdPrograms.Clear();
        }

        private void EnsureCompiled(string directory, string className)
        {
            string programPath = directory + "/" + className + ".asset";
            bool existed = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(programPath) != null;
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                directory + "/" + className + ".cs",
                programPath
            );
            // Generated bindings keep their program assets; only programs of
            // the copied samples are removed again.
            if (!existed && directory == SampleRoot)
                createdPrograms.Add(programPath);
            Assert.NotNull(
                AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(programPath),
                programPath
            );
        }
    }
}
