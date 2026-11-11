using System;
using System.Collections;
using System.IO;
using NUnit.Framework;
using UdonSharp;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using VRC.Udon;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class PublicTinyLlamaTokenizerLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/tinyllama_tokenizer";
        private const string BindingClass = "UdonclTinyLlamaTokenizerBinding";
        private const string ControllerClass = "UdonclTinyLlamaTokenizerTestController";
        private const string GeneratedNamespace = "Lemolatoon.Udoncl.Generated.";
        private const string ControllerSource =
            "Assets/UdonclSampleValidation/" + ControllerClass + ".cs";
        private const string ControllerProgram =
            GeneratedRoot + "/Tests/" + ControllerClass + ".asset";
        private string artifactDirectory;

        [OneTimeSetUp]
        public void ImportTokenizerArtifact()
        {
            string repository = ResolveRepository();
            artifactDirectory = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TOKENIZER_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifactDirectory))
                artifactDirectory = Path.Combine(
                    repository,
                    "build",
                    "udoncl",
                    "tests",
                    "cmake-package",
                    "consumer",
                    "udoncl",
                    "tinyllama-tokenizer-unity-library",
                    "artifact"
                );
            artifactDirectory = Path.GetFullPath(artifactDirectory);
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(artifactDirectory);
            Assert.AreEqual("tinyllama_tokenizer", artifact.ProgramName);
            if (!AssetDatabase.IsValidFolder(GeneratedRoot))
                Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));
            CollectionAssert.AreEqual(
                File.ReadAllBytes(Path.Combine(artifactDirectory, "binding.cs")),
                File.ReadAllBytes(GeneratedRoot + "/Bindings/" + BindingClass + ".cs")
            );
            Assert.NotNull(AssetDatabase.LoadAssetAtPath<MonoScript>(ControllerSource));
            if (!AssetDatabase.IsValidFolder(GeneratedRoot + "/Tests"))
                AssetDatabase.CreateFolder(GeneratedRoot, "Tests");
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                GeneratedRoot + "/Bindings/" + BindingClass + ".cs",
                GeneratedRoot + "/Bindings/" + BindingClass + ".asset"
            );
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                ControllerSource,
                ControllerProgram
            );
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
        }

        [UnityTest]
        public IEnumerator GeneratedUSharpBindingMatchesTinyLlamaTokenizer()
        {
            yield return new EnterPlayMode();

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/" + BindingClass + ".prefab"
            );
            Assert.NotNull(prefab);
            GameObject instance = UnityEngine.Object.Instantiate(prefab);
            UdonBehaviour controller = UdonclArtifactImporter.AddCompiledUdonSharpBehaviour(
                instance,
                ControllerProgram
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(instance);
            yield return null;

            UdonBehaviour binding = UdonclArtifactImporter.FindUdonSharpBacking(
                instance,
                GeneratedNamespace + BindingClass
            );
            Assert.NotNull(binding);
            Assert.NotNull(controller);
            int[] tokenIds = new int[16];
            UdonclArtifactImporter.SetUdonProgramVariable(controller, "binding", binding);
            UdonclArtifactImporter.SetUdonProgramVariable(controller, "text", "日本語 🍜");
            UdonclArtifactImporter.SetUdonProgramVariable(controller, "tokenIds", tokenIds);
            UdonclArtifactImporter.SendUdonCustomEvent(controller, "Run");
            yield return null;

            bool completed = (bool)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "completed");
            int status = (int)UdonclArtifactImporter.GetUdonProgramVariable(controller, "status");
            int count = (int)UdonclArtifactImporter.GetUdonProgramVariable(controller, "result");
            int[] actual = (int[])
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "tokenIds");
            yield return new ExitPlayMode();

            Assert.IsTrue(completed);
            Assert.AreEqual(0, status);
            Assert.AreEqual(10, count);
            CollectionAssert.AreEqual(
                new[] { 1, 29871, 30325, 30346, 30968, 29871, 243, 162, 144, 159 },
                new ArraySegment<int>(actual, 0, count)
            );
        }

        private static string ResolveRepository()
        {
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(PublicTinyLlamaTokenizerLifecycleTests).Assembly
                );
            Assert.NotNull(package);
            return Path.GetFullPath(Path.Combine(package.resolvedPath, "..", "..", ".."));
        }
    }
}
