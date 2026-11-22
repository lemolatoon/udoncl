using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor
{
    internal static class UdonclPackagedExampleImporter
    {
        private const string ArenaArchive = "onnx-arena-add.zip.bytes";
        private const string MnistArchive = "onnx-mnist.zip.bytes";
        private const string SaxpyArchive = "saxpy.zip.bytes";
        private const string TinyArchive = "onnx-tiny-llama.zip.bytes";
        private const string MnistControllerTemplate = "generic-onnx-mnist-test-controller.cs.txt";
        private const string TransformerControllerTemplate =
            "generic-onnx-transformer-test-controller.cs.txt";
        private const string TokenizerControllerTemplate =
            "tinyllama-tokenizer-test-controller.cs.txt";
        private const string ConversationControllerTemplate =
            "tinyllama-conversation-test-controller.cs.txt";
        private const string ConversationLifecycleTestTemplate =
            "tinyllama-conversation-lifecycle-test.cs.txt";
        private const string PrefillControllerTemplate = "tinyllama-prefill-test-controller.cs.txt";
        private const string PrefillLifecycleTestTemplate =
            "tinyllama-prefill-lifecycle-test.cs.txt";
        private const string StepControllerTemplate = "tinyllama-step-test-controller.cs.txt";
        private const string StepValidationTestTemplate = "tinyllama-step-lifecycle-test.cs.txt";
        private const string ChatControllerTemplate = "tinyllama-chat-test-controller.cs.txt";
        private const string ChatLifecycleTestTemplate = "tinyllama-chat-lifecycle-test.cs.txt";
        private const string MultiTurnChatControllerTemplate =
            "tinyllama-multi-turn-chat-test-controller.cs.txt";
        private const string MultiTurnChatLifecycleTestTemplate =
            "tinyllama-multi-turn-chat-lifecycle-test.cs.txt";
        private const string SampleValidationRoot = "Assets/UdonclSampleValidation";
        private const string TinyLlamaChatSample = "TinyLlama Chat";
        private const string TinyLlamaChatBindingStub = "tinyllama-chat-binding.cs.txt";

        [MenuItem("Tools/udoncl/Examples/Import C and OpenCL SAXPY")]
        private static void ImportSaxpy()
        {
            Import(SaxpyArchive);
        }

        [MenuItem("Tools/udoncl/Examples/Import ONNX MNIST")]
        private static void ImportMnist()
        {
            Import(MnistArchive);
        }

        [MenuItem("Tools/udoncl/Examples/Import ONNX Arena Add")]
        private static void ImportArenaAdd()
        {
            Import(ArenaArchive);
        }

        [MenuItem("Tools/udoncl/Examples/Import ONNX Tiny Llama")]
        private static void ImportTinyLlama()
        {
            Import(TinyArchive);
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Packaged Example Validation")]
        public static void PreparePackagedExampleValidation()
        {
            if (!Application.isBatchMode)
                throw new InvalidOperationException(
                    "Packaged example validation setup is batchmode-only."
                );
            PrepareValidationExample(SaxpyArchive, "saxpy_demo");
            // Prepare Public MNIST Validation already imported the CMake
            // artifact the MNIST archive is packaged from (scripts/precommit.py
            // checks that the archive is current), so only its controller is
            // added there; a fresh project imports the archive itself.
            const string mnistRoot = "Assets/udoncl-generated/generic_onnx_mnist";
            if (AssetDatabase.IsValidFolder(mnistRoot))
            {
                if (string.IsNullOrEmpty(InstallUsageExample(mnistRoot)))
                    throw new InvalidOperationException(
                        "Packaged example did not generate its U# controller: generic_onnx_mnist"
                    );
            }
            else
            {
                PrepareValidationExample(MnistArchive, "generic_onnx_mnist");
            }
            EnsureValidationSceneSaved();
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        }

        [MenuItem("Tools/udoncl/Internal/Ensure Validation Scene Saved")]
        public static void EnsureValidationSceneSaved()
        {
            UnityEngine.SceneManagement.Scene scene =
                UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene();
            if (!string.IsNullOrEmpty(scene.path) && File.Exists(scene.path))
                return;
            if (
                !UnityEditor.SceneManagement.EditorSceneManager.SaveScene(
                    scene,
                    "Assets/UdonclValidationScene.unity"
                )
            )
                throw new InvalidOperationException("Could not save the Unity validation scene.");
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public MNIST Validation")]
        public static void PreparePublicMnistValidation()
        {
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_MNIST_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
            {
                UnityEditor.PackageManager.PackageInfo package =
                    UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                        typeof(UdonclPackagedExampleImporter).Assembly
                    );
                if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                    return;
                artifact = Path.GetFullPath(
                    Path.Combine(
                        package.resolvedPath,
                        "..",
                        "..",
                        "..",
                        "build",
                        "udoncl",
                        "tests",
                        "cmake-package",
                        "onnx-mnist-consumer",
                        "udoncl",
                        "onnx-mnist",
                        "artifact"
                    )
                );
                if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                    return;
            }
            artifact = Path.GetFullPath(artifact);
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The MNIST validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/generic_onnx_mnist";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException("MNIST validation imported to an unexpected path.");
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public Transformer Validation")]
        public static void PreparePublicTransformerValidation()
        {
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TRANSFORMER_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
            {
                UnityEditor.PackageManager.PackageInfo package =
                    UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                        typeof(UdonclPackagedExampleImporter).Assembly
                    );
                if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                    return;
                artifact = Path.GetFullPath(
                    Path.Combine(
                        package.resolvedPath,
                        "..",
                        "..",
                        "..",
                        "build",
                        "udoncl",
                        "tests",
                        "cmake-package",
                        "onnx-transformer-consumer",
                        "udoncl",
                        "onnx-transformer-tiny",
                        "artifact"
                    )
                );
                if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                    return;
            }
            artifact = Path.GetFullPath(artifact);
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The Transformer validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/generic_onnx_transformer_tiny";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException(
                    "Transformer validation imported to an unexpected path."
                );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public Tokenizer Validation")]
        public static void PreparePublicTokenizerValidation()
        {
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                return;
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TOKENIZER_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
                artifact = Path.GetFullPath(
                    Path.Combine(
                        package.resolvedPath,
                        "..",
                        "..",
                        "..",
                        "build",
                        "udoncl",
                        "tests",
                        "cmake-package",
                        "consumer",
                        "udoncl",
                        "tinyllama-tokenizer-unity-library",
                        "artifact"
                    )
                );
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The TinyLlama tokenizer validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/tinyllama_tokenizer";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException(
                    "TinyLlama tokenizer validation imported to an unexpected path."
                );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public TinyLlama Conversation Validation")]
        public static void PreparePublicTinyLlamaConversationValidation()
        {
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TINYLLAMA_CONVERSATION_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
                throw new InvalidOperationException(
                    "UDONCL_PUBLIC_TINYLLAMA_CONVERSATION_UNITY_BUNDLE is required."
                );
            artifact = Path.GetFullPath(artifact);
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The TinyLlama conversation validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/tinyllama_real_conversation";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException(
                    "TinyLlama conversation validation imported to an unexpected path."
                );
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            const string validationRoot = "Assets/UdonclSampleValidation";
            string editorRoot = validationRoot + "/Editor";
            Directory.CreateDirectory(editorRoot);
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    ConversationControllerTemplate
                ),
                validationRoot + "/UdonclTinyLlamaConversationTestController.cs",
                true
            );
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    ConversationLifecycleTestTemplate
                ),
                editorRoot + "/PublicTinyLlamaConversationLifecycleTests.cs",
                true
            );
            File.WriteAllText(
                editorRoot + "/Udoncl.TinyLlama.Focused.Tests.asmdef",
                "{\n"
                    + "  \"name\": \"Udoncl.TinyLlama.Focused.Tests\",\n"
                    + "  \"references\": [\"Udoncl.Runtime\", \"Udoncl.Editor\", "
                    + "\"VRC.Udon\", \"VRC.Udon.Editor\", \"UdonSharp.Runtime\"],\n"
                    + "  \"includePlatforms\": [\"Editor\"],\n"
                    + "  \"overrideReferences\": true,\n"
                    + "  \"precompiledReferences\": [\"nunit.framework.dll\", "
                    + "\"VRC.Udon.Common.dll\"],\n"
                    + "  \"autoReferenced\": false,\n"
                    + "  \"optionalUnityReferences\": [\"TestAssemblies\"]\n"
                    + "}\n"
            );
            AssetDatabase.ImportAsset(
                validationRoot,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public TinyLlama Chat Validation")]
        public static void PreparePublicTinyLlamaChatValidation()
        {
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TINYLLAMA_CHAT_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
                throw new InvalidOperationException(
                    "UDONCL_PUBLIC_TINYLLAMA_CHAT_UNITY_BUNDLE is required."
                );
            artifact = Path.GetFullPath(artifact);
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The TinyLlama chat validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/tinyllama_real_chat";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException(
                    "TinyLlama chat validation imported to an unexpected path."
                );
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            const string validationRoot = "Assets/UdonclSampleValidation";
            string editorRoot = validationRoot + "/Editor";
            Directory.CreateDirectory(editorRoot);
            File.Copy(
                Path.Combine(package.resolvedPath, "Editor", "Examples", ChatControllerTemplate),
                validationRoot + "/UdonclTinyLlamaChatTestController.cs",
                true
            );
            File.Copy(
                Path.Combine(package.resolvedPath, "Editor", "Examples", ChatLifecycleTestTemplate),
                editorRoot + "/PublicTinyLlamaChatLifecycleTests.cs",
                true
            );
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    MultiTurnChatControllerTemplate
                ),
                validationRoot + "/UdonclTinyLlamaMultiTurnChatTestController.cs",
                true
            );
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    MultiTurnChatLifecycleTestTemplate
                ),
                editorRoot + "/PublicTinyLlamaMultiTurnChatLifecycleTests.cs",
                true
            );
            // The TinyLlama Chat Package Sample compiles against the binding
            // generated for this artifact.
            CopyPackageSamples(package.resolvedPath, TinyLlamaChatSample, validationRoot);
            File.WriteAllText(
                editorRoot + "/Udoncl.TinyLlama.Chat.Focused.Tests.asmdef",
                "{\n"
                    + "  \"name\": \"Udoncl.TinyLlama.Chat.Focused.Tests\",\n"
                    + "  \"references\": [\"Udoncl.Runtime\", \"Udoncl.Editor\", "
                    + "\"VRC.Udon\", \"VRC.Udon.Editor\", \"UdonSharp.Runtime\"],\n"
                    + "  \"includePlatforms\": [\"Editor\"],\n"
                    + "  \"overrideReferences\": true,\n"
                    + "  \"precompiledReferences\": [\"nunit.framework.dll\", "
                    + "\"VRC.Udon.Common.dll\"],\n"
                    + "  \"autoReferenced\": false,\n"
                    + "  \"optionalUnityReferences\": [\"TestAssemblies\"]\n"
                    + "}\n"
            );
            AssetDatabase.ImportAsset(
                validationRoot,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public TinyLlama Prefill Validation")]
        public static void PreparePublicTinyLlamaPrefillValidation()
        {
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TINYLLAMA_PREFILL_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
                throw new InvalidOperationException(
                    "UDONCL_PUBLIC_TINYLLAMA_PREFILL_UNITY_BUNDLE is required."
                );
            artifact = Path.GetFullPath(artifact);
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The TinyLlama prefill validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/tinyllama_real_prefill";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException(
                    "TinyLlama prefill validation imported to an unexpected path."
                );
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            const string validationRoot = "Assets/UdonclSampleValidation";
            string editorRoot = validationRoot + "/Editor";
            Directory.CreateDirectory(editorRoot);
            File.Copy(
                Path.Combine(package.resolvedPath, "Editor", "Examples", PrefillControllerTemplate),
                validationRoot + "/UdonclTinyLlamaPrefillTestController.cs",
                true
            );
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    PrefillLifecycleTestTemplate
                ),
                editorRoot + "/PublicTinyLlamaPrefillLifecycleTests.cs",
                true
            );
            File.WriteAllText(
                editorRoot + "/Udoncl.TinyLlama.Prefill.Focused.Tests.asmdef",
                "{\n"
                    + "  \"name\": \"Udoncl.TinyLlama.Prefill.Focused.Tests\",\n"
                    + "  \"references\": [\"Udoncl.Runtime\", \"Udoncl.Editor\", "
                    + "\"VRC.Udon\", \"VRC.Udon.Editor\", \"UdonSharp.Runtime\"],\n"
                    + "  \"includePlatforms\": [\"Editor\"],\n"
                    + "  \"overrideReferences\": true,\n"
                    + "  \"precompiledReferences\": [\"nunit.framework.dll\", "
                    + "\"VRC.Udon.Common.dll\"],\n"
                    + "  \"autoReferenced\": false,\n"
                    + "  \"optionalUnityReferences\": [\"TestAssemblies\"]\n"
                    + "}\n"
            );
            AssetDatabase.ImportAsset(
                validationRoot,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public TinyLlama M1 Step Validation")]
        public static void PreparePublicTinyLlamaStepValidation()
        {
            string artifact = Environment.GetEnvironmentVariable(
                "UDONCL_PUBLIC_TINYLLAMA_STEP_UNITY_BUNDLE"
            );
            if (string.IsNullOrWhiteSpace(artifact))
                throw new InvalidOperationException(
                    "UDONCL_PUBLIC_TINYLLAMA_STEP_UNITY_BUNDLE is required."
                );
            artifact = Path.GetFullPath(artifact);
            if (!File.Exists(Path.Combine(artifact, "manifest.json")))
                throw new FileNotFoundException(
                    "The TinyLlama M=1 step validation artifact is missing.",
                    artifact
                );
            const string root = "Assets/udoncl-generated/tinyllama_real_m1_step";
            AssetDatabase.DeleteAsset(root);
            if (UdonclArtifactImporter.Import(artifact) != root)
                throw new InvalidDataException(
                    "TinyLlama M=1 step validation imported to an unexpected path."
                );
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            const string validationRoot = "Assets/UdonclSampleValidation";
            string editorRoot = validationRoot + "/Editor";
            Directory.CreateDirectory(editorRoot);
            File.Copy(
                Path.Combine(package.resolvedPath, "Editor", "Examples", StepControllerTemplate),
                validationRoot + "/UdonclTinyLlamaStepTestController.cs",
                true
            );
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    StepValidationTestTemplate
                ),
                editorRoot + "/PublicTinyLlamaStepValidationTests.cs",
                true
            );
            File.WriteAllText(
                editorRoot + "/Udoncl.TinyLlama.Step.Focused.Tests.asmdef",
                "{\n"
                    + "  \"name\": \"Udoncl.TinyLlama.Step.Focused.Tests\",\n"
                    + "  \"references\": [\"Udoncl.Runtime\", \"Udoncl.Editor\", "
                    + "\"VRC.Udon\", \"VRC.Udon.Editor\", \"UdonSharp.Runtime\"],\n"
                    + "  \"includePlatforms\": [\"Editor\"],\n"
                    + "  \"overrideReferences\": true,\n"
                    + "  \"precompiledReferences\": [\"nunit.framework.dll\", "
                    + "\"VRC.Udon.Common.dll\"],\n"
                    + "  \"autoReferenced\": false,\n"
                    + "  \"optionalUnityReferences\": [\"TestAssemblies\"]\n"
                    + "}\n"
            );
            AssetDatabase.ImportAsset(
                validationRoot,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Package Sample Validation")]
        public static void PreparePackageSampleValidation()
        {
            if (!Application.isBatchMode)
                throw new InvalidOperationException(
                    "Package sample validation setup is batchmode-only."
                );
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            // Samples use the bindings of imported artifacts, so they are
            // copied after Prepare Public MNIST Validation and Prepare
            // Packaged Example Validation imported MNIST and SAXPY.
            CopyPackageSamples(package.resolvedPath, null, SampleValidationRoot);
            // The TinyLlama Chat Artifact needs the real model. Without it,
            // the sample compiles against the public API of its binding.
            if (!AssetDatabase.IsValidFolder("Assets/udoncl-generated/tinyllama_real_chat"))
                File.Copy(
                    Path.Combine(
                        package.resolvedPath,
                        "Tests",
                        "Editor",
                        "Fixtures",
                        TinyLlamaChatBindingStub
                    ),
                    SampleValidationRoot + "/UdonclTinyLlamaChatBinding.cs",
                    true
                );
            AssetDatabase.ImportAsset(
                SampleValidationRoot,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        // Copies the U# scripts of one Package Sample, or of all of them when
        // sampleName is null, flat into destination.
        private static void CopyPackageSamples(
            string packageRoot,
            string sampleName,
            string destination
        )
        {
            string samples = Path.Combine(packageRoot, "Samples~");
            if (sampleName != null)
                samples = Path.Combine(samples, sampleName);
            string[] sources = Directory.GetFiles(samples, "*.cs", SearchOption.AllDirectories);
            if (sources.Length == 0)
                throw new InvalidDataException(
                    "The UdonCL package sample has no scripts: " + samples
                );
            Array.Sort(sources, StringComparer.Ordinal);
            Directory.CreateDirectory(destination);
            foreach (string source in sources)
            {
                string target = destination + "/" + Path.GetFileName(source);
                if (File.Exists(target))
                    throw new InvalidDataException(
                        "Duplicate UdonCL package sample script name: " + Path.GetFileName(source)
                    );
                File.Copy(source, target);
            }
        }

        [MenuItem("Tools/udoncl/Internal/Prepare Public MNIST Controller Validation")]
        public static void PreparePublicMnistControllerValidation()
        {
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            string source = Path.Combine(
                package.resolvedPath,
                "Editor",
                "Examples",
                MnistControllerTemplate
            );
            string destinationDirectory = "Assets/UdonclSampleValidation";
            string destination = destinationDirectory + "/UdonclGenericOnnxMnistTestController.cs";

            Directory.CreateDirectory(destinationDirectory);
            File.Copy(source, destination, true);
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    TransformerControllerTemplate
                ),
                destinationDirectory + "/UdonclGenericOnnxTransformerTinyTestController.cs",
                true
            );
            File.Copy(
                Path.Combine(
                    package.resolvedPath,
                    "Editor",
                    "Examples",
                    TokenizerControllerTemplate
                ),
                destinationDirectory + "/UdonclTinyLlamaTokenizerTestController.cs",
                true
            );
            AssetDatabase.ImportAsset(
                destinationDirectory,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            UnityEditor.Compilation.CompilationPipeline.RequestScriptCompilation();
        }

        internal static string Extract(string archiveName)
        {
            if (
                archiveName != ArenaArchive
                && archiveName != MnistArchive
                && archiveName != SaxpyArchive
                && archiveName != TinyArchive
            )
                throw new ArgumentException(
                    "Unknown packaged example archive.",
                    nameof(archiveName)
                );
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonclPackagedExampleImporter).Assembly
                );
            if (package == null || string.IsNullOrEmpty(package.resolvedPath))
                throw new InvalidOperationException(
                    "Could not resolve the installed UdonCL package."
                );
            string archivePath = Path.Combine(
                package.resolvedPath,
                "Editor",
                "Examples",
                archiveName
            );
            if (!File.Exists(archivePath))
                throw new FileNotFoundException(
                    "The packaged example artifact is missing.",
                    archivePath
                );

            string destination = Path.Combine(
                Path.GetTempPath(),
                "udoncl-example-" + Guid.NewGuid().ToString("N")
            );
            Directory.CreateDirectory(destination);
            string destinationRoot = Path.GetFullPath(destination) + Path.DirectorySeparatorChar;
            StringComparison pathComparison =
                Path.DirectorySeparatorChar == '\\'
                    ? StringComparison.OrdinalIgnoreCase
                    : StringComparison.Ordinal;
            try
            {
                using (FileStream stream = File.OpenRead(archivePath))
                using (ZipArchive archive = new ZipArchive(stream, ZipArchiveMode.Read))
                {
                    long extractedBytes = 0;
                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        if (string.IsNullOrEmpty(entry.Name))
                            continue;
                        if (((entry.ExternalAttributes >> 16) & 0xF000) == 0xA000)
                            throw new InvalidDataException(
                                "Packaged example contains a symbolic link: " + entry.FullName
                            );
                        extractedBytes += entry.Length;
                        if (extractedBytes > 16 * 1024 * 1024)
                            throw new InvalidDataException(
                                "Packaged example is unexpectedly large."
                            );
                        string relative = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
                        string output = Path.GetFullPath(Path.Combine(destination, relative));
                        if (!output.StartsWith(destinationRoot, pathComparison))
                            throw new InvalidDataException(
                                "Packaged example contains an unsafe archive path: "
                                    + entry.FullName
                            );
                        Directory.CreateDirectory(Path.GetDirectoryName(output));
                        using (Stream input = entry.Open())
                        using (FileStream file = File.Create(output))
                            input.CopyTo(file);
                    }
                }
                if (!File.Exists(Path.Combine(destination, "manifest.json")))
                    throw new InvalidDataException("Packaged example has no root manifest.json.");
                return destination;
            }
            catch
            {
                Directory.Delete(destination, true);
                throw;
            }
        }

        private static void Import(string archiveName)
        {
            string directory = null;
            try
            {
                directory = Extract(archiveName);
                string destination = UdonclImportMenu.ImportAndOfferScene(directory);
                if (!string.IsNullOrEmpty(destination))
                    InstallUsageExample(destination);
            }
            catch (Exception exception)
            {
                UnityEngine.Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "udoncl Example Import Failed",
                    exception.Message,
                    "OK"
                );
            }
            finally
            {
                if (!string.IsNullOrEmpty(directory) && Directory.Exists(directory))
                    Directory.Delete(directory, true);
            }
        }

        internal static string InstallUsageExample(string destination)
        {
            string manifestPath = destination + "/manifest.json";
            UdonclManifest manifest = UnityEngine.JsonUtility.FromJson<UdonclManifest>(
                File.ReadAllText(manifestPath)
            );
            PackagedExampleUsage usage = PackagedExampleUsageGenerator.ForProgram(
                manifest?.program?.name
            );
            if (usage == null)
                return null;

            string examplesRoot = destination + "/Examples";
            if (!AssetDatabase.IsValidFolder(examplesRoot))
                AssetDatabase.CreateFolder(destination, "Examples");
            string sourcePath = examplesRoot + "/" + usage.ClassName + ".cs";
            File.WriteAllText(sourcePath, usage.Source, new UTF8Encoding(false));
            AssetDatabase.ImportAsset(
                sourcePath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            GeneratedBindingFinalizer.Schedule();
            return sourcePath;
        }

        private static void PrepareValidationExample(string archiveName, string programName)
        {
            string expected = "Assets/udoncl-generated/" + programName;
            if (AssetDatabase.IsValidFolder(expected))
                throw new InvalidOperationException(
                    "Packaged example validation requires a fresh path: " + expected
                );

            string directory = Extract(archiveName);
            try
            {
                string destination = UdonclArtifactImporter.Import(directory);
                if (destination != expected)
                    throw new InvalidOperationException(
                        "Packaged example imported to an unexpected path: " + destination
                    );
                if (string.IsNullOrEmpty(InstallUsageExample(destination)))
                    throw new InvalidOperationException(
                        "Packaged example did not generate its U# controller: " + programName
                    );
            }
            finally
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, true);
            }
        }
    }
}
