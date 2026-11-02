using System;
using System.Collections.Generic;
using System.IO;
using UdonSharp;
using UdonSharp.Compiler;
using UdonSharpEditor;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using VRC.Udon;
using VRC.Udon.Common;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;
using VRC.Udon.ProgramSources;
using SyncType = VRC.SDKBase.Networking.SyncType;

namespace Lemolatoon.Udoncl.Editor
{
    public static class UdonclArtifactImporter
    {
        private const string GeneratedRoot = "Assets/udoncl-generated";

        public static string Import(string artifactDirectory)
        {
            ValidatedArtifact artifacts = ArtifactManifestValidator.Validate(artifactDirectory);
            string programRoot = GeneratedRoot + "/" + artifacts.ProgramName;

            try
            {
                using (AssetImportTransaction transaction = new AssetImportTransaction())
                {
                    EnsureFolder("Assets", "udoncl-generated");
                    EnsureFolder(GeneratedRoot, artifacts.ProgramName);

                    transaction.CopyFile(
                        artifacts.ManifestAssetPath,
                        programRoot + "/manifest.json"
                    );
                    GenerateTypedWrapper(transaction, artifacts, programRoot);
                    string sourceDirectory = programRoot + "/Sources";
                    EnsureFolders(sourceDirectory);
                    string assemblySourcePath =
                        sourceDirectory
                        + "/"
                        + Path.GetFileName(artifacts.AssemblyAssetPath)
                        + ".txt";
                    transaction.CopyFile(artifacts.AssemblyAssetPath, assemblySourcePath);
                    UdonAssemblyProgramAsset programAsset = CreateOrUpdateProgramAsset(
                        transaction,
                        programRoot,
                        File.ReadAllText(artifacts.AssemblyAssetPath)
                    );

                    string displayShaderPath = programRoot + "/Shaders/display.shader";
                    EnsureFolders(programRoot + "/Shaders");
                    transaction.CopyFile(artifacts.DisplayShaderAssetPath, displayShaderPath);

                    if (!string.IsNullOrEmpty(artifacts.BufferCopyShaderAssetPath))
                    {
                        string bufferCopyShaderPath = programRoot + "/Shaders/buffer-copy.shader";
                        transaction.CopyFile(
                            artifacts.BufferCopyShaderAssetPath,
                            bufferCopyShaderPath
                        );
                        Shader bufferCopyShader = AssetDatabase.LoadAssetAtPath<Shader>(
                            bufferCopyShaderPath
                        );
                        if (bufferCopyShader == null)
                            throw new InvalidDataException(
                                "Unity could not import buffer copy shader: " + bufferCopyShaderPath
                            );
                        CreateOrUpdateNamedMaterial(
                            transaction,
                            programRoot,
                            "buffer-copy.mat",
                            bufferCopyShader
                        );
                    }

                    Dictionary<ValidatedKernel, string> shaderAssetPaths =
                        new Dictionary<ValidatedKernel, string>();
                    foreach (ValidatedKernel kernel in artifacts.Kernels)
                    {
                        string shaderAssetPath =
                            programRoot + "/Shaders/" + kernel.RelativeShaderPath;
                        string shaderDirectory = Path.GetDirectoryName(shaderAssetPath)
                            .Replace('\\', '/');
                        EnsureFolders(shaderDirectory);
                        transaction.CopyFile(kernel.AbsoluteShaderPath, shaderAssetPath);
                        shaderAssetPaths.Add(kernel, shaderAssetPath);
                    }

                    Dictionary<int, Material> kernelMaterials = new Dictionary<int, Material>();
                    Dictionary<string, Shader> kernelShaders = new Dictionary<string, Shader>(
                        StringComparer.Ordinal
                    );
                    foreach (KeyValuePair<ValidatedKernel, string> shaderEntry in shaderAssetPaths)
                    {
                        Shader shader = AssetDatabase.LoadAssetAtPath<Shader>(shaderEntry.Value);
                        if (shader == null)
                            throw new InvalidDataException(
                                "Unity could not import generated shader: " + shaderEntry.Value
                            );
                        kernelMaterials.Add(
                            shaderEntry.Key.Id,
                            CreateOrUpdateMaterial(
                                transaction,
                                programRoot,
                                shaderEntry.Key,
                                shader
                            )
                        );
                        kernelShaders.Add(shaderEntry.Key.RelativeShaderPath, shader);
                        string shaderFileName = Path.GetFileName(
                            shaderEntry.Key.RelativeShaderPath
                        );
                        if (!kernelShaders.ContainsKey(shaderFileName))
                            kernelShaders.Add(shaderFileName, shader);
                    }

                    Shader displayShader = AssetDatabase.LoadAssetAtPath<Shader>(displayShaderPath);
                    if (displayShader == null)
                        throw new InvalidDataException(
                            "Unity could not import display shader: " + displayShaderPath
                        );
                    Material displayMaterial = CreateOrUpdateDisplayMaterial(
                        transaction,
                        programRoot,
                        displayShader
                    );
                    foreach (ValidatedDeviceBuffer buffer in artifacts.DeviceBuffers)
                        CreateOrUpdateRgbaRenderTexture(
                            transaction,
                            programRoot,
                            DeviceBufferFileName(buffer.Id, buffer.Name),
                            buffer.Width,
                            buffer.Height
                        );
                    ImportOnnxRuntimeResources(
                        transaction,
                        artifacts,
                        programRoot,
                        programAsset,
                        kernelMaterials,
                        kernelShaders
                    );
                    if (artifacts.ArtifactStage == "unity-udoncl-library")
                    {
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        GeneratedBindingFinalizer.Schedule();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udon-onnx-model")
                    {
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-mnist-prototype")
                    {
                        ImportMnistDemo(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-tiny-llama-prototype")
                    {
                        ImportTinyLlamaDemo(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonblas-sgemm")
                    {
                        ImportUdonBlasSgemm(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonblas-sgemv")
                    {
                        ImportUdonBlasSgemv(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonaten-int4-mm")
                    {
                        ImportUdonAtenWeightInt4packMm(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonaten-tile")
                    {
                        ImportUdonAtenTile(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonblas-saxpy")
                    {
                        ImportUdonBlasSaxpy(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonblas-sdot")
                    {
                        ImportUdonBlasSdot(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udondnn-convolution")
                    {
                        ImportUdonDnnConvolution(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udondnn-activation")
                    {
                        ImportUdonDnnActivation(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udondnn-pooling")
                    {
                        ImportUdonDnnPooling(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udondnn-softmax")
                    {
                        ImportUdonDnnSoftmax(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udondnn-batchnorm")
                    {
                        ImportUdonDnnBatchNorm(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udondnn-op-tensor")
                    {
                        ImportUdonDnnOpTensor(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage == "unity-udonaten-rms-norm")
                    {
                        ImportUdonAtenRmsNorm(
                            transaction,
                            artifacts,
                            programRoot,
                            programAsset,
                            kernelMaterials,
                            displayMaterial
                        );
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        return programRoot;
                    }
                    if (artifacts.ArtifactStage != "unity-lj-prototype")
                        throw new InvalidDataException(
                            "Unsupported Unity artifact stage: " + artifacts.ArtifactStage
                        );
                    Texture2D initialState = CreateOrUpdateParticleTexture(
                        transaction,
                        programRoot
                    );
                    RenderTexture stateA = CreateOrUpdateRenderTexture(
                        transaction,
                        programRoot,
                        "lj_state_a.asset"
                    );
                    RenderTexture stateB = CreateOrUpdateRenderTexture(
                        transaction,
                        programRoot,
                        "lj_state_b.asset"
                    );
                    if (
                        !kernelMaterials.TryGetValue(10, out Material driftMaterial)
                        || !kernelMaterials.TryGetValue(11, out Material kickMaterial)
                    )
                        throw new InvalidDataException(
                            "LJ demo requires kernel ids 10 (drift) and 11 (kick)."
                        );

                    CreateOrUpdateDemoPrefab(
                        transaction,
                        programRoot,
                        programAsset,
                        initialState,
                        stateA,
                        stateB,
                        driftMaterial,
                        kickMaterial,
                        displayMaterial
                    );

                    AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                    transaction.Commit();
                }
            }
            catch
            {
                throw;
            }

            return programRoot;
        }

        private static void GenerateTypedWrapper(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot
        )
        {
            string manifestJson = File.ReadAllText(artifacts.ManifestAssetPath);
            UdonclManifest manifest = JsonUtility.FromJson<UdonclManifest>(manifestJson);
            if (manifest == null || string.IsNullOrEmpty(manifest.physical_export_abi))
                return;

            string className = VRCWrapperGenerator.BindingClassName(artifacts.ProgramName);
            string bindingRoot = programRoot + "/Bindings";
            EnsureFolders(bindingRoot);
            transaction.WriteText(
                bindingRoot + "/" + className + ".cs",
                VRCWrapperGenerator.GenerateFromJson(
                    manifestJson,
                    "Lemolatoon.Udoncl.Generated",
                    className
                )
            );
            CreateOrUpdateUdonSharpProgramAsset(
                transaction,
                bindingRoot + "/" + className + ".cs",
                bindingRoot + "/" + className + ".asset"
            );
        }

        private static void CreateOrUpdateUdonSharpProgramAsset(
            AssetImportTransaction transaction,
            string scriptPath,
            string programAssetPath
        )
        {
            MonoScript script = AssetDatabase.LoadAssetAtPath<MonoScript>(scriptPath);
            if (script == null)
                throw new InvalidDataException(
                    "Unity could not import generated UdonSharp source: " + scriptPath
                );

            transaction.Track(programAssetPath);
            UdonSharpProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonSharpProgramAsset>(programAssetPath);
            if (programAsset == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(programAssetPath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + programAssetPath
                    );
                programAsset = ScriptableObject.CreateInstance<UdonSharpProgramAsset>();
                programAsset.sourceCsScript = script;
                AssetDatabase.CreateAsset(programAsset, programAssetPath);
            }
            else if (programAsset.sourceCsScript != script)
            {
                programAsset.sourceCsScript = script;
                EditorUtility.SetDirty(programAsset);
            }
        }

        private static void EnsureOnnxRuntimeProgramAsset(AssetImportTransaction transaction)
        {
            const string scriptPath =
                "Packages/com.lemolatoon.udoncl/Runtime/UdonOnnxRuntimeBehaviour.cs";
            string runtimeRoot = GeneratedRoot + "/Runtime";
            string programAssetPath = runtimeRoot + "/UdonOnnxRuntimeBehaviour.asset";
            EnsureFolders(runtimeRoot);
            CreateOrUpdateUdonSharpProgramAsset(transaction, scriptPath, programAssetPath);
            AssetDatabase.ImportAsset(programAssetPath, ImportAssetOptions.ForceSynchronousImport);

            UdonSharpProgramAsset generatedProgramAsset =
                AssetDatabase.LoadAssetAtPath<UdonSharpProgramAsset>(programAssetPath);
            if (generatedProgramAsset == null)
                throw new InvalidDataException(
                    "Unity could not create the Udon ONNX runtime U# program asset."
                );
            UdonSharpProgramAsset programAsset = UdonSharpEditorUtility.GetUdonSharpProgramAsset(
                typeof(Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour)
            );
            if (
                programAsset != null
                && programAsset.SerializedProgramAsset != null
                && programAsset.ScriptVersion == UdonSharpProgramVersion.CurrentVersion
                && programAsset.CompiledVersion == UdonSharpProgramVersion.CurrentVersion
            )
                return;

            generatedProgramAsset.ScriptVersion = UdonSharpProgramVersion.CurrentVersion;
            EditorUtility.SetDirty(generatedProgramAsset);

            UdonSharpCompilerV1.CompileSync();
            programAsset = UdonSharpEditorUtility.GetUdonSharpProgramAsset(
                typeof(Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour)
            );
            if (
                programAsset == null
                || programAsset.SerializedProgramAsset == null
                || programAsset.ScriptVersion != UdonSharpProgramVersion.CurrentVersion
                || programAsset.CompiledVersion != UdonSharpProgramVersion.CurrentVersion
            )
                throw new InvalidDataException(
                    "Unity could not compile the Udon ONNX runtime U# program asset."
                );
        }

        private static void ImportMnistDemo(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (
                !kernelMaterials.TryGetValue(20, out Material convMaterial)
                || !kernelMaterials.TryGetValue(21, out Material poolMaterial)
                || !kernelMaterials.TryGetValue(22, out Material denseMaterial)
            )
                throw new InvalidDataException("MNIST demo requires kernel ids 20, 21 and 22.");
            ValidatedResource packShaderResource = FindResource(artifacts, "pack_input", "shader");
            ValidatedResource brushShaderResource = FindResource(artifacts, "brush", "shader");
            ValidatedResource weightsResource = FindResource(artifacts, "packed_weights", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "reference_blank_logits",
                "bytes"
            );
            string packShaderPath = programRoot + "/Shaders/pack.shader";
            transaction.CopyFile(packShaderResource.AbsolutePath, packShaderPath);
            Shader packShader = AssetDatabase.LoadAssetAtPath<Shader>(packShaderPath);
            if (packShader == null)
                throw new InvalidDataException(
                    "Unity could not import MNIST pack shader: " + packShaderPath
                );
            Material packMaterial = CreateOrUpdateNamedMaterial(
                transaction,
                programRoot,
                "mnist_pack.mat",
                packShader
            );
            string brushShaderPath = programRoot + "/Shaders/brush.shader";
            transaction.CopyFile(brushShaderResource.AbsolutePath, brushShaderPath);
            Shader brushShader = AssetDatabase.LoadAssetAtPath<Shader>(brushShaderPath);
            if (brushShader == null)
                throw new InvalidDataException(
                    "Unity could not import MNIST brush shader: " + brushShaderPath
                );
            Material brushMaterial = CreateOrUpdateNamedMaterial(
                transaction,
                programRoot,
                "mnist_brush.mat",
                brushShader
            );
            Texture2D weights = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "mnist_weights.asset",
                10800,
                1,
                File.ReadAllBytes(weightsResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "mnist_reference_blank_logits.asset",
                10,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            packMaterial.SetTexture("_Weights", weights);
            EditorUtility.SetDirty(packMaterial);

            RenderTexture drawing = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "mnist_drawing.asset",
                28,
                28
            );
            RenderTexture packed = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "mnist_packed.asset",
                10800,
                1
            );
            RenderTexture drawingScratch = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "mnist_drawing_scratch.asset",
                28,
                28
            );
            RenderTexture logits = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "mnist_logits.asset",
                10800,
                1
            );
            displayMaterial.SetTexture("_MainTex", drawing);
            displayMaterial.SetTexture("_Logits", logits);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateMnistPrefab(
                transaction,
                programRoot,
                programAsset,
                drawing,
                drawingScratch,
                packed,
                logits,
                packMaterial,
                brushMaterial,
                convMaterial,
                poolMaterial,
                denseMaterial,
                displayMaterial
            );
        }

        private static void ImportTinyLlamaDemo(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            Material[] passes = new Material[7];
            for (int index = 0; index < passes.Length; ++index)
            {
                if (!kernelMaterials.TryGetValue(30 + index, out Material material))
                    throw new InvalidDataException(
                        "Tiny LLaMA demo requires kernel ids 30 through 36."
                    );
                passes[index] = material;
            }
            ValidatedResource packShaderResource = FindResource(artifacts, "prompt_pack", "shader");
            ValidatedResource argmaxShaderResource = FindResource(
                artifacts,
                "argmax_readback",
                "shader"
            );
            string packShaderPath = programRoot + "/Shaders/prompt-pack.shader";
            transaction.CopyFile(packShaderResource.AbsolutePath, packShaderPath);
            Shader packShader = AssetDatabase.LoadAssetAtPath<Shader>(packShaderPath);
            if (packShader == null)
                throw new InvalidDataException(
                    "Unity could not import Tiny LLaMA prompt pack shader: " + packShaderPath
                );
            Material packMaterial = CreateOrUpdateNamedMaterial(
                transaction,
                programRoot,
                "tiny_llama_prompt_pack.mat",
                packShader
            );
            string argmaxShaderPath = programRoot + "/Shaders/argmax-readback.shader";
            transaction.CopyFile(argmaxShaderResource.AbsolutePath, argmaxShaderPath);
            Shader argmaxShader = AssetDatabase.LoadAssetAtPath<Shader>(argmaxShaderPath);
            if (argmaxShader == null)
                throw new InvalidDataException(
                    "Unity could not import Tiny LLaMA argmax shader: " + argmaxShaderPath
                );
            Material argmaxMaterial = CreateOrUpdateNamedMaterial(
                transaction,
                programRoot,
                "tiny_llama_argmax.mat",
                argmaxShader
            );
            ValidatedResource packedResource = FindResource(artifacts, "packed_state", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "reference_logits",
                "bytes"
            );
            Texture2D initialState = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "tiny_llama_initial.asset",
                2048,
                1,
                File.ReadAllBytes(packedResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "tiny_llama_reference_logits.asset",
                16,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture stateA = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "tiny_llama_state_a.asset",
                2048,
                1
            );
            RenderTexture stateB = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "tiny_llama_state_b.asset",
                2048,
                1
            );
            RenderTexture argmaxState = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "tiny_llama_argmax.asset",
                1,
                1
            );
            displayMaterial.SetTexture("_Logits", stateB);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateTinyLlamaPrefab(
                transaction,
                programRoot,
                programAsset,
                initialState,
                stateA,
                stateB,
                argmaxState,
                packMaterial,
                passes,
                argmaxMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonBlasSgemm(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(100, out Material sgemmMaterial))
                throw new InvalidDataException("UdonBLAS SGEMM requires kernel id 100.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_sgemm_input.asset",
                32,
                32,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_sgemm_reference.asset",
                20,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonblas_sgemm_output.asset",
                32,
                32
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonBlasSgemmPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                sgemmMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonBlasSgemv(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(101, out Material sgemvMaterial))
                throw new InvalidDataException("UdonBLAS SGEMV requires kernel id 101.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_sgemv_input.asset",
                1024,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_sgemv_reference.asset",
                5,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonblas_sgemv_output.asset",
                1024,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonBlasSgemvPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                sgemvMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonDnnConvolution(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(200, out Material convolutionMaterial))
                throw new InvalidDataException("UdonDNN convolution requires kernel id 200.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_convolution_input.asset",
                2048,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_convolution_reference.asset",
                48,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udondnn_convolution_output.asset",
                2048,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonDnnConvolutionPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                convolutionMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonBlasSaxpy(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(102, out Material saxpyMaterial))
                throw new InvalidDataException("UdonBLAS SAXPY requires kernel id 102.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_saxpy_input.asset",
                1024,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_saxpy_reference.asset",
                16,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonblas_saxpy_output.asset",
                1024,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonBlasSaxpyPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                saxpyMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonBlasSdot(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(103, out Material productMaterial))
                throw new InvalidDataException("UdonBLAS SDOT requires product kernel id 103.");
            if (!kernelMaterials.TryGetValue(104, out Material reduceMaterial))
                throw new InvalidDataException("UdonBLAS SDOT requires reduction kernel id 104.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_sdot_input.asset",
                1024,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonblas_sdot_reference.asset",
                1,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonblas_sdot_output.asset",
                1024,
                1
            );
            RenderTexture scratch = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonblas_sdot_scratch.asset",
                1024,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonBlasSdotPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                scratch,
                productMaterial,
                reduceMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonAtenWeightInt4packMm(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(208, out Material int4Material))
                throw new InvalidDataException("UdonATen INT4 MM requires kernel id 208.");
            ValidatedResource weightAtlasResource = FindResource(
                artifacts,
                "fixture_weight_atlas",
                "bytes"
            );
            ValidatedResource activationResource = FindResource(
                artifacts,
                "fixture_activation",
                "bytes"
            );
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            ValidatedResource composeShaderResource = FindResource(
                artifacts,
                "page_compose",
                "shader"
            );
            ValidatedResource scatterShaderResource = FindResource(
                artifacts,
                "result_scatter",
                "shader"
            );
            string composeShaderPath = programRoot + "/Shaders/page-compose.shader";
            transaction.CopyFile(composeShaderResource.AbsolutePath, composeShaderPath);
            Shader composeShader = AssetDatabase.LoadAssetAtPath<Shader>(composeShaderPath);
            if (composeShader == null)
                throw new InvalidDataException(
                    "Unity could not import the UdonATen INT4 page compose shader: "
                        + composeShaderPath
                );
            Material composeMaterial = CreateOrUpdateNamedMaterial(
                transaction,
                programRoot,
                "page-compose.mat",
                composeShader
            );
            string scatterShaderPath = programRoot + "/Shaders/result-scatter.shader";
            transaction.CopyFile(scatterShaderResource.AbsolutePath, scatterShaderPath);
            Shader scatterShader = AssetDatabase.LoadAssetAtPath<Shader>(scatterShaderPath);
            if (scatterShader == null)
                throw new InvalidDataException(
                    "Unity could not import the UdonATen INT4 result scatter shader: "
                        + scatterShaderPath
                );
            Material scatterMaterial = CreateOrUpdateNamedMaterial(
                transaction,
                programRoot,
                "result-scatter.mat",
                scatterShader
            );
            Texture2D weightAtlas = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_weight_atlas.asset",
                1280,
                2,
                File.ReadAllBytes(weightAtlasResource.AbsolutePath)
            );
            Texture2D activation = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_activation.asset",
                256,
                1,
                File.ReadAllBytes(activationResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_reference.asset",
                8,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture assembly = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_assembly.asset",
                8192,
                1
            );
            RenderTexture partial = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_partial.asset",
                8192,
                1
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_output.asset",
                8192,
                1
            );
            RenderTexture result = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonaten_weight_int4pack_mm_result.asset",
                8,
                1
            );
            displayMaterial.SetTexture("_Result", result);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonAtenWeightInt4packMmPrefab(
                transaction,
                programRoot,
                programAsset,
                weightAtlas,
                activation,
                assembly,
                partial,
                output,
                result,
                composeMaterial,
                int4Material,
                scatterMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonAtenTile(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(209, out Material tileMaterial))
                throw new InvalidDataException("UdonATen Tile requires kernel id 209.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_tile_input.asset",
                8192,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_tile_reference.asset",
                72,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonaten_tile_output.asset",
                8192,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonAtenTilePrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                tileMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonDnnActivation(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(201, out Material activationMaterial))
                throw new InvalidDataException("UdonDNN activation requires kernel id 201.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_activation_input.asset",
                2048,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_activation_reference.asset",
                512,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udondnn_activation_output.asset",
                2048,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonDnnActivationPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                activationMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonDnnPooling(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(202, out Material poolingMaterial))
                throw new InvalidDataException("UdonDNN pooling requires kernel id 202.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_pooling_input.asset",
                2048,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_pooling_reference.asset",
                48,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udondnn_pooling_output.asset",
                2048,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonDnnPoolingPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                poolingMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonDnnSoftmax(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(203, out Material softmaxMaterial))
                throw new InvalidDataException("UdonDNN softmax requires kernel id 203.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_softmax_input.asset",
                2048,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_softmax_reference.asset",
                72,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udondnn_softmax_output.asset",
                2048,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonDnnSoftmaxPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                softmaxMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonDnnBatchNorm(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(204, out Material batchnormMaterial))
                throw new InvalidDataException(
                    "UdonDNN batch normalization requires kernel id 204."
                );
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_batchnorm_input.asset",
                4096,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_batchnorm_reference.asset",
                72,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udondnn_batchnorm_output.asset",
                4096,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonDnnBatchNormPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                batchnormMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonDnnOpTensor(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(205, out Material opTensorMaterial))
                throw new InvalidDataException("UdonDNN OpTensor requires kernel id 205.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_op_tensor_input.asset",
                2048,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udondnn_op_tensor_reference.asset",
                72,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udondnn_op_tensor_output.asset",
                2048,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonDnnOpTensorPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                opTensorMaterial,
                displayMaterial
            );
        }

        private static void ImportUdonAtenRmsNorm(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            Material displayMaterial
        )
        {
            if (!kernelMaterials.TryGetValue(206, out Material rmsNormMaterial))
                throw new InvalidDataException("UdonATen RMSNorm requires kernel id 206.");
            ValidatedResource inputResource = FindResource(artifacts, "fixture_input", "bytes");
            ValidatedResource referenceResource = FindResource(
                artifacts,
                "fixture_reference",
                "bytes"
            );
            Texture2D input = CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_rms_norm_input.asset",
                2048,
                1,
                File.ReadAllBytes(inputResource.AbsolutePath)
            );
            CreateOrUpdateByteTexture(
                transaction,
                programRoot,
                "udonaten_rms_norm_reference.asset",
                64,
                1,
                File.ReadAllBytes(referenceResource.AbsolutePath)
            );
            RenderTexture output = CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                "udonaten_rms_norm_output.asset",
                2048,
                1
            );
            displayMaterial.SetTexture("_Result", output);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateUdonAtenRmsNormPrefab(
                transaction,
                programRoot,
                programAsset,
                input,
                output,
                rmsNormMaterial,
                displayMaterial
            );
        }

        private static ValidatedResource FindResource(
            ValidatedArtifact artifact,
            string name,
            string kind
        )
        {
            foreach (ValidatedResource resource in artifact.Resources)
                if (resource.Name == name && resource.Kind == kind)
                    return resource;
            throw new InvalidDataException(
                "Artifact is missing required " + kind + " resource '" + name + "'."
            );
        }

        private static ValidatedResource FindOptionalResource(
            ValidatedArtifact artifact,
            string name,
            string kind
        )
        {
            foreach (ValidatedResource resource in artifact.Resources)
                if (resource.Name == name && resource.Kind == kind)
                    return resource;
            return null;
        }

        private static void ImportOnnxRuntimeResources(
            AssetImportTransaction transaction,
            ValidatedArtifact artifact,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            IDictionary<string, Shader> kernelShaders
        )
        {
            ValidatedResource model = FindOptionalResource(artifact, "onnx_model", "bytes");
            ValidatedResource plan = FindOptionalResource(artifact, "onnx_execution_plan", "bytes");
            if (model == null && plan == null)
                return;
            if (model == null || plan == null)
                throw new InvalidDataException(
                    "ONNX artifacts must provide both onnx_model and onnx_execution_plan."
                );
            string modelDirectory = programRoot + "/Models";
            EnsureFolders(modelDirectory);
            transaction.CopyFile(model.AbsolutePath, modelDirectory + "/model.onnx.bytes");
            transaction.CopyFile(plan.AbsolutePath, modelDirectory + "/execution-plan.json.bytes");
            TextAsset modelText = AssetDatabase.LoadAssetAtPath<TextAsset>(
                modelDirectory + "/model.onnx.bytes"
            );
            TextAsset planText = AssetDatabase.LoadAssetAtPath<TextAsset>(
                modelDirectory + "/execution-plan.json.bytes"
            );
            if (modelText == null || planText == null)
                throw new InvalidDataException(
                    "Unity could not import the ONNX model and execution plan resources."
                );
            UdonOnnxExecutionPlan executionPlan = JsonUtility.FromJson<UdonOnnxExecutionPlan>(
                planText.text
            );
            int textureArenaTotalWords = 0;
            int[] textureSlotCapacities = Array.Empty<int>();
            int[] textureSlotWidths = Array.Empty<int>();
            int[] textureSlotHeights = Array.Empty<int>();
            RenderTexture[] textureArenaSlots = Array.Empty<RenderTexture>();
            // JsonUtility may materialize a default nested object even when an older
            // execution plan omits the optional field, so presence must be checked
            // against the source JSON for backward compatibility.
            if (
                executionPlan.texture_arena != null
                && planText.text.IndexOf("\"texture_arena\"", StringComparison.Ordinal) >= 0
            )
            {
                UdonOnnxTextureArenaInfo arena = executionPlan.texture_arena;
                if (
                    arena.abi != "linear-rgba32-words"
                    || arena.slots == null
                    || arena.slot_count != arena.slots.Length
                    || arena.slot_count <= 0
                )
                    throw new InvalidDataException("ONNX texture arena header is invalid.");
                textureSlotCapacities = new int[arena.slot_count];
                textureSlotWidths = new int[arena.slot_count];
                textureSlotHeights = new int[arena.slot_count];
                textureArenaSlots = new RenderTexture[arena.slot_count];
                long totalWords = 0;
                for (int index = 0; index < arena.slot_count; ++index)
                {
                    UdonOnnxTextureSlotInfo slot = arena.slots[index];
                    int expectedWidth = Math.Min(8192, slot.capacity_words);
                    int expectedHeight = (slot.capacity_words + 8191) / 8192;
                    if (
                        slot.index != index
                        || slot.capacity_words <= 0
                        || slot.width != expectedWidth
                        || slot.height != expectedHeight
                    )
                        throw new InvalidDataException("ONNX texture arena slot is invalid.");
                    textureSlotCapacities[index] = slot.capacity_words;
                    textureSlotWidths[index] = slot.width;
                    textureSlotHeights[index] = slot.height;
                    textureArenaSlots[index] = CreateOrUpdateRgbaRenderTexture(
                        transaction,
                        programRoot,
                        "onnx_arena_slot_" + index + ".asset",
                        slot.width,
                        slot.height
                    );
                    totalWords += slot.capacity_words;
                }
                if (
                    totalWords > int.MaxValue
                    || arena.total_words != totalWords
                    || arena.total_bytes != totalWords * 4
                )
                    throw new InvalidDataException("ONNX texture arena capacity is invalid.");
                textureArenaTotalWords = arena.total_words;
            }
            List<Material> passMaterials = new List<Material>();
            Dictionary<string, Texture2D> staticInputs = new Dictionary<string, Texture2D>(
                StringComparer.Ordinal
            );
            Texture2D[] staticInputTextures = new Texture2D[artifact.OnnxStaticInputs.Count];
            for (int index = 0; index < artifact.OnnxStaticInputs.Count; ++index)
            {
                ValidatedOnnxStaticInput input = artifact.OnnxStaticInputs[index];
                Texture2D texture = CreateOrUpdateByteTexture(
                    transaction,
                    programRoot,
                    "onnx_static_input_" + index + ".asset",
                    input.Width,
                    input.Height,
                    File.ReadAllBytes(input.AbsolutePath)
                );
                staticInputs.Add(input.Name, texture);
                staticInputTextures[index] = texture;
            }
            for (int index = 0; index < executionPlan.passes.Length; ++index)
            {
                UdonOnnxPassInfo pass = executionPlan.passes[index];
                if (pass.has_host_material_index && pass.host_material_index < 0)
                    continue;
                int expectedMaterialIndex = pass.has_host_material_index
                    ? pass.host_material_index
                    : passMaterials.Count;
                if (expectedMaterialIndex != passMaterials.Count)
                    throw new InvalidDataException(
                        "ONNX Host material indexes must be contiguous and topological."
                    );
                Material passMaterial;
                if (!string.IsNullOrWhiteSpace(pass.shader))
                {
                    if (!kernelShaders.TryGetValue(pass.shader, out Shader shader))
                        throw new InvalidDataException(
                            "ONNX execution pass references missing shader " + pass.shader + "."
                        );
                    Material material = CreateOrUpdateNamedMaterial(
                        transaction,
                        programRoot,
                        "onnx_pass_" + pass.id + "_" + pass.kernel + ".mat",
                        shader
                    );
                    foreach (
                        UdonOnnxIntMaterialBinding binding in pass.material_int_bindings
                            ?? Array.Empty<UdonOnnxIntMaterialBinding>()
                    )
                        material.SetInteger(binding.property, binding.value);
                    foreach (
                        UdonOnnxFloatMaterialBinding binding in pass.material_float_bindings
                            ?? Array.Empty<UdonOnnxFloatMaterialBinding>()
                    )
                        material.SetFloat(binding.property, binding.value);
                    foreach (
                        UdonOnnxPassStaticInput binding in pass.static_inputs
                            ?? Array.Empty<UdonOnnxPassStaticInput>()
                    )
                    {
                        if (!staticInputs.TryGetValue(binding.initializer, out Texture2D texture))
                            throw new InvalidDataException(
                                "ONNX pass references missing static input "
                                    + binding.initializer
                                    + "."
                            );
                        material.SetTexture(binding.property, texture);
                    }
                    EditorUtility.SetDirty(material);
                    passMaterial = material;
                }
                else if (!kernelMaterials.TryGetValue(pass.id, out passMaterial))
                    throw new InvalidDataException(
                        "ONNX execution pass references missing kernel id " + pass.id + "."
                    );
                passMaterials.Add(passMaterial);
            }
            GameObject runtimePrefab = CreateOrUpdateOnnxModelPrefab(
                transaction,
                artifact,
                programRoot,
                programAsset,
                kernelMaterials,
                textureArenaSlots,
                passMaterials.ToArray()
            );
            string assetPath = modelDirectory + "/UdonOnnxModel.asset";
            transaction.Track(assetPath);
            Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(assetPath);
            if (modelAsset == null)
            {
                modelAsset =
                    ScriptableObject.CreateInstance<Lemolatoon.Udoncl.UdonOnnxModelAsset>();
                modelAsset.name = "UdonOnnxModel";
                AssetDatabase.CreateAsset(modelAsset, assetPath);
            }
            SerializedObject serialized = new SerializedObject(modelAsset);
            serialized.FindProperty("model").objectReferenceValue = modelText;
            serialized.FindProperty("executionPlan").objectReferenceValue = planText;
            serialized.FindProperty("executionProvider").stringValue =
                executionPlan.execution_provider;
            serialized.FindProperty("profile").stringValue = executionPlan.profile ?? "";
            serialized.FindProperty("hostProgram").objectReferenceValue = programAsset;
            SerializedProperty materials = serialized.FindProperty("passMaterials");
            materials.arraySize = passMaterials.Count;
            for (int index = 0; index < passMaterials.Count; ++index)
                materials.GetArrayElementAtIndex(index).objectReferenceValue = passMaterials[index];
            serialized.FindProperty("textureArenaTotalWords").intValue = textureArenaTotalWords;
            SetIntArray(serialized.FindProperty("textureSlotCapacities"), textureSlotCapacities);
            SetIntArray(serialized.FindProperty("textureSlotWidths"), textureSlotWidths);
            SetIntArray(serialized.FindProperty("textureSlotHeights"), textureSlotHeights);
            SerializedProperty arenaSlots = serialized.FindProperty("textureArenaSlots");
            arenaSlots.arraySize = textureArenaSlots.Length;
            for (int index = 0; index < textureArenaSlots.Length; ++index)
                arenaSlots.GetArrayElementAtIndex(index).objectReferenceValue = textureArenaSlots[
                    index
                ];
            SerializedProperty staticTextures = serialized.FindProperty("staticInputTextures");
            staticTextures.arraySize = staticInputTextures.Length;
            for (int index = 0; index < staticInputTextures.Length; ++index)
                staticTextures.GetArrayElementAtIndex(index).objectReferenceValue =
                    staticInputTextures[index];
            SetOnnxIoProperties(serialized, "input", artifact.OnnxInputs);
            SetOnnxIoProperties(serialized, "output", artifact.OnnxOutputs);
            serialized.FindProperty("runtimePrefab").objectReferenceValue = runtimePrefab;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(modelAsset);
            // Later stage-specific imports refresh the AssetDatabase before the
            // transaction commits. Persist these references first so that refresh
            // cannot reload the new asset with unassigned fields.
            AssetDatabase.SaveAssetIfDirty(modelAsset);
        }

        private static GameObject CreateOrUpdateOnnxModelPrefab(
            AssetImportTransaction transaction,
            ValidatedArtifact artifact,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            IDictionary<int, Material> kernelMaterials,
            RenderTexture[] arenaSlots,
            Material[] passMaterials
        )
        {
            if (
                artifact.ArtifactStage != "unity-udon-onnx-model"
                && artifact.OnnxInt4Calls.Count == 0
            )
                return null;
            Material int4Template = null;
            Shader composeShader = null;
            Shader scatterShader = null;
            if (artifact.OnnxInt4Calls.Count != 0)
            {
                if (!kernelMaterials.TryGetValue(208, out int4Template))
                    throw new InvalidDataException(
                        "ONNX INT4 runners require UdonATen kernel 208."
                    );
                ValidatedResource composeResource = FindResource(
                    artifact,
                    "page_compose",
                    "shader"
                );
                ValidatedResource scatterResource = FindResource(
                    artifact,
                    "result_scatter",
                    "shader"
                );
                string composeShaderPath = programRoot + "/Shaders/onnx-int4-compose.shader";
                string scatterShaderPath = programRoot + "/Shaders/onnx-int4-scatter.shader";
                transaction.CopyFile(composeResource.AbsolutePath, composeShaderPath);
                transaction.CopyFile(scatterResource.AbsolutePath, scatterShaderPath);
                composeShader = AssetDatabase.LoadAssetAtPath<Shader>(composeShaderPath);
                scatterShader = AssetDatabase.LoadAssetAtPath<Shader>(scatterShaderPath);
                if (composeShader == null || scatterShader == null)
                    throw new InvalidDataException(
                        "Unity could not import ONNX INT4 runner shaders."
                    );
            }

            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonOnnxModel.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("Udon ONNX model");
            try
            {
                UdonBehaviour model = root.AddComponent<UdonBehaviour>();
                BindProgramAsset(model, programAsset);
                model.interactText = "Run ONNX model";
                model.proximity = 3.0f;
                model.SyncMethod = SyncType.None;
                for (int index = 0; index < arenaSlots.Length; ++index)
                    SetPublicVariable(model, "arenaSlot" + index, arenaSlots[index]);
                SetPublicVariable(model, "passMaterials", passMaterials);

                EnsureOnnxRuntimeProgramAsset(transaction);
                UdonSharpBehaviour runtime = root.AddUdonSharpComponent(
                    typeof(Lemolatoon.Udoncl.UdonOnnxRuntimeBehaviour)
                );
                SerializedObject serializedRuntime = new SerializedObject(runtime);
                serializedRuntime.FindProperty("program").objectReferenceValue = model;
                SetOnnxIoProperties(serializedRuntime, "input", artifact.OnnxInputs);
                SetOnnxIoProperties(serializedRuntime, "output", artifact.OnnxOutputs);
                SerializedProperty serializedSlots = serializedRuntime.FindProperty("arenaSlots");
                serializedSlots.arraySize = arenaSlots.Length;
                for (int index = 0; index < arenaSlots.Length; ++index)
                    serializedSlots.GetArrayElementAtIndex(index).objectReferenceValue = arenaSlots[
                        index
                    ];
                string copyMaterialPath = programRoot + "/Materials/buffer-copy.mat";
                Material copyMaterial = AssetDatabase.LoadAssetAtPath<Material>(copyMaterialPath);
                if (copyMaterial == null)
                    throw new InvalidDataException(
                        "Generated ONNX transfer material is missing: " + copyMaterialPath
                    );
                serializedRuntime.FindProperty("bufferCopyMaterial").objectReferenceValue =
                    copyMaterial;
                serializedRuntime.ApplyModifiedPropertiesWithoutUndo();
                UdonSharpEditorUtility.CopyProxyToUdon(runtime, ProxySerializationPolicy.All);

                IUdonEventReceiver[] runners = new IUdonEventReceiver[artifact.OnnxInt4Calls.Count];
                for (int runnerIndex = 0; runnerIndex < artifact.OnnxInt4Calls.Count; ++runnerIndex)
                {
                    ValidatedOnnxInt4Call call = artifact.OnnxInt4Calls[runnerIndex];
                    string runnerName = "call-" + call.CallIndex.ToString("D4");
                    string runnerRoot = programRoot + "/Int4Runners/" + runnerName;
                    UdonAssemblyProgramAsset runnerProgram = CreateOrUpdateProgramAsset(
                        transaction,
                        runnerRoot,
                        File.ReadAllText(call.AssemblyAbsolutePath)
                    );
                    Material composeMaterial = CreateOrUpdateNamedMaterial(
                        transaction,
                        runnerRoot,
                        "compose.mat",
                        composeShader
                    );
                    Material int4Material = CreateOrUpdateNamedMaterial(
                        transaction,
                        runnerRoot,
                        "int4.mat",
                        int4Template.shader
                    );
                    Material scatterMaterial = CreateOrUpdateNamedMaterial(
                        transaction,
                        runnerRoot,
                        "scatter.mat",
                        scatterShader
                    );
                    RenderTexture composeState = CreateOrUpdateRgbaRenderTexture(
                        transaction,
                        runnerRoot,
                        "compose.asset",
                        8192,
                        1
                    );
                    RenderTexture accumulatorState = CreateOrUpdateRgbaRenderTexture(
                        transaction,
                        runnerRoot,
                        "accumulator.asset",
                        8192,
                        1
                    );
                    RenderTexture resultA = CreateOrUpdateRgbaRenderTexture(
                        transaction,
                        runnerRoot,
                        "result-a.asset",
                        call.ResultWidth,
                        call.ResultHeight
                    );
                    RenderTexture resultB = CreateOrUpdateRgbaRenderTexture(
                        transaction,
                        runnerRoot,
                        "result-b.asset",
                        call.ResultWidth,
                        call.ResultHeight
                    );

                    GameObject child = new GameObject(runnerName);
                    child.transform.SetParent(root.transform, false);
                    UdonBehaviour runner = child.AddComponent<UdonBehaviour>();
                    BindProgramAsset(runner, runnerProgram);
                    runner.SyncMethod = SyncType.None;
                    SetPublicVariable(runner, "composeState", composeState);
                    SetPublicVariable(runner, "accumulatorState", accumulatorState);
                    SetPublicVariable(runner, "resultA", resultA);
                    SetPublicVariable(runner, "resultB", resultB);
                    SetPublicVariable(runner, "composeMaterial", composeMaterial);
                    SetPublicVariable(runner, "int4Material", int4Material);
                    SetPublicVariable(runner, "scatterMaterial", scatterMaterial);
                    for (int atlasIndex = 0; atlasIndex < call.Atlases.Count; ++atlasIndex)
                    {
                        ValidatedOnnxInt4Atlas atlas = call.Atlases[atlasIndex];
                        Texture2D texture = CreateOrUpdateByteTexture(
                            transaction,
                            runnerRoot,
                            "atlas-" + atlasIndex.ToString("D4") + ".asset",
                            atlas.Width,
                            atlas.Height,
                            File.ReadAllBytes(atlas.AbsolutePath)
                        );
                        SetPublicVariable(runner, atlas.Name, (Texture)texture);
                    }
                    runners[runnerIndex] = runner;
                }
                SetPublicVariable(model, "int4Runners", runners);
                if (arenaSlots.Length > 0)
                    SetPublicVariable(model, "resultState", arenaSlots[0]);

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
                if (!success || prefab == null)
                    throw new InvalidOperationException(
                        "Could not create ONNX model prefab: " + prefabPath
                    );
                return prefab;
            }
            finally
            {
                GameObject pendingRoot = root;
                EditorApplication.delayCall += () =>
                {
                    if (pendingRoot != null)
                        UnityEngine.Object.DestroyImmediate(pendingRoot);
                };
            }
        }

        private static void BindProgramAsset(
            UdonBehaviour behaviour,
            UdonAssemblyProgramAsset programAsset
        )
        {
            behaviour.programSource = programAsset;
            SerializedObject serialized = new SerializedObject(behaviour);
            serialized.FindProperty("serializedProgramAsset").objectReferenceValue =
                programAsset.SerializedProgramAsset;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void SetIntArray(SerializedProperty property, int[] values)
        {
            property.arraySize = values.Length;
            for (int index = 0; index < values.Length; ++index)
                property.GetArrayElementAtIndex(index).intValue = values[index];
        }

        private static void SetOnnxIoProperties(
            SerializedObject serialized,
            string prefix,
            IList<ValidatedOnnxIo> entries
        )
        {
            SerializedProperty names = serialized.FindProperty(prefix + "Names");
            SerializedProperty dataTypes = serialized.FindProperty(prefix + "DataTypes");
            SerializedProperty words = serialized.FindProperty(prefix + "Words");
            SerializedProperty slots = serialized.FindProperty(prefix + "Slots");
            names.arraySize = entries.Count;
            dataTypes.arraySize = entries.Count;
            words.arraySize = entries.Count;
            slots.arraySize = entries.Count;
            for (int index = 0; index < entries.Count; ++index)
            {
                ValidatedOnnxIo entry = entries[index];
                names.GetArrayElementAtIndex(index).stringValue = entry.Name;
                dataTypes.GetArrayElementAtIndex(index).stringValue = entry.DataType;
                words.GetArrayElementAtIndex(index).intValue = entry.Words;
                slots.GetArrayElementAtIndex(index).intValue = entry.Slot;
            }
        }

        private static UdonAssemblyProgramAsset CreateOrUpdateProgramAsset(
            AssetImportTransaction transaction,
            string programRoot,
            string assembly
        )
        {
            string programDirectory = programRoot + "/Programs";
            EnsureFolders(programDirectory);
            string programPath = programDirectory + "/host_program.asset";
            transaction.Track(programPath);

            UdonAssemblyProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(programPath);
            if (programAsset == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(programPath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + programPath
                    );
                programAsset = ScriptableObject.CreateInstance<UdonAssemblyProgramAsset>();
                programAsset.name = "host_program";
                AssetDatabase.CreateAsset(programAsset, programPath);
            }

            SerializedObject serialized = new SerializedObject(programAsset);
            serialized.Update();
            SerializedProperty source = serialized.FindProperty("udonAssembly");
            if (source == null)
                throw new InvalidDataException(
                    "Worlds SDK Udon Assembly asset has no source field."
                );
            source.stringValue = assembly;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(programAsset);
            AssetDatabase.SaveAssetIfDirty(programAsset);
            serialized.Update();
            if (source.stringValue != assembly)
                throw new InvalidDataException(
                    "Worlds SDK did not persist Udon Assembly source: " + programPath
                );

            string guid = AssetDatabase.AssetPathToGUID(programPath);
            if (string.IsNullOrEmpty(guid))
                throw new InvalidDataException("Unity did not assign a GUID to " + programPath);
            transaction.Track("Assets/SerializedUdonPrograms/" + guid + ".asset");

            programAsset.RefreshProgram();
            if (!string.IsNullOrEmpty(programAsset.AssemblyError))
                throw new InvalidDataException(
                    "Worlds SDK rejected Udon Assembly: "
                        + (programAsset.AssemblyError ?? programPath)
                );

            // A newly created SDK SerializedUdonProgramAsset can retain the
            // empty program it deserialized before RefreshProgram stored the
            // assembled bytes. Persist and synchronously reimport it so the
            // object referenced by the prefab has a fresh deserialization
            // cache in this same Editor session.
            AbstractSerializedUdonProgramAsset serializedProgram =
                programAsset.SerializedProgramAsset;
            string serializedProgramPath = AssetDatabase.GetAssetPath(serializedProgram);
            if (serializedProgram == null || string.IsNullOrEmpty(serializedProgramPath))
                throw new InvalidDataException(
                    "Worlds SDK did not create a serialized Udon program for " + programPath
                );
            AssetDatabase.SaveAssetIfDirty(serializedProgram);
            Resources.UnloadAsset(serializedProgram);
            AssetDatabase.ImportAsset(
                serializedProgramPath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
            programAsset = AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(programPath);
            IUdonProgram compiledProgram = programAsset?.SerializedProgramAsset?.RetrieveProgram();
            if (compiledProgram == null)
                throw new InvalidDataException(
                    "Worlds SDK did not persist the assembled Udon program: " + programPath
                );
            EditorUtility.SetDirty(programAsset);
            return programAsset;
        }

        private static Material CreateOrUpdateMaterial(
            AssetImportTransaction transaction,
            string programRoot,
            ValidatedKernel kernel,
            Shader shader
        )
        {
            string materialDirectory = programRoot + "/Materials";
            EnsureFolders(materialDirectory);
            string materialPath =
                materialDirectory + "/kernel_" + kernel.Id + "_" + kernel.Name + ".mat";
            transaction.Track(materialPath);

            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(materialPath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + materialPath
                    );
                material = new Material(shader);
                material.name = Path.GetFileNameWithoutExtension(materialPath);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            else
            {
                material.shader = shader;
                material.name = Path.GetFileNameWithoutExtension(materialPath);
                EditorUtility.SetDirty(material);
            }
            return material;
        }

        private static Material CreateOrUpdateDisplayMaterial(
            AssetImportTransaction transaction,
            string programRoot,
            Shader shader
        )
        {
            string materialDirectory = programRoot + "/Materials";
            EnsureFolders(materialDirectory);
            string materialPath = materialDirectory + "/display.mat";
            transaction.Track(materialPath);
            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(materialPath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + materialPath
                    );
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            else
            {
                material.shader = shader;
            }
            material.name = Path.GetFileNameWithoutExtension(materialPath);
            EditorUtility.SetDirty(material);
            return material;
        }

        private static Material CreateOrUpdateNamedMaterial(
            AssetImportTransaction transaction,
            string programRoot,
            string fileName,
            Shader shader
        )
        {
            string materialDirectory = programRoot + "/Materials";
            EnsureFolders(materialDirectory);
            string materialPath = materialDirectory + "/" + fileName;
            transaction.Track(materialPath);
            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(materialPath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + materialPath
                    );
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            else
            {
                material.shader = shader;
            }
            material.name = Path.GetFileNameWithoutExtension(fileName);
            EditorUtility.SetDirty(material);
            return material;
        }

        private static Texture2D CreateOrUpdateByteTexture(
            AssetImportTransaction transaction,
            string programRoot,
            string fileName,
            int width,
            int height,
            byte[] bytes
        )
        {
            if (bytes.Length != width * height * 4)
                throw new InvalidDataException(
                    "RGBA32 resource has "
                        + bytes.Length
                        + " bytes; expected "
                        + (width * height * 4)
                        + "."
                );
            string textureDirectory = programRoot + "/Textures";
            EnsureFolders(textureDirectory);
            string texturePath = textureDirectory + "/" + fileName;
            transaction.Track(texturePath);
            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (texture == null)
            {
                texture = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
                AssetDatabase.CreateAsset(texture, texturePath);
            }
            else if (!texture.Reinitialize(width, height, TextureFormat.RGBA32, false))
            {
                throw new InvalidOperationException("Could not resize texture " + texturePath);
            }
            texture.filterMode = FilterMode.Point;
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.anisoLevel = 0;
            texture.LoadRawTextureData(bytes);
            texture.Apply(false, false);
            EditorUtility.SetDirty(texture);
            return texture;
        }

        private static RenderTexture CreateOrUpdateRgbaRenderTexture(
            AssetImportTransaction transaction,
            string programRoot,
            string fileName,
            int width,
            int height
        )
        {
            string textureDirectory = programRoot + "/RenderTextures";
            EnsureFolders(textureDirectory);
            string texturePath = textureDirectory + "/" + fileName;
            transaction.Track(texturePath);
            RenderTexture texture = AssetDatabase.LoadAssetAtPath<RenderTexture>(texturePath);
            if (texture == null)
            {
                texture = new RenderTexture(
                    width,
                    height,
                    0,
                    RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Linear
                );
                AssetDatabase.CreateAsset(texture, texturePath);
            }
            else
            {
                texture.Release();
                texture.width = width;
                texture.height = height;
                texture.depth = 0;
                texture.format = RenderTextureFormat.ARGB32;
            }
            texture.useMipMap = false;
            texture.autoGenerateMips = false;
            texture.antiAliasing = 1;
            if (texture.sRGB)
                throw new InvalidOperationException(
                    "Generated RenderTexture unexpectedly uses sRGB: " + texturePath
                );
            if (!texture.Create() && !texture.IsCreated())
                throw new InvalidOperationException(
                    "Could not create generated RenderTexture: " + texturePath
                );
            // RenderTexture.Create can restore sampling defaults on an asset
            // that was released and resized during a repeated import.
            texture.filterMode = FilterMode.Point;
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.anisoLevel = 0;
            EditorUtility.SetDirty(texture);
            // The importer performs a synchronous refresh before its final
            // transaction commit. Persist this newly created native asset now
            // so that refresh cannot reload Unity's default Bilinear mode.
            AssetDatabase.SaveAssetIfDirty(texture);
            return texture;
        }

        internal static string DeviceBufferFileName(int id, string name)
        {
            return "udoncl_buffer_" + id + "_" + name + ".asset";
        }

        private static Texture2D CreateOrUpdateParticleTexture(
            AssetImportTransaction transaction,
            string programRoot
        )
        {
            string textureDirectory = programRoot + "/Textures";
            EnsureFolders(textureDirectory);
            string texturePath = textureDirectory + "/lj_particles_initial.asset";
            transaction.Track(texturePath);

            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (texture == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(texturePath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + texturePath
                    );
                texture = new Texture2D(
                    ParticleTextureEncoder.TextureWidth,
                    ParticleTextureEncoder.TextureHeight,
                    TextureFormat.RGBA32,
                    false,
                    true
                );
                AssetDatabase.CreateAsset(texture, texturePath);
            }
            else if (
                !texture.Reinitialize(
                    ParticleTextureEncoder.TextureWidth,
                    ParticleTextureEncoder.TextureHeight,
                    TextureFormat.RGBA32,
                    false
                )
            )
            {
                throw new InvalidOperationException(
                    "Could not resize existing particle texture " + texturePath
                );
            }

            texture.name = Path.GetFileNameWithoutExtension(texturePath);
            texture.filterMode = FilterMode.Point;
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.anisoLevel = 0;
            texture.LoadRawTextureData(
                ParticleTextureEncoder.Encode(
                    ParticleTextureEncoder.CreateLennardJonesInitialState()
                )
            );
            texture.Apply(false, false);
            EditorUtility.SetDirty(texture);
            return texture;
        }

        private static RenderTexture CreateOrUpdateRenderTexture(
            AssetImportTransaction transaction,
            string programRoot,
            string fileName
        )
        {
            string textureDirectory = programRoot + "/RenderTextures";
            EnsureFolders(textureDirectory);
            string renderTexturePath = textureDirectory + "/" + fileName;
            transaction.Track(renderTexturePath);

            RenderTexture renderTexture = AssetDatabase.LoadAssetAtPath<RenderTexture>(
                renderTexturePath
            );
            if (renderTexture == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(renderTexturePath) != null)
                    throw new InvalidDataException(
                        "An incompatible asset already occupies " + renderTexturePath
                    );
                renderTexture = new RenderTexture(
                    ParticleTextureEncoder.TextureWidth,
                    ParticleTextureEncoder.TextureHeight,
                    0,
                    RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Linear
                );
                AssetDatabase.CreateAsset(renderTexture, renderTexturePath);
            }
            else
            {
                renderTexture.Release();
                renderTexture.width = ParticleTextureEncoder.TextureWidth;
                renderTexture.height = ParticleTextureEncoder.TextureHeight;
                renderTexture.depth = 0;
                renderTexture.format = RenderTextureFormat.ARGB32;
            }

            renderTexture.name = Path.GetFileNameWithoutExtension(fileName);
            renderTexture.filterMode = FilterMode.Point;
            renderTexture.wrapMode = TextureWrapMode.Clamp;
            renderTexture.useMipMap = false;
            renderTexture.autoGenerateMips = false;
            renderTexture.antiAliasing = 1;
            renderTexture.anisoLevel = 0;
            if (renderTexture.sRGB)
                throw new InvalidOperationException(
                    "Generated RenderTexture unexpectedly uses sRGB: " + renderTexturePath
                );
            if (!renderTexture.Create() && !renderTexture.IsCreated())
                throw new InvalidOperationException(
                    "Could not create generated RenderTexture: " + renderTexturePath
                );
            EditorUtility.SetDirty(renderTexture);
            return renderTexture;
        }

        private static void CreateOrUpdateDemoPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture initialState,
            RenderTexture stateA,
            RenderTexture stateB,
            Material driftMaterial,
            Material kickMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/LennardJonesDemo.prefab";
            transaction.Track(prefabPath);

            GameObject root = new GameObject("udoncl Lennard-Jones ice");
            try
            {
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 4.0f, 4.0f);
                collider.isTrigger = true;

                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Touch to heat / melt";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "initialState", initialState);
                SetPublicVariable(behaviour, "stateA", stateA);
                SetPublicVariable(behaviour, "stateB", stateB);
                SetPublicVariable(behaviour, "driftMaterial", driftMaterial);
                SetPublicVariable(behaviour, "kickMaterial", kickMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "amplitude", 0.0f);
                SetPublicVariable(behaviour, "frame", 0.0f);

                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "LJ 3D particle volume";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 4.0f, 4.0f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create LJ demo prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateMnistPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            RenderTexture drawing,
            RenderTexture drawingScratch,
            RenderTexture packed,
            RenderTexture logits,
            Material packMaterial,
            Material brushMaterial,
            Material convMaterial,
            Material poolMaterial,
            Material denseMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/MnistDeviceDemo.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("udoncl MNIST Device demo");
            try
            {
                root.transform.position = new Vector3(8.0f, 1.5f, 0.0f);
                BoxCollider trigger = root.AddComponent<BoxCollider>();
                trigger.size = new Vector3(4.4f, 4.4f, 0.6f);
                trigger.isTrigger = true;
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Clear MNIST drawing";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "drawingTexture", drawing);
                SetPublicVariable(behaviour, "drawingScratch", drawingScratch);
                SetPublicVariable(behaviour, "packedState", packed);
                SetPublicVariable(behaviour, "logitsState", logits);
                SetPublicVariable(behaviour, "packMaterial", packMaterial);
                SetPublicVariable(behaviour, "brushMaterial", brushMaterial);
                SetPublicVariable(behaviour, "convMaterial", convMaterial);
                SetPublicVariable(behaviour, "poolMaterial", poolMaterial);
                SetPublicVariable(behaviour, "denseMaterial", denseMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject board = GameObject.CreatePrimitive(PrimitiveType.Cube);
                board.name = "3D digit drawing board and prediction";
                board.transform.SetParent(root.transform, false);
                board.transform.localScale = new Vector3(4.0f, 4.0f, 0.12f);
                board.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;
                Collider boardCollider = board.GetComponent<Collider>();
                if (boardCollider != null)
                    UnityEngine.Object.DestroyImmediate(boardCollider);
                SetPublicVariable(behaviour, "boardTransform", board.transform);
                SetPublicVariable(behaviour, "rightFinger", HumanBodyBones.RightIndexDistal);

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create MNIST Device demo prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateTinyLlamaPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture initialState,
            RenderTexture stateA,
            RenderTexture stateB,
            RenderTexture argmaxState,
            Material packMaterial,
            IReadOnlyList<Material> passes,
            Material argmaxMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/TinyLlamaDeviceDemo.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("udoncl Tiny LLaMA Device demo");
            try
            {
                root.transform.position = new Vector3(14.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.4f, 2.8f, 0.5f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Submit Tiny LLaMA prompt";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.Manual;
                SetPublicVariable(behaviour, "initialState", initialState);
                SetPublicVariable(behaviour, "stateA", stateA);
                SetPublicVariable(behaviour, "stateB", stateB);
                SetPublicVariable(behaviour, "argmaxState", argmaxState);
                SetPublicVariable(behaviour, "packMaterial", packMaterial);
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
                for (int index = 0; index < symbols.Length; ++index)
                    SetPublicVariable(behaviour, symbols[index], passes[index]);
                SetPublicVariable(behaviour, "argmaxMaterial", argmaxMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "readbackFormat", TextureFormat.RGBA32);
                CreateTinyLlamaChatUi(root, behaviour);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "Tiny LLaMA logits and predicted token";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                display.transform.localPosition = new Vector3(0.0f, 0.0f, 0.08f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create Tiny LLaMA Device demo prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonBlasSgemmPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture inputState,
            RenderTexture outputState,
            Material sgemmMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonBlasSgemm.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonBLAS SGEMM");
            try
            {
                root.transform.position = new Vector3(20.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonBLAS SGEMM";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "inputState", inputState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "sgemmMaterial", sgemmMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "m", 4);
                SetPublicVariable(behaviour, "n", 5);
                SetPublicVariable(behaviour, "k", 3);
                SetPublicVariable(behaviour, "transa", 0);
                SetPublicVariable(behaviour, "transb", 0);
                SetPublicVariable(behaviour, "lda", 4);
                SetPublicVariable(behaviour, "ldb", 3);
                SetPublicVariable(behaviour, "ldc", 4);
                SetPublicVariable(behaviour, "alpha", 0.75f);
                SetPublicVariable(behaviour, "beta", -0.25f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonBLAS matrix result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonBLAS SGEMM prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonBlasSgemvPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture inputState,
            RenderTexture outputState,
            Material sgemvMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonBlasSgemv.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonBLAS SGEMV");
            try
            {
                root.transform.position = new Vector3(20.0f, 4.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonBLAS SGEMV";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "inputState", inputState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "sgemvMaterial", sgemvMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "trans", 2);
                SetPublicVariable(behaviour, "m", 3);
                SetPublicVariable(behaviour, "n", 5);
                SetPublicVariable(behaviour, "lda", 4);
                SetPublicVariable(behaviour, "incx", -2);
                SetPublicVariable(behaviour, "incy", -3);
                SetPublicVariable(behaviour, "alpha", 0.5f);
                SetPublicVariable(behaviour, "beta", 1.0f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonBLAS SGEMV vector result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonBLAS SGEMV prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonAtenWeightInt4packMmPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture weightAtlas,
            Texture activationState,
            RenderTexture assemblyState,
            RenderTexture partialState,
            RenderTexture outputState,
            RenderTexture resultState,
            Material composeMaterial,
            Material int4Material,
            Material scatterMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonAtenWeightInt4packMm.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonATen weight INT4 pack MM");
            try
            {
                root.transform.position = new Vector3(24.0f, 4.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonATen INT4 MM";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "weightAtlas", weightAtlas);
                SetPublicVariable(behaviour, "activationState", activationState);
                SetPublicVariable(behaviour, "assemblyState", assemblyState);
                SetPublicVariable(behaviour, "partialState", partialState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "resultState", resultState);
                SetPublicVariable(behaviour, "composeMaterial", composeMaterial);
                SetPublicVariable(behaviour, "int4Material", int4Material);
                SetPublicVariable(behaviour, "scatterMaterial", scatterMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "n", 8);
                SetPublicVariable(behaviour, "k", 128);
                SetPublicVariable(behaviour, "rowStride", 72);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonATen INT4 matrix result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonATen INT4 MM prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonBlasSaxpyPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture inputState,
            RenderTexture outputState,
            Material saxpyMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonBlasSaxpy.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonBLAS SAXPY");
            try
            {
                root.transform.position = new Vector3(20.0f, 7.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonBLAS SAXPY";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "inputState", inputState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "saxpyMaterial", saxpyMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "n", 16);
                SetPublicVariable(behaviour, "incx", -2);
                SetPublicVariable(behaviour, "incy", 3);
                SetPublicVariable(behaviour, "alpha", -0.75f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonBLAS SAXPY vector result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonBLAS SAXPY prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonBlasSdotPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture inputState,
            RenderTexture outputState,
            RenderTexture scratchState,
            Material productMaterial,
            Material reduceMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonBlasSdot.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonBLAS SDOT");
            try
            {
                root.transform.position = new Vector3(20.0f, 10.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonBLAS SDOT";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "inputState", inputState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "scratchState", scratchState);
                SetPublicVariable(behaviour, "productMaterial", productMaterial);
                SetPublicVariable(behaviour, "reduceMaterial", reduceMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "n", 23);
                SetPublicVariable(behaviour, "incx", -2);
                SetPublicVariable(behaviour, "incy", 3);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonBLAS SDOT scalar result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonBLAS SDOT prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonDnnConvolutionPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material convolutionMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonDnnConvolution.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonDNN convolution");
            try
            {
                root.transform.position = new Vector3(26.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 3.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonDNN convolution";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "convolutionMaterial", convolutionMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "inputShape", 17441);
                SetPublicVariable(behaviour, "filterShape", 13091);
                SetPublicVariable(behaviour, "convolutionShape", 1118481);
                SetPublicVariable(behaviour, "convolutionMode", 1);
                SetPublicVariable(behaviour, "outputShape", 17457);
                SetPublicVariable(behaviour, "alpha", 0.75f);
                SetPublicVariable(behaviour, "beta", -0.25f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonDNN convolution result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 3.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonDNN convolution prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonDnnActivationPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material activationMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonDnnActivation.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonDNN activation");
            try
            {
                root.transform.position = new Vector3(32.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonDNN activation";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "activationMaterial", activationMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "words", 64);
                SetPublicVariable(behaviour, "mode", 4);
                SetPublicVariable(behaviour, "nanPropagation", 0);
                SetPublicVariable(behaviour, "coefficient", 1.2f);
                SetPublicVariable(behaviour, "alpha", 0.5f);
                SetPublicVariable(behaviour, "beta", -0.5f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonDNN activation result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonDNN activation prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonDnnPoolingPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material poolingMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonDnnPooling.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonDNN pooling");
            try
            {
                root.transform.position = new Vector3(38.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 2.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonDNN pooling";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "poolingMaterial", poolingMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "inputShape", 17441);
                SetPublicVariable(behaviour, "poolingShape", 2232627);
                SetPublicVariable(behaviour, "poolingMode", 0);
                SetPublicVariable(behaviour, "nanPropagation", 0);
                SetPublicVariable(behaviour, "outputShape", 8737);
                SetPublicVariable(behaviour, "alpha", 1.0f);
                SetPublicVariable(behaviour, "beta", 0.0f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonDNN pooling result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonDNN pooling prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonDnnSoftmaxPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material softmaxMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonDnnSoftmax.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonDNN softmax");
            try
            {
                root.transform.position = new Vector3(44.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 3.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonDNN softmax";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "softmaxMaterial", softmaxMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "inputShape", 8753);
                SetPublicVariable(behaviour, "algorithm", 1);
                SetPublicVariable(behaviour, "mode", 1);
                SetPublicVariable(behaviour, "alpha", 1.0f);
                SetPublicVariable(behaviour, "beta", 0.0f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonDNN softmax result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 3.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonDNN softmax prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonDnnBatchNormPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material batchnormMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonDnnBatchNorm.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonDNN batch normalization");
            try
            {
                root.transform.position = new Vector3(50.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 3.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonDNN batch normalization";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "batchnormMaterial", batchnormMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "inputShape", 8753);
                SetPublicVariable(behaviour, "mode", 1);
                SetPublicVariable(behaviour, "epsilon", 1.0e-4f);
                SetPublicVariable(behaviour, "alpha", 1.0f);
                SetPublicVariable(behaviour, "beta", 0.0f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonDNN batch normalization result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 3.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonDNN batch normalization prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonDnnOpTensorPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material opTensorMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonDnnOpTensor.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonDNN OpTensor");
            try
            {
                root.transform.position = new Vector3(56.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 3.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonDNN OpTensor";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "opTensorMaterial", opTensorMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "aShape", 8753);
                SetPublicVariable(behaviour, "bShape", 4401);
                SetPublicVariable(behaviour, "cShape", 8753);
                SetPublicVariable(behaviour, "operation", 0);
                SetPublicVariable(behaviour, "nanPropagation", 0);
                SetPublicVariable(behaviour, "alphaA", 1.0f);
                SetPublicVariable(behaviour, "alphaB", 0.75f);
                SetPublicVariable(behaviour, "beta", -0.25f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonDNN OpTensor result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 3.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonDNN OpTensor prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonAtenRmsNormPrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture packedState,
            RenderTexture outputState,
            Material rmsNormMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonAtenRmsNorm.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonATen RMS normalization");
            try
            {
                root.transform.position = new Vector3(62.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 3.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonATen RMS normalization";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "packedState", packedState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "rmsNormMaterial", rmsNormMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "words", 16);
                SetPublicVariable(behaviour, "normalizedWords", 8);
                SetPublicVariable(behaviour, "epsilon", 1.0e-5f);
                SetPublicVariable(behaviour, "alpha", 1.0f);
                SetPublicVariable(behaviour, "beta", 0.0f);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonATen RMS normalization result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 3.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonATen RMS normalization prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateOrUpdateUdonAtenTilePrefab(
            AssetImportTransaction transaction,
            string programRoot,
            UdonAssemblyProgramAsset programAsset,
            Texture inputState,
            RenderTexture outputState,
            Material tileMaterial,
            Material displayMaterial
        )
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/UdonAtenTile.prefab";
            transaction.Track(prefabPath);
            GameObject root = new GameObject("UdonATen Tile");
            try
            {
                root.transform.position = new Vector3(68.0f, 1.5f, 0.0f);
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(4.0f, 3.0f, 0.4f);
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run UdonATen Tile";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "inputState", inputState);
                SetPublicVariable(behaviour, "outputState", outputState);
                SetPublicVariable(behaviour, "tileMaterial", tileMaterial);
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SetPublicVariable(behaviour, "inputN", 1);
                SetPublicVariable(behaviour, "inputC", 2);
                SetPublicVariable(behaviour, "inputH", 2);
                SetPublicVariable(behaviour, "inputW", 3);
                SetPublicVariable(behaviour, "repeatN", 1);
                SetPublicVariable(behaviour, "repeatC", 1);
                SetPublicVariable(behaviour, "repeatH", 2);
                SetPublicVariable(behaviour, "repeatW", 3);
                SetPublicVariable(behaviour, "outputWords", 72);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "UdonATen Tile result";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 3.0f, 0.12f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create UdonATen Tile prefab: " + prefabPath
                    );
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void CreateTinyLlamaChatUi(GameObject root, UdonBehaviour behaviour)
        {
            Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            GameObject canvasObject = new GameObject("Shared world chat");
            canvasObject.transform.SetParent(root.transform, false);
            Canvas canvas = canvasObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasObject.AddComponent<CanvasScaler>();
            canvasObject.AddComponent<GraphicRaycaster>();
            Type uiShapeType = typeof(VRC.SDKBase.VRCPlayerApi).Assembly.GetType(
                "VRC.SDKBase.VRC_UiShape"
            );
            if (uiShapeType == null)
                throw new InvalidOperationException("The Worlds SDK did not provide VRC_UiShape.");
            canvasObject.AddComponent(uiShapeType);
            RectTransform canvasRect = canvas.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(800.0f, 520.0f);
            canvasRect.localScale = Vector3.one * 0.005f;
            canvasRect.localPosition = new Vector3(0.0f, 0.0f, -0.1f);

            GameObject panel = CreateUiObject("Chat panel", canvasObject.transform);
            Image panelImage = panel.AddComponent<Image>();
            panelImage.color = new Color(0.015f, 0.025f, 0.05f, 0.92f);
            Stretch(panel.GetComponent<RectTransform>(), 0.0f, 0.0f, 0.0f, 0.0f);

            GameObject historyObject = CreateUiObject(
                "Shared conversation history",
                panel.transform
            );
            Text history = historyObject.AddComponent<Text>();
            history.font = font;
            history.fontSize = 28;
            history.color = Color.white;
            history.alignment = TextAnchor.UpperLeft;
            history.horizontalOverflow = HorizontalWrapMode.Wrap;
            history.verticalOverflow = VerticalWrapMode.Truncate;
            Stretch(historyObject.GetComponent<RectTransform>(), 24.0f, 24.0f, 24.0f, 112.0f);

            GameObject inputObject = CreateUiObject("Local prompt input", panel.transform);
            Image inputBackground = inputObject.AddComponent<Image>();
            inputBackground.color = new Color(0.1f, 0.14f, 0.22f, 1.0f);
            RectTransform inputRect = inputObject.GetComponent<RectTransform>();
            inputRect.anchorMin = new Vector2(0.0f, 0.0f);
            inputRect.anchorMax = new Vector2(1.0f, 0.0f);
            inputRect.pivot = new Vector2(0.5f, 0.0f);
            inputRect.offsetMin = new Vector2(24.0f, 24.0f);
            inputRect.offsetMax = new Vector2(-24.0f, 92.0f);
            InputField input = inputObject.AddComponent<InputField>();
            input.characterLimit = 160;
            input.lineType = InputField.LineType.SingleLine;

            GameObject textObject = CreateUiObject("Text", inputObject.transform);
            Text inputText = textObject.AddComponent<Text>();
            inputText.font = font;
            inputText.fontSize = 26;
            inputText.color = Color.white;
            inputText.alignment = TextAnchor.MiddleLeft;
            Stretch(textObject.GetComponent<RectTransform>(), 14.0f, 14.0f, 8.0f, 8.0f);
            input.textComponent = inputText;

            GameObject placeholderObject = CreateUiObject("Placeholder", inputObject.transform);
            Text placeholder = placeholderObject.AddComponent<Text>();
            placeholder.font = font;
            placeholder.fontSize = 24;
            placeholder.fontStyle = FontStyle.Italic;
            placeholder.color = new Color(0.7f, 0.75f, 0.82f, 0.7f);
            placeholder.alignment = TextAnchor.MiddleLeft;
            placeholder.text = "Type locally, then interact with the panel to submit";
            Stretch(placeholderObject.GetComponent<RectTransform>(), 14.0f, 14.0f, 8.0f, 8.0f);
            input.placeholder = placeholder;

            SetPublicVariable(behaviour, "promptInput", input);
            SetPublicVariable(behaviour, "historyText", history);
            SetPublicVariable(
                behaviour,
                "responseWords",
                new[]
                {
                    "<pad>",
                    "hello",
                    "world",
                    "yes",
                    "no",
                    "I",
                    "you",
                    "like",
                    "VRChat",
                    "GPU",
                    "water",
                    "ice",
                    "tell",
                    "more",
                    "thanks",
                    ".",
                }
            );
            SetPublicVariable(behaviour, "readbackPixels", new Color32[1]);
        }

        private static GameObject CreateUiObject(string name, Transform parent)
        {
            GameObject result = new GameObject(name, typeof(RectTransform));
            result.transform.SetParent(parent, false);
            return result;
        }

        private static void Stretch(
            RectTransform transform,
            float left,
            float right,
            float bottom,
            float top
        )
        {
            transform.anchorMin = Vector2.zero;
            transform.anchorMax = Vector2.one;
            transform.offsetMin = new Vector2(left, bottom);
            transform.offsetMax = new Vector2(-right, -top);
        }

        private static void SetPublicVariable<T>(UdonBehaviour behaviour, string symbol, T value)
        {
            if (
                behaviour.publicVariables.TryGetVariableType(symbol, out Type existingType)
                && existingType != typeof(T)
            )
                behaviour.publicVariables.RemoveVariable(symbol);
            if (!behaviour.publicVariables.TrySetVariableValue(symbol, value))
            {
                behaviour.publicVariables.RemoveVariable(symbol);
                if (!behaviour.publicVariables.TryAddVariable(new UdonVariable<T>(symbol, value)))
                    throw new InvalidOperationException(
                        "Could not bind Udon public variable: " + symbol
                    );
            }
        }

        private static void EnsureFolders(string assetPath)
        {
            string normalized = assetPath.Replace('\\', '/');
            string[] parts = normalized.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                EnsureFolder(current, parts[i]);
                current = next;
            }
        }

        private static void EnsureFolder(string parent, string name)
        {
            string path = parent + "/" + name;
            if (!AssetDatabase.IsValidFolder(path))
            {
                if (!AssetDatabase.IsValidFolder(parent))
                    throw new DirectoryNotFoundException(
                        "Unity asset folder does not exist: " + parent
                    );
                AssetDatabase.CreateFolder(parent, name);
            }
        }
    }

    internal static class UdonclImportMenu
    {
        [MenuItem("Tools/udoncl/Import Artifacts...")]
        private static void ImportArtifacts()
        {
            string directory = EditorUtility.OpenFolderPanel(
                "Select udoncl artifact directory",
                "",
                ""
            );
            if (string.IsNullOrEmpty(directory))
                return;

            ImportAndOfferScene(directory);
        }

        internal static void ImportAndOfferScene(string directory)
        {
            try
            {
                string destination = UdonclArtifactImporter.Import(directory);
                Lemolatoon.Udoncl.UdonOnnxModelAsset modelAsset =
                    AssetDatabase.LoadAssetAtPath<Lemolatoon.Udoncl.UdonOnnxModelAsset>(
                        destination + "/Models/UdonOnnxModel.asset"
                    );
                if (modelAsset == null)
                    EditorUtility.DisplayDialog(
                        "udoncl Import",
                        "Imported assets to " + destination,
                        "OK"
                    );
                else
                {
                    int choice = EditorUtility.DisplayDialogComplex(
                        "udoncl ONNX Import",
                        "Imported the ONNX model to "
                            + destination
                            + ". Add its runtime prefab to the active scene?",
                        "Add to Scene",
                        "Close",
                        "Show Model Asset"
                    );
                    if (choice == 0)
                        UdonOnnxModelEditorUtility.InstantiateRuntime(modelAsset);
                    else if (choice == 2)
                    {
                        Selection.activeObject = modelAsset;
                        EditorGUIUtility.PingObject(modelAsset);
                    }
                }
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("udoncl Import Failed", exception.Message, "OK");
            }
        }
    }
}
