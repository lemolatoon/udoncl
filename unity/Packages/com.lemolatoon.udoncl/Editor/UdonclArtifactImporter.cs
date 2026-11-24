using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UdonSharp;
using UdonSharp.Compiler;
using UdonSharpEditor;
using UnityEditor;
using UnityEngine;
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
                    InstallTypedBinding(transaction, artifacts, programRoot);
                    string sourceDirectory = programRoot + "/Sources";
                    EnsureFolders(sourceDirectory);
                    string assemblySourcePath =
                        sourceDirectory
                        + "/"
                        + Path.GetFileName(artifacts.AssemblyAssetPath)
                        + ".txt";
                    transaction.CopyFile(artifacts.AssemblyAssetPath, assemblySourcePath);
                    foreach (ValidatedResource resource in artifacts.Resources)
                    {
                        if (resource.Name == "raw_heap_initializer" && resource.Kind == "bytes")
                            transaction.CopyFile(
                                resource.AbsolutePath,
                                sourceDirectory + "/raw-heap.bytes"
                            );
                        else if (resource.Kind == "json")
                        {
                            string resourcePath =
                                programRoot + "/Resources/" + resource.RelativePath;
                            EnsureFolders(Path.GetDirectoryName(resourcePath).Replace('\\', '/'));
                            transaction.CopyFile(resource.AbsolutePath, resourcePath);
                        }
                    }
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
                    }

                    CreateOrUpdateDispatchMaterials(
                        transaction,
                        programRoot,
                        artifacts.DispatchMaterials,
                        kernelMaterials
                    );

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
                            buffer.Height,
                            DeviceRenderTextureFormat(buffer.Format)
                        );
                    // A word-texel program snapshots the destination of a
                    // partial transfer into this texture.
                    if (artifacts.TransferScratch != null)
                        CreateOrUpdateRgbaRenderTexture(
                            transaction,
                            programRoot,
                            TransferScratchFileName,
                            artifacts.TransferScratch.Width,
                            artifacts.TransferScratch.Height,
                            DeviceRenderTextureFormat(artifacts.TransferScratch.Format)
                        );
                    foreach (ValidatedStaticTensor tensor in artifacts.StaticTensors)
                    {
                        byte[] bytes = File.ReadAllBytes(tensor.AbsolutePath);
                        // Rows of texels hold consecutive words, so a word
                        // texture's bytes are the tensor's bytes padded.
                        Array.Resize(ref bytes, tensor.Width * tensor.Height * tensor.TexelBytes);
                        if (tensor.Format == "rgba32_float")
                            CreateOrUpdateWordTexture(
                                transaction,
                                programRoot,
                                StaticTensorFileName(tensor.Id, tensor.Name),
                                tensor.Width,
                                tensor.Height,
                                bytes
                            );
                        else
                            CreateOrUpdateByteTexture(
                                transaction,
                                programRoot,
                                StaticTensorFileName(tensor.Id, tensor.Name),
                                tensor.Width,
                                tensor.Height,
                                bytes
                            );
                    }
                    if (artifacts.ArtifactStage == "unity-udoncl-library")
                    {
                        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                        transaction.Commit();
                        GeneratedBindingFinalizer.Schedule();
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

        private static void InstallTypedBinding(
            AssetImportTransaction transaction,
            ValidatedArtifact artifacts,
            string programRoot
        )
        {
            if (string.IsNullOrEmpty(artifacts.BindingSourceAssetPath))
            {
                // udonclc writes the U# binding (binding.cs) of every library
                // it can bind; the importer no longer generates one.
                UdonclManifest manifest = JsonUtility.FromJson<UdonclManifest>(
                    File.ReadAllText(artifacts.ManifestAssetPath)
                );
                if (manifest == null || string.IsNullOrEmpty(manifest.physical_export_abi))
                    return;
                throw new InvalidDataException(
                    "Artifact '"
                        + artifacts.ProgramName
                        + "' has exports but no program.binding_source; "
                        + "rebuild the artifact with the current udonclc."
                );
            }

            string bindingRoot = programRoot + "/Bindings";
            EnsureFolders(bindingRoot);
            string bindingPath = bindingRoot + "/" + artifacts.BindingClass + ".cs";
            transaction.CopyFile(artifacts.BindingSourceAssetPath, bindingPath);
            CreateOrUpdateUdonSharpProgramAsset(
                transaction,
                bindingPath,
                bindingRoot + "/" + artifacts.BindingClass + ".asset"
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

        internal static void EnsureCompiledUdonSharpProgramAsset(
            string scriptPath,
            string programAssetPath
        )
        {
            WarmUdonSharpAssemblyCache();
            MonoScript script = AssetDatabase.LoadAssetAtPath<MonoScript>(scriptPath);
            if (script == null)
                throw new InvalidDataException(
                    "Unity could not import UdonSharp source: " + scriptPath
                );
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
            else
            {
                programAsset.sourceCsScript = script;
            }
            programAsset.ScriptVersion = UdonSharpProgramVersion.CurrentVersion;
            EditorUtility.SetDirty(programAsset);
            AssetDatabase.ImportAsset(programAssetPath, ImportAssetOptions.ForceSynchronousImport);
            UdonSharpCompilerV1.CompileSync();
            programAsset = AssetDatabase.LoadAssetAtPath<UdonSharpProgramAsset>(programAssetPath);
            if (
                programAsset == null
                || programAsset.SerializedProgramAsset == null
                || programAsset.CompiledVersion != UdonSharpProgramVersion.CurrentVersion
            )
                throw new InvalidDataException(
                    "Unity could not compile UdonSharp source: " + scriptPath
                );
        }

        internal static UdonBehaviour AddCompiledUdonSharpBehaviour(
            GameObject target,
            string programAssetPath
        )
        {
            UdonSharpProgramAsset programAsset =
                AssetDatabase.LoadAssetAtPath<UdonSharpProgramAsset>(programAssetPath);
            if (programAsset == null || programAsset.SerializedProgramAsset == null)
                throw new InvalidDataException(
                    "Compiled UdonSharp program asset is missing: " + programAssetPath
                );
            UdonBehaviour behaviour = target.AddComponent<UdonBehaviour>();
            behaviour.programSource = programAsset;
            behaviour.SyncMethod = SyncType.None;
            SerializedObject serialized = new SerializedObject(behaviour);
            serialized.FindProperty("serializedProgramAsset").objectReferenceValue =
                programAsset.SerializedProgramAsset;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            return behaviour;
        }

        internal static UdonBehaviour FindUdonSharpBacking(GameObject target)
        {
            foreach (UdonBehaviour behaviour in target.GetComponents<UdonBehaviour>())
                if (behaviour.programSource is UdonSharpProgramAsset)
                    return behaviour;
            return null;
        }

        internal static UdonBehaviour FindUdonSharpBacking(GameObject target, string proxyTypeName)
        {
            foreach (UdonSharpBehaviour proxy in target.GetComponents<UdonSharpBehaviour>())
                if (proxy.GetType().FullName == proxyTypeName)
                    return UdonSharpEditorUtility.GetBackingUdonBehaviour(proxy);
            return null;
        }

        internal static void InitializeUdonBehaviours(GameObject target)
        {
            foreach (UdonBehaviour behaviour in target.GetComponents<UdonBehaviour>())
                behaviour.InitializeUdonContent();
        }

        internal static string GetUdonBehaviourGlobalObjectId(GameObject target)
        {
            return GlobalObjectId
                .GetGlobalObjectIdSlow(target.GetComponent<UdonBehaviour>())
                .ToString();
        }

        internal static string GetUdonEventReceiverGlobalObjectId(IUdonEventReceiver receiver)
        {
            return GlobalObjectId.GetGlobalObjectIdSlow((UdonBehaviour)receiver).ToString();
        }

        internal static void SetUdonProgramVariable(
            UdonBehaviour behaviour,
            string symbol,
            object value
        )
        {
            behaviour.SetProgramVariable(symbol, value);
        }

        internal static object GetUdonProgramVariable(UdonBehaviour behaviour, string symbol)
        {
            return behaviour.GetProgramVariable(symbol);
        }

        internal static void SendUdonCustomEvent(UdonBehaviour behaviour, string eventName)
        {
            behaviour.SendCustomEvent(eventName);
        }

        private static void WarmUdonSharpAssemblyCache()
        {
            const string typeName = "UdonSharp.Compiler.Udon.CompilerUdonInterface";
            Type interfaceType = typeof(UdonSharpCompilerV1).Assembly.GetType(typeName);
            PropertyInfo definitions = interfaceType?.GetProperty(
                "UdonSharpAssemblyDefinitions",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static
            );
            if (definitions == null || definitions.GetValue(null) == null)
                throw new InvalidDataException(
                    "Unity could not initialize the UdonSharp assembly cache."
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
            SerializedProperty assemblyError = serialized.FindProperty("assemblyError");
            if (assemblyError != null)
                assemblyError.stringValue = null;
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
            string serializedProgramPath = "Assets/SerializedUdonPrograms/" + guid + ".asset";
            transaction.Track(serializedProgramPath);

            SerializedUdonProgramAsset serializedProgram =
                AssetDatabase.LoadAssetAtPath<SerializedUdonProgramAsset>(serializedProgramPath);
            if (serializedProgram == null)
            {
                EnsureFolders("Assets/SerializedUdonPrograms");
                serializedProgram = ScriptableObject.CreateInstance<SerializedUdonProgramAsset>();
                serializedProgram.name = guid;
                AssetDatabase.CreateAsset(serializedProgram, serializedProgramPath);
            }
            SerializedObject programReference = new SerializedObject(programAsset);
            SerializedProperty serializedProgramReference = programReference.FindProperty(
                "serializedUdonProgramAsset"
            );
            if (serializedProgramReference == null)
                throw new InvalidDataException(
                    "Worlds SDK Udon Assembly asset has no serialized program field."
                );
            serializedProgramReference.objectReferenceValue = serializedProgram;
            programReference.ApplyModifiedPropertiesWithoutUndo();

            IUdonProgram assembledProgram;
            try
            {
                assembledProgram = UdonAssemblyCompiler.Assemble(assembly);
            }
            catch (Exception exception)
            {
                throw new InvalidDataException(
                    "Worlds SDK rejected Udon Assembly: " + exception.Message,
                    exception
                );
            }
            serializedProgram.StoreProgram(assembledProgram);

            // A newly created SDK SerializedUdonProgramAsset can retain the
            // empty program it deserialized before the custom assembler stored the
            // assembled bytes. Persist and synchronously reimport it so the
            // object referenced by the prefab has a fresh deserialization
            // cache in this same Editor session.
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

        /// <summary>
        /// Creates one Material per dispatch site from its kernel's material.
        /// The Udon program binds each site's fixed arguments to its own
        /// material once and only rebinds the values that change, so sites
        /// must not share a material. Materials of sites that no longer exist
        /// are deleted.
        /// </summary>
        internal static void CreateOrUpdateDispatchMaterials(
            AssetImportTransaction transaction,
            string programRoot,
            IList<ValidatedDispatchMaterial> dispatchMaterials,
            IDictionary<int, Material> kernelMaterials
        )
        {
            string materialDirectory = programRoot + "/Materials/Dispatch";
            HashSet<string> current = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (dispatchMaterials != null && dispatchMaterials.Count != 0)
            {
                EnsureFolders(materialDirectory);
                foreach (ValidatedDispatchMaterial dispatch in dispatchMaterials)
                {
                    if (!kernelMaterials.TryGetValue(dispatch.Kernel.Id, out Material kernel))
                        throw new InvalidDataException(
                            "Dispatch material " + dispatch.Slot + " has no kernel material."
                        );
                    string materialPath = DispatchMaterialAssetPath(
                        programRoot,
                        dispatch.AssetName
                    );
                    current.Add(materialPath);
                    transaction.Track(materialPath);
                    Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
                    if (material == null)
                    {
                        if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(materialPath) != null)
                            throw new InvalidDataException(
                                "An incompatible asset already occupies " + materialPath
                            );
                        material = new Material(kernel.shader);
                        material.name = dispatch.AssetName;
                        AssetDatabase.CreateAsset(material, materialPath);
                    }
                    else
                    {
                        material.shader = kernel.shader;
                        material.name = dispatch.AssetName;
                        EditorUtility.SetDirty(material);
                    }
                }
            }
            if (!AssetDatabase.IsValidFolder(materialDirectory))
                return;
            foreach (
                string guid in AssetDatabase.FindAssets("t:Material", new[] { materialDirectory })
            )
            {
                string stalePath = AssetDatabase.GUIDToAssetPath(guid);
                if (current.Contains(stalePath))
                    continue;
                transaction.Track(stalePath);
                AssetDatabase.DeleteAsset(stalePath);
            }
        }

        /// <summary>The asset path of the dispatch material named <paramref name="assetName"/>.</summary>
        public static string DispatchMaterialAssetPath(string programRoot, string assetName)
        {
            return programRoot + "/Materials/Dispatch/" + assetName + ".mat";
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

        internal const string TransferScratchFileName = "udoncl_transfer_scratch.asset";

        /// <summary>
        /// The RenderTexture format of a Device buffer texel format: four
        /// words per RGBA32 float texel, or one word per RGBA8 texel.
        /// </summary>
        private static RenderTextureFormat DeviceRenderTextureFormat(string format)
        {
            return format == "rgba32_float"
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGB32;
        }

        /// <summary>
        /// An immutable word texture: four little-endian words per RGBA32
        /// float texel, loaded bit for bit (raw data, no conversion).
        /// </summary>
        private static Texture2D CreateOrUpdateWordTexture(
            AssetImportTransaction transaction,
            string programRoot,
            string fileName,
            int width,
            int height,
            byte[] bytes
        )
        {
            if (bytes.Length != width * height * 16)
                throw new InvalidDataException(
                    "Word texture resource has "
                        + bytes.Length
                        + " bytes; expected "
                        + (width * height * 16)
                        + "."
                );
            string textureDirectory = programRoot + "/Textures";
            EnsureFolders(textureDirectory);
            string texturePath = textureDirectory + "/" + fileName;
            transaction.Track(texturePath);
            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (texture == null)
            {
                texture = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
                AssetDatabase.CreateAsset(texture, texturePath);
            }
            else if (!texture.Reinitialize(width, height, TextureFormat.RGBAFloat, false))
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
            return CreateOrUpdateRgbaRenderTexture(
                transaction,
                programRoot,
                fileName,
                width,
                height,
                RenderTextureFormat.ARGB32
            );
        }

        private static RenderTexture CreateOrUpdateRgbaRenderTexture(
            AssetImportTransaction transaction,
            string programRoot,
            string fileName,
            int width,
            int height,
            RenderTextureFormat format
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
                    format,
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
                texture.format = format;
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

        internal static string StaticTensorFileName(int id, string name)
        {
            return "udoncl_static_tensor_" + id + "_" + SanitizeFileComponent(name) + ".asset";
        }

        private static string SanitizeFileComponent(string value)
        {
            char[] characters = value.ToCharArray();
            for (int index = 0; index < characters.Length; ++index)
                if (!char.IsLetterOrDigit(characters[index]) && characters[index] != '_')
                    characters[index] = '_';
            return new string(characters);
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

        internal static string ImportAndOfferScene(string directory)
        {
            try
            {
                string destination = UdonclArtifactImporter.Import(directory);
                UdonclManifest manifest = JsonUtility.FromJson<UdonclManifest>(
                    File.ReadAllText(destination + "/manifest.json")
                );
                bool hasGeneratedBinding =
                    manifest != null
                    && manifest.artifact_stage == "unity-udoncl-library"
                    && manifest.program != null
                    && !string.IsNullOrEmpty(manifest.program.name);
                if (!hasGeneratedBinding)
                    EditorUtility.DisplayDialog(
                        "udoncl Import",
                        "Imported assets to " + destination,
                        "OK"
                    );
                else if (
                    EditorUtility.DisplayDialog(
                        "udoncl Library Import",
                        "Imported assets to "
                            + destination
                            + ". Add the generated binding prefab to the active scene after its U# wrapper compiles?",
                        "Add after Compile",
                        "Import Only"
                    )
                )
                    GeneratedBindingScenePlacement.Request(manifest.program.name);
                return destination;
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("udoncl Import Failed", exception.Message, "OK");
                return null;
            }
        }
    }
}
