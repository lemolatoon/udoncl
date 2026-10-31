using System;
using System.Collections.Generic;
using System.IO;
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
            displayMaterial.SetTexture("_Logits", stateB);
            EditorUtility.SetDirty(displayMaterial);
            CreateOrUpdateTinyLlamaPrefab(
                transaction,
                programRoot,
                programAsset,
                initialState,
                stateA,
                stateB,
                passes,
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
            texture.filterMode = FilterMode.Point;
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.useMipMap = false;
            texture.autoGenerateMips = false;
            texture.antiAliasing = 1;
            texture.anisoLevel = 0;
            if (texture.sRGB)
                throw new InvalidOperationException(
                    "Generated RenderTexture unexpectedly uses sRGB: " + texturePath
                );
            if (!texture.Create() && !texture.IsCreated())
                throw new InvalidOperationException(
                    "Could not create generated RenderTexture: " + texturePath
                );
            EditorUtility.SetDirty(texture);
            return texture;
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
            IReadOnlyList<Material> passes,
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
                UdonBehaviour behaviour = root.AddComponent<UdonBehaviour>();
                behaviour.programSource = programAsset;
                behaviour.interactText = "Run Tiny LLaMA";
                behaviour.proximity = 3.0f;
                behaviour.SyncMethod = SyncType.None;
                SetPublicVariable(behaviour, "initialState", initialState);
                SetPublicVariable(behaviour, "stateA", stateA);
                SetPublicVariable(behaviour, "stateB", stateB);
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
                SetPublicVariable(behaviour, "displayMaterial", displayMaterial);
                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Cube);
                display.name = "Tiny LLaMA logits and predicted token";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(4.0f, 2.0f, 0.12f);
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

            try
            {
                string destination = UdonclArtifactImporter.Import(directory);
                EditorUtility.DisplayDialog(
                    "udoncl Import",
                    "Imported assets to " + destination,
                    "OK"
                );
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("udoncl Import Failed", exception.Message, "OK");
            }
        }
    }
}
