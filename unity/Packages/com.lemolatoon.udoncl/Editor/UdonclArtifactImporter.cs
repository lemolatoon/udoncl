using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using VRC.Udon;
using VRC.Udon.Common;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;
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

                    transaction.CopyFile(artifacts.ManifestAssetPath, programRoot + "/manifest.json");
                    string assemblyAssetPath = programRoot + "/" + Path.GetFileName(artifacts.AssemblyAssetPath);
                    transaction.CopyFile(artifacts.AssemblyAssetPath, assemblyAssetPath);

                    UdonAssemblyProgramAsset programAsset =
                        AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(assemblyAssetPath);
                    if (programAsset == null)
                        throw new InvalidDataException("Worlds SDK did not import Udon Assembly: " + assemblyAssetPath);
                    programAsset.RefreshProgram();
                    if (!string.IsNullOrEmpty(programAsset.AssemblyError) ||
                        programAsset.SerializedProgramAsset.RetrieveProgram() == null)
                        throw new InvalidDataException("Worlds SDK rejected Udon Assembly: " +
                            (programAsset.AssemblyError ?? assemblyAssetPath));

                    string displayShaderPath = programRoot + "/Shaders/display.shader";
                    EnsureFolders(programRoot + "/Shaders");
                    transaction.CopyFile(artifacts.DisplayShaderAssetPath, displayShaderPath);

                    Dictionary<ValidatedKernel, string> shaderAssetPaths = new Dictionary<ValidatedKernel, string>();
                    foreach (ValidatedKernel kernel in artifacts.Kernels)
                    {
                        string shaderAssetPath = programRoot + "/Shaders/" + kernel.RelativeShaderPath;
                        string shaderDirectory = Path.GetDirectoryName(shaderAssetPath).Replace('\\', '/');
                        EnsureFolders(shaderDirectory);
                        transaction.CopyFile(kernel.AbsoluteShaderPath, shaderAssetPath);
                        shaderAssetPaths.Add(kernel, shaderAssetPath);
                    }

                    Dictionary<int, Material> kernelMaterials = new Dictionary<int, Material>();
                    foreach (KeyValuePair<ValidatedKernel, string> shaderEntry in shaderAssetPaths)
                    {
                        Shader shader = AssetDatabase.LoadAssetAtPath<Shader>(shaderEntry.Value);
                        if (shader == null)
                            throw new InvalidDataException("Unity could not import generated shader: " + shaderEntry.Value);
                        kernelMaterials.Add(shaderEntry.Key.Id,
                            CreateOrUpdateMaterial(transaction, programRoot, shaderEntry.Key, shader));
                    }

                    Shader displayShader = AssetDatabase.LoadAssetAtPath<Shader>(displayShaderPath);
                    if (displayShader == null)
                        throw new InvalidDataException("Unity could not import display shader: " + displayShaderPath);
                    Material displayMaterial = CreateOrUpdateDisplayMaterial(transaction, programRoot, displayShader);
                    Texture2D initialState = CreateOrUpdateParticleTexture(transaction, programRoot);
                    RenderTexture stateA = CreateOrUpdateRenderTexture(transaction, programRoot, "lj_state_a.asset");
                    RenderTexture stateB = CreateOrUpdateRenderTexture(transaction, programRoot, "lj_state_b.asset");
                    if (!kernelMaterials.TryGetValue(10, out Material driftMaterial) ||
                        !kernelMaterials.TryGetValue(11, out Material kickMaterial))
                        throw new InvalidDataException("LJ demo requires kernel ids 10 (drift) and 11 (kick).");

                    CreateOrUpdateDemoPrefab(transaction, programRoot, programAsset, initialState,
                        stateA, stateB, driftMaterial, kickMaterial, displayMaterial);

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

        private static Material CreateOrUpdateMaterial(AssetImportTransaction transaction, string programRoot,
            ValidatedKernel kernel, Shader shader)
        {
            string materialDirectory = programRoot + "/Materials";
            EnsureFolders(materialDirectory);
            string materialPath = materialDirectory + "/kernel_" + kernel.Id + "_" + kernel.Name + ".mat";
            transaction.Track(materialPath);

            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(materialPath) != null)
                    throw new InvalidDataException("An incompatible asset already occupies " + materialPath);
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

        private static Material CreateOrUpdateDisplayMaterial(AssetImportTransaction transaction,
            string programRoot, Shader shader)
        {
            string materialDirectory = programRoot + "/Materials";
            EnsureFolders(materialDirectory);
            string materialPath = materialDirectory + "/lj_display.mat";
            transaction.Track(materialPath);
            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(materialPath) != null)
                    throw new InvalidDataException("An incompatible asset already occupies " + materialPath);
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

        private static Texture2D CreateOrUpdateParticleTexture(AssetImportTransaction transaction, string programRoot)
        {
            string textureDirectory = programRoot + "/Textures";
            EnsureFolders(textureDirectory);
            string texturePath = textureDirectory + "/lj_particles_initial.asset";
            transaction.Track(texturePath);

            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
            if (texture == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(texturePath) != null)
                    throw new InvalidDataException("An incompatible asset already occupies " + texturePath);
                texture = new Texture2D(ParticleTextureEncoder.TextureWidth, ParticleTextureEncoder.TextureHeight,
                    TextureFormat.RGBA32, false, true);
                AssetDatabase.CreateAsset(texture, texturePath);
            }
            else if (!texture.Reinitialize(ParticleTextureEncoder.TextureWidth, ParticleTextureEncoder.TextureHeight,
                TextureFormat.RGBA32, false))
            {
                throw new InvalidOperationException("Could not resize existing particle texture " + texturePath);
            }

            texture.name = Path.GetFileNameWithoutExtension(texturePath);
            texture.filterMode = FilterMode.Point;
            texture.wrapMode = TextureWrapMode.Clamp;
            texture.anisoLevel = 0;
            texture.LoadRawTextureData(ParticleTextureEncoder.Encode(ParticleTextureEncoder.CreateLennardJonesInitialState()));
            texture.Apply(false, false);
            EditorUtility.SetDirty(texture);
            return texture;
        }

        private static RenderTexture CreateOrUpdateRenderTexture(AssetImportTransaction transaction,
            string programRoot, string fileName)
        {
            string textureDirectory = programRoot + "/RenderTextures";
            EnsureFolders(textureDirectory);
            string renderTexturePath = textureDirectory + "/" + fileName;
            transaction.Track(renderTexturePath);

            RenderTexture renderTexture = AssetDatabase.LoadAssetAtPath<RenderTexture>(renderTexturePath);
            if (renderTexture == null)
            {
                if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(renderTexturePath) != null)
                    throw new InvalidDataException("An incompatible asset already occupies " + renderTexturePath);
                renderTexture = new RenderTexture(ParticleTextureEncoder.TextureWidth, ParticleTextureEncoder.TextureHeight,
                    0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
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
                throw new InvalidOperationException("Generated RenderTexture unexpectedly uses sRGB: " + renderTexturePath);
            if (!renderTexture.Create() && !renderTexture.IsCreated())
                throw new InvalidOperationException("Could not create generated RenderTexture: " + renderTexturePath);
            EditorUtility.SetDirty(renderTexture);
            return renderTexture;
        }

        private static void CreateOrUpdateDemoPrefab(AssetImportTransaction transaction, string programRoot,
            UdonAssemblyProgramAsset programAsset, Texture initialState, RenderTexture stateA,
            RenderTexture stateB, Material driftMaterial, Material kickMaterial, Material displayMaterial)
        {
            string prefabDirectory = programRoot + "/Prefabs";
            EnsureFolders(prefabDirectory);
            string prefabPath = prefabDirectory + "/LennardJonesDemo.prefab";
            transaction.Track(prefabPath);

            GameObject root = new GameObject("udoncl Lennard-Jones ice");
            try
            {
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.size = new Vector3(8.0f, 6.0f, 0.25f);
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
                SetPublicVariable(behaviour, "frame", 0);

                SerializedObject serializedBehaviour = new SerializedObject(behaviour);
                serializedBehaviour.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedBehaviour.ApplyModifiedPropertiesWithoutUndo();

                GameObject display = GameObject.CreatePrimitive(PrimitiveType.Quad);
                display.name = "LJ particle display";
                display.transform.SetParent(root.transform, false);
                display.transform.localScale = new Vector3(8.0f, 6.0f, 1.0f);
                Collider displayCollider = display.GetComponent<Collider>();
                if (displayCollider != null)
                    UnityEngine.Object.DestroyImmediate(displayCollider);
                display.GetComponent<MeshRenderer>().sharedMaterial = displayMaterial;

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException("Could not create LJ demo prefab: " + prefabPath);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static void SetPublicVariable<T>(UdonBehaviour behaviour, string symbol, T value)
        {
            if (!behaviour.publicVariables.TrySetVariableValue(symbol, value))
            {
                behaviour.publicVariables.RemoveVariable(symbol);
                if (!behaviour.publicVariables.TryAddVariable(new UdonVariable<T>(symbol, value)))
                    throw new InvalidOperationException("Could not bind Udon public variable: " + symbol);
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
                    throw new DirectoryNotFoundException("Unity asset folder does not exist: " + parent);
                AssetDatabase.CreateFolder(parent, name);
            }
        }
    }

    internal static class UdonclImportMenu
    {
        [MenuItem("Tools/udoncl/Import Artifacts...")]
        private static void ImportArtifacts()
        {
            string directory = EditorUtility.OpenFolderPanel("Select udoncl artifact directory", "", "");
            if (string.IsNullOrEmpty(directory))
                return;

            try
            {
                string destination = UdonclArtifactImporter.Import(directory);
                EditorUtility.DisplayDialog("udoncl Import", "Imported assets to " + destination, "OK");
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("udoncl Import Failed", exception.Message, "OK");
            }
        }
    }
}
