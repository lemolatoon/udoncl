using System;
using System.IO;
using System.Linq;
using UdonSharp;
using UdonSharpEditor;
using UnityEditor;
using UnityEngine;
using VRC.Udon;
using VRC.Udon.Common;
using VRC.Udon.Editor.ProgramSources;
using SyncType = VRC.SDKBase.Networking.SyncType;

namespace Lemolatoon.Udoncl.Editor
{
    [InitializeOnLoad]
    public static class GeneratedBindingFinalizer
    {
        private const string GeneratedRoot = "Assets/udoncl-generated";
        private const string GeneratedNamespace = "Lemolatoon.Udoncl.Generated";
        private static bool _scheduled;

        static GeneratedBindingFinalizer()
        {
            Schedule();
        }

        internal static void Schedule()
        {
            if (_scheduled)
                return;
            _scheduled = true;
            EditorApplication.delayCall += RunScheduled;
        }

        public static int FinalizeAvailableBindings()
        {
            if (!Directory.Exists(GeneratedRoot))
                return 0;

            int finalized = 0;
            foreach (string programRoot in Directory.GetDirectories(GeneratedRoot))
            {
                string manifestPath = Path.Combine(programRoot, "manifest.json");
                if (!File.Exists(manifestPath))
                    continue;
                UdonclManifest manifest = JsonUtility.FromJson<UdonclManifest>(
                    File.ReadAllText(manifestPath)
                );
                if (
                    manifest == null
                    || manifest.artifact_stage != "unity-udoncl-library"
                    || manifest.physical_export_abi != "udonsharp-worlds-sdk-3.10.5"
                    || manifest.program == null
                )
                    continue;

                string className = VRCWrapperGenerator.BindingClassName(manifest.program.name);
                string prefabDirectory = programRoot.Replace('\\', '/') + "/Prefabs";
                string prefabPath = prefabDirectory + "/" + className + ".prefab";

                Type wrapperType = TypeCache
                    .GetTypesDerivedFrom<UdonSharpBehaviour>()
                    .SingleOrDefault(type => type.FullName == GeneratedNamespace + "." + className);
                if (wrapperType == null)
                    continue;

                PackagedExampleUsage usage = PackagedExampleUsageGenerator.ForProgram(
                    manifest.program.name
                );
                bool expectsExample =
                    usage != null
                    && File.Exists(Path.Combine(programRoot, "Examples", usage.ClassName + ".cs"));
                Type exampleType =
                    usage == null
                        ? null
                        : TypeCache
                            .GetTypesDerivedFrom<UdonSharpBehaviour>()
                            .SingleOrDefault(type =>
                                type.FullName == GeneratedNamespace + "." + usage.ClassName
                            );
                if (expectsExample && exampleType == null)
                    continue;
                if (exampleType != null)
                {
                    UdonSharpProgramAsset exampleProgram =
                        UdonSharpEditorUtility.GetUdonSharpProgramAsset(exampleType);
                    if (
                        exampleProgram == null
                        || exampleProgram.SerializedProgramAsset == null
                        || exampleProgram.CompiledVersion != UdonSharpProgramVersion.CurrentVersion
                    )
                        continue;
                }
                UdonSharpProgramAsset wrapperProgram =
                    UdonSharpEditorUtility.GetUdonSharpProgramAsset(wrapperType);
                if (
                    wrapperProgram == null
                    || wrapperProgram.SerializedProgramAsset == null
                    || wrapperProgram.CompiledVersion != UdonSharpProgramVersion.CurrentVersion
                )
                    continue;

                string programPath =
                    programRoot.Replace('\\', '/') + "/Programs/host_program.asset";
                UdonAssemblyProgramAsset programAsset =
                    AssetDatabase.LoadAssetAtPath<UdonAssemblyProgramAsset>(programPath);
                if (programAsset == null)
                    throw new InvalidDataException(
                        "Generated Udon program asset is missing: " + programPath
                    );

                EnsureFolders(prefabDirectory);
                using (AssetImportTransaction transaction = new AssetImportTransaction())
                {
                    CreateBindingPrefab(
                        transaction,
                        prefabPath,
                        className,
                        wrapperType,
                        exampleType,
                        programAsset,
                        manifest,
                        programRoot.Replace('\\', '/')
                    );
                    transaction.Commit();
                }
                if (GeneratedBindingScenePlacement.ConsumeRequest(manifest.program.name))
                {
                    GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
                    GeneratedBindingScenePlacement.Instantiate(prefab);
                }
                finalized++;
            }
            return finalized;
        }

        private static void RunScheduled()
        {
            _scheduled = false;
            if (EditorApplication.isCompiling || EditorApplication.isUpdating)
            {
                Schedule();
                return;
            }
            FinalizeAvailableBindings();
        }

        private static void CreateBindingPrefab(
            AssetImportTransaction transaction,
            string prefabPath,
            string className,
            Type wrapperType,
            Type exampleType,
            UdonAssemblyProgramAsset programAsset,
            UdonclManifest manifest,
            string programRoot
        )
        {
            transaction.Track(prefabPath);
            GameObject root = new GameObject(className);
            try
            {
                UdonBehaviour program = root.AddComponent<UdonBehaviour>();
                program.programSource = programAsset;
                program.SyncMethod = SyncType.None;
                SerializedObject serializedProgram = new SerializedObject(program);
                serializedProgram.FindProperty("serializedProgramAsset").objectReferenceValue =
                    programAsset.SerializedProgramAsset;
                serializedProgram.ApplyModifiedPropertiesWithoutUndo();
                if (
                    manifest.device != null
                    && manifest.device.buffers != null
                    && manifest.device.buffers.Length != 0
                )
                    foreach (DeviceBufferInfo buffer in manifest.device.buffers)
                    {
                        string texturePath =
                            programRoot
                            + "/RenderTextures/"
                            + UdonclArtifactImporter.DeviceBufferFileName(buffer.id, buffer.name);
                        RenderTexture texture = AssetDatabase.LoadAssetAtPath<RenderTexture>(
                            texturePath
                        );
                        if (texture == null)
                            throw new InvalidDataException(
                                "Generated Device buffer is missing: " + texturePath
                            );
                        SetPublicVariable(program, buffer.heap_slot, texture);
                    }
                if (
                    manifest.program != null
                    && !string.IsNullOrEmpty(manifest.program.buffer_copy_shader)
                    && manifest.program.buffer_copy_program_slot
                )
                {
                    string materialPath = programRoot + "/Materials/buffer-copy.mat";
                    Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
                    if (material == null)
                        throw new InvalidDataException(
                            "Generated buffer copy material is missing: " + materialPath
                        );
                    SetPublicVariable(program, "__udoncl_buffer_copy_material", material);
                }
                if (manifest.dispatch_sites != null)
                    foreach (DispatchSiteInfo dispatch in manifest.dispatch_sites)
                    {
                        KernelInfo kernel = Array.Find(
                            manifest.device?.kernels,
                            candidate => candidate.id == dispatch.kernel_id
                        );
                        if (kernel == null)
                            throw new InvalidDataException(
                                "Generated dispatch references a missing kernel material."
                            );
                        string materialPath =
                            programRoot
                            + "/Materials/kernel_"
                            + kernel.id
                            + "_"
                            + kernel.name
                            + ".mat";
                        Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
                        if (material == null)
                            throw new InvalidDataException(
                                "Generated kernel material is missing: " + materialPath
                            );
                        SetPublicVariable(
                            program,
                            "__udoncl_kernel_material_" + kernel.id,
                            material
                        );
                    }

                UdonSharpBehaviour wrapper = root.AddUdonSharpComponent(wrapperType);
                SerializedObject serializedWrapper = new SerializedObject(wrapper);
                SerializedProperty programProperty = serializedWrapper.FindProperty("program");
                if (programProperty == null)
                    throw new InvalidDataException(
                        "Generated wrapper does not contain its program binding."
                    );
                programProperty.objectReferenceValue = program;
                if (
                    manifest.device != null
                    && manifest.device.buffers != null
                    && manifest.device.buffers.Length != 0
                )
                {
                    DeviceBufferInfo[] buffers = manifest.device.buffers;
                    SerializedProperty ids = serializedWrapper.FindProperty("deviceBufferIds");
                    SerializedProperty logicalBytes = serializedWrapper.FindProperty(
                        "deviceBufferLogicalBytes"
                    );
                    SerializedProperty textures = serializedWrapper.FindProperty("deviceBuffers");
                    SerializedProperty copyMaterial = serializedWrapper.FindProperty(
                        "bufferCopyMaterial"
                    );
                    if (
                        ids == null
                        || logicalBytes == null
                        || textures == null
                        || copyMaterial == null
                    )
                        throw new InvalidDataException(
                            "Generated wrapper does not contain its Device buffer transfer binding."
                        );
                    ids.arraySize = buffers.Length;
                    logicalBytes.arraySize = buffers.Length;
                    textures.arraySize = buffers.Length;
                    for (int index = 0; index < buffers.Length; ++index)
                    {
                        DeviceBufferInfo buffer = buffers[index];
                        ids.GetArrayElementAtIndex(index).intValue = buffer.id;
                        logicalBytes.GetArrayElementAtIndex(index).intValue = buffer.logical_bytes;
                        string texturePath =
                            programRoot
                            + "/RenderTextures/"
                            + UdonclArtifactImporter.DeviceBufferFileName(buffer.id, buffer.name);
                        textures.GetArrayElementAtIndex(index).objectReferenceValue =
                            AssetDatabase.LoadAssetAtPath<RenderTexture>(texturePath);
                    }
                    string materialPath = programRoot + "/Materials/buffer-copy.mat";
                    copyMaterial.objectReferenceValue = AssetDatabase.LoadAssetAtPath<Material>(
                        materialPath
                    );
                    if (copyMaterial.objectReferenceValue == null)
                        throw new InvalidDataException(
                            "Generated Device buffer transfer material is missing: " + materialPath
                        );
                }
                serializedWrapper.ApplyModifiedPropertiesWithoutUndo();
                UdonSharpEditorUtility.CopyProxyToUdon(wrapper, ProxySerializationPolicy.All);

                if (exampleType != null)
                {
                    UdonSharpBehaviour example = root.AddUdonSharpComponent(exampleType);
                    SerializedObject serializedExample = new SerializedObject(example);
                    SerializedProperty binding = serializedExample.FindProperty("binding");
                    if (binding == null)
                        throw new InvalidDataException(
                            "Packaged example does not contain its generated binding reference."
                        );
                    binding.objectReferenceValue = wrapper;
                    serializedExample.ApplyModifiedPropertiesWithoutUndo();
                    UdonSharpEditorUtility.CopyProxyToUdon(example, ProxySerializationPolicy.All);
                }

                PrefabUtility.SaveAsPrefabAsset(root, prefabPath, out bool success);
                if (!success || AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) == null)
                    throw new InvalidOperationException(
                        "Could not create generated UdonCL binding prefab: " + prefabPath
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
                        "Could not bind generated Udon variable: " + symbol
                    );
            }
        }

        private static void EnsureFolders(string path)
        {
            string[] parts = path.Split('/');
            string current = parts[0];
            for (int index = 1; index < parts.Length; ++index)
            {
                string next = current + "/" + parts[index];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[index]);
                current = next;
            }
        }
    }

    internal static class GeneratedBindingScenePlacement
    {
        private const string PendingProgramKey = "Lemolatoon.Udoncl.PendingBindingSceneProgram";

        internal static void Request(string programName)
        {
            if (string.IsNullOrEmpty(programName))
                throw new ArgumentException("Program name is empty.", nameof(programName));
            SessionState.SetString(PendingProgramKey, programName);
        }

        internal static bool ConsumeRequest(string programName)
        {
            if (
                string.IsNullOrEmpty(programName)
                || SessionState.GetString(PendingProgramKey, "") != programName
            )
                return false;
            SessionState.EraseString(PendingProgramKey);
            return true;
        }

        internal static void ClearRequest()
        {
            SessionState.EraseString(PendingProgramKey);
        }

        internal static GameObject Instantiate(GameObject prefab)
        {
            if (prefab == null)
                throw new ArgumentNullException(nameof(prefab));
            GameObject instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
            if (instance == null)
                throw new InvalidOperationException(
                    "Unity could not instantiate the generated UdonCL binding prefab."
                );
            Undo.RegisterCreatedObjectUndo(instance, "Add UdonCL Binding");
            Selection.activeGameObject = instance;
            return instance;
        }
    }
}
