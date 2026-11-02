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
                        programAsset,
                        manifest,
                        programRoot.Replace('\\', '/')
                    );
                    transaction.Commit();
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
                if (manifest.device != null && manifest.device.buffers != null)
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

                UdonSharpBehaviour wrapper = root.AddUdonSharpComponent(wrapperType);
                SerializedObject serializedWrapper = new SerializedObject(wrapper);
                SerializedProperty programProperty = serializedWrapper.FindProperty("program");
                if (programProperty == null)
                    throw new InvalidDataException(
                        "Generated wrapper does not contain its program binding."
                    );
                programProperty.objectReferenceValue = program;
                serializedWrapper.ApplyModifiedPropertiesWithoutUndo();
                UdonSharpEditorUtility.CopyProxyToUdon(wrapper, ProxySerializationPolicy.All);

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
}
