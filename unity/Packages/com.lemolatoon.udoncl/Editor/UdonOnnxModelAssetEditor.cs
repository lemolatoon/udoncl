using System;
using Lemolatoon.Udoncl;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor
{
    public static class UdonOnnxModelEditorUtility
    {
        public static GameObject InstantiateRuntime(UdonOnnxModelAsset modelAsset)
        {
            if (modelAsset == null)
                throw new ArgumentNullException(nameof(modelAsset));
            if (modelAsset.RuntimePrefab == null)
                throw new InvalidOperationException("The ONNX model has no runtime prefab.");

            GameObject instance =
                PrefabUtility.InstantiatePrefab(modelAsset.RuntimePrefab) as GameObject;
            if (instance == null)
                throw new InvalidOperationException(
                    "Unity could not instantiate the runtime prefab."
                );
            Undo.RegisterCreatedObjectUndo(instance, "Add Udon ONNX Runtime");
            instance.name = GameObjectUtility.GetUniqueNameForSibling(
                null,
                modelAsset.RuntimePrefab.name
            );
            Selection.activeGameObject = instance;
            EditorGUIUtility.PingObject(instance);
            EditorSceneManager.MarkSceneDirty(instance.scene);
            return instance;
        }
    }

    [CustomEditor(typeof(UdonOnnxModelAsset))]
    internal sealed class UdonOnnxModelAssetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            UdonOnnxModelAsset modelAsset = (UdonOnnxModelAsset)target;
            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(modelAsset.RuntimePrefab == null))
                if (GUILayout.Button("Add Runtime to Active Scene"))
                    UdonOnnxModelEditorUtility.InstantiateRuntime(modelAsset);
            if (modelAsset.RuntimePrefab == null)
                EditorGUILayout.HelpBox(
                    "Reimport the artifact to generate its runtime prefab.",
                    MessageType.Warning
                );
        }
    }
}
