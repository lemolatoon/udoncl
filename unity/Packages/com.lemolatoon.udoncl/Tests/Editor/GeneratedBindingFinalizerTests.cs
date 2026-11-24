using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class GeneratedBindingFinalizerTests
    {
        [Test]
        public void FinalizerWaitsUntilGeneratedWrapperTypeIsCompiled()
        {
            const string root = "Assets/udoncl-generated/__binding_type_pending";
            AssetDatabase.DeleteAsset(root);
            Directory.CreateDirectory(root);
            File.WriteAllText(
                root + "/manifest.json",
                "{"
                    + "\"artifact_stage\":\"unity-udoncl-library\","
                    + "\"physical_export_abi\":\"udonsharp-worlds-sdk-3.10.5\","
                    + "\"program\":{\"name\":\"type_pending\","
                    + "\"binding_class\":\"UdonclTypePendingBinding\"}"
                    + "}"
            );
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            try
            {
                GeneratedBindingFinalizer.FinalizeAvailableBindings();
                Assert.IsNull(
                    AssetDatabase.LoadAssetAtPath<GameObject>(
                        root + "/Prefabs/UdonclTypePendingBinding.prefab"
                    )
                );
            }
            finally
            {
                AssetDatabase.DeleteAsset(root);
                if (Directory.Exists(root))
                    Directory.Delete(root, true);
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
        }

        [Test]
        public void PendingBindingScenePlacementIsScopedAndConsumedOnce()
        {
            GeneratedBindingScenePlacement.ClearRequest();
            try
            {
                GeneratedBindingScenePlacement.Request("hello_dispatch");
                Assert.IsFalse(GeneratedBindingScenePlacement.ConsumeRequest("saxpy_demo"));
                Assert.IsTrue(GeneratedBindingScenePlacement.ConsumeRequest("hello_dispatch"));
                Assert.IsFalse(GeneratedBindingScenePlacement.ConsumeRequest("hello_dispatch"));
            }
            finally
            {
                GeneratedBindingScenePlacement.ClearRequest();
            }
        }
    }
}
