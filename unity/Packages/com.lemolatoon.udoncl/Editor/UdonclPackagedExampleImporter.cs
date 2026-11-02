using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using UnityEditor;

namespace Lemolatoon.Udoncl.Editor
{
    internal static class UdonclPackagedExampleImporter
    {
        private const string ArenaArchive = "onnx-arena-add.zip.bytes";
        private const string HelloArchive = "hello-dispatch.zip.bytes";
        private const string SaxpyArchive = "saxpy.zip.bytes";
        private const string TinyArchive = "onnx-tiny-llama.zip.bytes";

        [MenuItem("Tools/udoncl/Examples/Import C and OpenCL Hello Dispatch")]
        private static void ImportHelloDispatch()
        {
            Import(HelloArchive);
        }

        [MenuItem("Tools/udoncl/Examples/Import C and OpenCL SAXPY")]
        private static void ImportSaxpy()
        {
            Import(SaxpyArchive);
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

        internal static string Extract(string archiveName)
        {
            if (
                archiveName != ArenaArchive
                && archiveName != HelloArchive
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
    }
}
