using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;

namespace Lemolatoon.Udoncl.Editor
{
    internal sealed class AssetImportTransaction : IDisposable
    {
        private sealed class BackupRecord
        {
            public string AssetPath;
            public string AssetBackup;
            public string MetaBackup;
            public bool HadAsset;
            public bool HadMeta;
        }

        private readonly string _backupDirectory;
        private readonly List<BackupRecord> _records = new List<BackupRecord>();
        private readonly HashSet<string> _trackedPaths = new HashSet<string>(
            StringComparer.OrdinalIgnoreCase
        );
        private bool _completed;

        public AssetImportTransaction()
        {
            _backupDirectory = Path.Combine(
                Path.GetTempPath(),
                "udoncl-import-" + Guid.NewGuid().ToString("N")
            );
            Directory.CreateDirectory(_backupDirectory);
        }

        public void Track(string assetPath)
        {
            if (_completed)
                throw new InvalidOperationException("The import transaction is already complete.");
            if (!_trackedPaths.Add(assetPath))
                return;

            string absolutePath = ToAbsoluteAssetPath(assetPath);
            string metaPath = absolutePath + ".meta";
            BackupRecord record = new BackupRecord();
            record.AssetPath = assetPath;
            record.HadAsset = File.Exists(absolutePath);
            record.HadMeta = File.Exists(metaPath);
            record.AssetBackup = Path.Combine(
                _backupDirectory,
                _records.Count.ToString("D4") + ".asset"
            );
            record.MetaBackup = Path.Combine(
                _backupDirectory,
                _records.Count.ToString("D4") + ".meta"
            );
            if (record.HadAsset)
                File.Copy(absolutePath, record.AssetBackup, true);
            if (record.HadMeta)
                File.Copy(metaPath, record.MetaBackup, true);
            _records.Add(record);
        }

        public void CopyFile(string sourcePath, string destinationAssetPath)
        {
            Track(destinationAssetPath);
            string destinationPath = ToAbsoluteAssetPath(destinationAssetPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
            if (!PathsEqual(sourcePath, destinationPath))
                File.Copy(sourcePath, destinationPath, true);
            AssetDatabase.ImportAsset(
                destinationAssetPath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
        }

        public void WriteText(string destinationAssetPath, string contents)
        {
            Track(destinationAssetPath);
            string destinationPath = ToAbsoluteAssetPath(destinationAssetPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
            File.WriteAllText(destinationPath, contents, new UTF8Encoding(false));
            AssetDatabase.ImportAsset(
                destinationAssetPath,
                ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
            );
        }

        public void Commit()
        {
            AssetDatabase.SaveAssets();
            _completed = true;
            TryDeleteBackupDirectory();
        }

        public void Rollback()
        {
            if (_completed)
                return;

            for (int i = _records.Count - 1; i >= 0; i--)
            {
                BackupRecord record = _records[i];
                string absolutePath = ToAbsoluteAssetPath(record.AssetPath);
                string metaPath = absolutePath + ".meta";

                if (record.HadAsset)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(absolutePath));
                    File.Copy(record.AssetBackup, absolutePath, true);
                }
                else
                {
                    AssetDatabase.DeleteAsset(record.AssetPath);
                    if (File.Exists(absolutePath))
                        File.Delete(absolutePath);
                }

                if (record.HadMeta)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(metaPath));
                    File.Copy(record.MetaBackup, metaPath, true);
                }
                else if (File.Exists(metaPath))
                {
                    File.Delete(metaPath);
                }

                if (record.HadAsset)
                {
                    AssetDatabase.ImportAsset(
                        record.AssetPath,
                        ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate
                    );
                }
            }

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            AssetDatabase.SaveAssets();
            _completed = true;
            TryDeleteBackupDirectory();
        }

        public void Dispose()
        {
            if (!_completed)
                Rollback();
        }

        private static string ToAbsoluteAssetPath(string assetPath)
        {
            if (
                string.IsNullOrWhiteSpace(assetPath)
                || !assetPath.StartsWith("Assets/", StringComparison.Ordinal)
            )
                throw new InvalidDataException(
                    "Generated assets must be written under Assets/: " + assetPath
                );
            string absolutePath = Path.GetFullPath(
                assetPath.Replace('/', Path.DirectorySeparatorChar)
            );
            string assetsRoot = Path.GetFullPath("Assets") + Path.DirectorySeparatorChar;
            if (!absolutePath.StartsWith(assetsRoot, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Generated asset path resolves outside Assets/: " + assetPath
                );
            return absolutePath;
        }

        private static bool PathsEqual(string left, string right)
        {
            StringComparison comparison =
                Path.DirectorySeparatorChar == '\\'
                    ? StringComparison.OrdinalIgnoreCase
                    : StringComparison.Ordinal;
            return string.Equals(Path.GetFullPath(left), Path.GetFullPath(right), comparison);
        }

        private void TryDeleteBackupDirectory()
        {
            try
            {
                if (Directory.Exists(_backupDirectory))
                    Directory.Delete(_backupDirectory, true);
            }
            catch (IOException)
            {
                // A retained backup is harmless; the asset transaction has already completed.
            }
            catch (UnauthorizedAccessException)
            {
                // A retained backup is harmless; the asset transaction has already completed.
            }
        }
    }
}
