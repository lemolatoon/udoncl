using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor
{
    public sealed class ValidatedArtifact
    {
        internal ValidatedArtifact(
            string root,
            string manifestAssetPath,
            string assemblyAssetPath,
            string displayShaderAssetPath,
            string programName,
            IList<ValidatedKernel> kernels
        )
        {
            RootDirectory = root;
            ManifestAssetPath = manifestAssetPath;
            AssemblyAssetPath = assemblyAssetPath;
            DisplayShaderAssetPath = displayShaderAssetPath;
            ProgramName = programName;
            Kernels = kernels;
        }

        public string RootDirectory { get; private set; }
        public string ManifestAssetPath { get; private set; }
        public string AssemblyAssetPath { get; private set; }
        public string DisplayShaderAssetPath { get; private set; }
        public string ProgramName { get; private set; }
        public IList<ValidatedKernel> Kernels { get; private set; }
    }

    public sealed class ValidatedKernel
    {
        internal ValidatedKernel(
            int id,
            string name,
            string relativeShaderPath,
            string absoluteShaderPath
        )
        {
            Id = id;
            Name = name;
            RelativeShaderPath = relativeShaderPath;
            AbsoluteShaderPath = absoluteShaderPath;
        }

        public int Id { get; private set; }
        public string Name { get; private set; }
        public string RelativeShaderPath { get; private set; }
        public string AbsoluteShaderPath { get; private set; }
    }

    public static class ArtifactManifestValidator
    {
        private static readonly Regex SafeIdentifier = new Regex(
            @"^[A-Za-z0-9_-]+$",
            RegexOptions.Compiled
        );

        public static ValidatedArtifact Validate(string artifactDirectory)
        {
            if (string.IsNullOrWhiteSpace(artifactDirectory))
                throw new InvalidDataException("Artifact directory is empty.");

            string root = Path.GetFullPath(artifactDirectory.Trim());
            if (!Directory.Exists(root))
                throw new DirectoryNotFoundException("Artifact directory does not exist: " + root);

            string manifestPath = Path.Combine(root, "manifest.json");
            if (!File.Exists(manifestPath))
                throw new InvalidDataException("Artifact directory must contain manifest.json.");

            UdonclManifest manifest;
            try
            {
                manifest = JsonUtility.FromJson<UdonclManifest>(ReadUtf8(manifestPath));
            }
            catch (Exception ex)
            {
                throw new InvalidDataException(
                    "manifest.json is not valid JSON: " + ex.Message,
                    ex
                );
            }

            if (manifest == null || manifest.schema_version != 1)
                throw new InvalidDataException(
                    "Unsupported or missing manifest schema_version; expected 1."
                );
            if (string.IsNullOrWhiteSpace(manifest.compiler_version))
                throw new InvalidDataException("manifest.compiler_version is required.");
            if (manifest.runtime_abi_version != 1)
                throw new InvalidDataException(
                    "Unsupported runtime_abi_version; this importer supports ABI 1."
                );

            ValidateSdk(manifest.sdk_compatibility);
            ValidateTarget(manifest.target);

            if (manifest.program == null || !IsSafeIdentifier(manifest.program.name))
                throw new InvalidDataException(
                    "manifest.program.name must contain only letters, digits, '_' or '-'."
                );

            string assemblyRelativePath = ValidateRelativeFilePath(
                manifest.program.assembly,
                ".uasm"
            );
            string assemblyPath = ResolveExistingFile(root, assemblyRelativePath);
            if (new FileInfo(assemblyPath).Length == 0)
                throw new InvalidDataException(
                    "Udon Assembly artifact is empty: " + assemblyRelativePath
                );
            ReadUtf8(assemblyPath);

            string displayRelativePath = ValidateRelativeFilePath(
                manifest.program.display_shader,
                ".shader"
            );
            string displayShaderPath = ResolveExistingFile(root, displayRelativePath);
            if (new FileInfo(displayShaderPath).Length == 0)
                throw new InvalidDataException(
                    "Display shader artifact is empty: " + displayRelativePath
                );
            ReadUtf8(displayShaderPath);

            List<ValidatedKernel> kernels = new List<ValidatedKernel>();
            HashSet<int> kernelIds = new HashSet<int>();
            HashSet<string> shaderPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (manifest.device != null && manifest.device.kernels != null)
            {
                foreach (KernelInfo kernel in manifest.device.kernels)
                {
                    if (kernel == null || kernel.id < 0 || !IsSafeIdentifier(kernel.name))
                        throw new InvalidDataException(
                            "Each device kernel needs a non-negative id and safe name."
                        );
                    if (!kernelIds.Add(kernel.id))
                        throw new InvalidDataException("Duplicate device kernel id: " + kernel.id);

                    string relativeShader = ValidateRelativeFilePath(kernel.shader, ".shader");
                    if (!shaderPaths.Add(relativeShader))
                        throw new InvalidDataException(
                            "A shader path is referenced more than once: " + relativeShader
                        );

                    string absoluteShader = ResolveExistingFile(root, relativeShader);
                    if (new FileInfo(absoluteShader).Length == 0)
                        throw new InvalidDataException(
                            "Shader artifact is empty: " + relativeShader
                        );
                    ReadUtf8(absoluteShader);
                    kernels.Add(
                        new ValidatedKernel(kernel.id, kernel.name, relativeShader, absoluteShader)
                    );
                }
            }

            return new ValidatedArtifact(
                root,
                manifestPath,
                assemblyPath,
                displayShaderPath,
                manifest.program.name,
                kernels
            );
        }

        private static void ValidateSdk(SdkCompatibility compatibility)
        {
            if (compatibility == null || string.IsNullOrWhiteSpace(compatibility.sdk_version))
                throw new InvalidDataException(
                    "manifest.sdk_compatibility.sdk_version is required."
                );

            string version = compatibility.sdk_version.Trim();
            Match match = Regex.Match(
                version,
                @"^(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][0-9A-Za-z.-]+)?$"
            );
            if (!match.Success || match.Groups[1].Value != "3")
                throw new InvalidDataException(
                    "This package supports VRChat Worlds SDK 3.x only; manifest sdk_version was '"
                        + version
                        + "'."
                );
        }

        private static void ValidateTarget(TargetInfo target)
        {
            if (
                target == null
                || target.host_pointer_bits != 32
                || target.device_pointer_bits != 32
                || !string.Equals(target.endianness, "little", StringComparison.OrdinalIgnoreCase)
                || !string.Equals(target.platform, "VRChat-PC", StringComparison.Ordinal)
            )
            {
                throw new InvalidDataException(
                    "Unsupported target. Expected 32-bit little-endian VRChat-PC ABI."
                );
            }
        }

        private static string ValidateRelativeFilePath(string path, string requiredExtension)
        {
            if (
                string.IsNullOrWhiteSpace(path)
                || Path.IsPathRooted(path)
                || path.IndexOf('\\') >= 0
                || path.IndexOf(':') >= 0
            )
                throw new InvalidDataException(
                    "Artifact paths must be relative paths using '/': " + (path ?? "<missing>")
                );

            string[] segments = path.Split('/');
            foreach (string segment in segments)
            {
                if (segment.Length == 0 || segment == "." || segment == "..")
                    throw new InvalidDataException(
                        "Artifact path contains an unsafe segment: " + path
                    );
            }

            if (
                !string.Equals(
                    Path.GetExtension(path),
                    requiredExtension,
                    StringComparison.OrdinalIgnoreCase
                )
            )
                throw new InvalidDataException(
                    "Artifact path must end in " + requiredExtension + ": " + path
                );

            return string.Join("/", segments);
        }

        private static string ResolveExistingFile(string root, string relativePath)
        {
            string localPath = relativePath.Replace('/', Path.DirectorySeparatorChar);
            string fullPath = Path.GetFullPath(Path.Combine(root, localPath));
            string rootPrefix =
                root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            StringComparison comparison =
                Application.platform == RuntimePlatform.WindowsEditor
                    ? StringComparison.OrdinalIgnoreCase
                    : StringComparison.Ordinal;
            if (!fullPath.StartsWith(rootPrefix, comparison))
                throw new InvalidDataException(
                    "Artifact path resolves outside its directory: " + relativePath
                );
            if (!File.Exists(fullPath))
                throw new FileNotFoundException(
                    "Manifest references a missing artifact: " + relativePath,
                    fullPath
                );
            return fullPath;
        }

        private static string ReadUtf8(string path)
        {
            return File.ReadAllText(path, new UTF8Encoding(false, true));
        }

        private static bool IsSafeIdentifier(string value)
        {
            return !string.IsNullOrWhiteSpace(value) && SafeIdentifier.IsMatch(value);
        }
    }
}
