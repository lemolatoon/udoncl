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
            string bufferCopyShaderAssetPath,
            string artifactStage,
            string programName,
            IList<ValidatedKernel> kernels,
            IList<ValidatedDeviceBuffer> deviceBuffers,
            IList<ValidatedResource> resources,
            IList<ValidatedOnnxStaticInput> onnxStaticInputs,
            IList<ValidatedOnnxInt4Call> onnxInt4Calls,
            IList<ValidatedOnnxIo> onnxInputs,
            IList<ValidatedOnnxIo> onnxOutputs
        )
        {
            RootDirectory = root;
            ManifestAssetPath = manifestAssetPath;
            AssemblyAssetPath = assemblyAssetPath;
            DisplayShaderAssetPath = displayShaderAssetPath;
            BufferCopyShaderAssetPath = bufferCopyShaderAssetPath;
            ArtifactStage = artifactStage;
            ProgramName = programName;
            Kernels = kernels;
            DeviceBuffers = deviceBuffers;
            Resources = resources;
            OnnxStaticInputs = onnxStaticInputs;
            OnnxInt4Calls = onnxInt4Calls;
            OnnxInputs = onnxInputs;
            OnnxOutputs = onnxOutputs;
        }

        public string RootDirectory { get; private set; }
        public string ManifestAssetPath { get; private set; }
        public string AssemblyAssetPath { get; private set; }
        public string DisplayShaderAssetPath { get; private set; }
        public string BufferCopyShaderAssetPath { get; private set; }
        public string ArtifactStage { get; private set; }
        public string ProgramName { get; private set; }
        public IList<ValidatedKernel> Kernels { get; private set; }
        public IList<ValidatedDeviceBuffer> DeviceBuffers { get; private set; }
        public IList<ValidatedResource> Resources { get; private set; }
        public IList<ValidatedOnnxStaticInput> OnnxStaticInputs { get; private set; }
        public IList<ValidatedOnnxInt4Call> OnnxInt4Calls { get; private set; }
        public IList<ValidatedOnnxIo> OnnxInputs { get; private set; }
        public IList<ValidatedOnnxIo> OnnxOutputs { get; private set; }
    }

    public sealed class ValidatedOnnxIo
    {
        internal ValidatedOnnxIo(int index, string name, string dataType, int words, int slot)
        {
            Index = index;
            Name = name;
            DataType = dataType;
            Words = words;
            Slot = slot;
        }

        public int Index { get; private set; }
        public string Name { get; private set; }
        public string DataType { get; private set; }
        public int Words { get; private set; }
        public int Slot { get; private set; }
    }

    public sealed class ValidatedOnnxInt4Atlas
    {
        internal ValidatedOnnxInt4Atlas(string name, string absolutePath, int width, int height)
        {
            Name = name;
            AbsolutePath = absolutePath;
            Width = width;
            Height = height;
        }

        public string Name { get; private set; }
        public string AbsolutePath { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
    }

    public sealed class ValidatedOnnxInt4Call
    {
        internal ValidatedOnnxInt4Call(
            int callIndex,
            string assemblyAbsolutePath,
            int resultWidth,
            int resultHeight,
            IList<ValidatedOnnxInt4Atlas> atlases
        )
        {
            CallIndex = callIndex;
            AssemblyAbsolutePath = assemblyAbsolutePath;
            ResultWidth = resultWidth;
            ResultHeight = resultHeight;
            Atlases = atlases;
        }

        public int CallIndex { get; private set; }
        public string AssemblyAbsolutePath { get; private set; }
        public int ResultWidth { get; private set; }
        public int ResultHeight { get; private set; }
        public IList<ValidatedOnnxInt4Atlas> Atlases { get; private set; }
    }

    public sealed class ValidatedOnnxStaticInput
    {
        internal ValidatedOnnxStaticInput(
            string name,
            string absolutePath,
            int words,
            int width,
            int height
        )
        {
            Name = name;
            AbsolutePath = absolutePath;
            Words = words;
            Width = width;
            Height = height;
        }

        public string Name { get; private set; }
        public string AbsolutePath { get; private set; }
        public int Words { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
    }

    public sealed class ValidatedResource
    {
        internal ValidatedResource(
            string name,
            string kind,
            string relativePath,
            string absolutePath
        )
        {
            Name = name;
            Kind = kind;
            RelativePath = relativePath;
            AbsolutePath = absolutePath;
        }

        public string Name { get; private set; }
        public string Kind { get; private set; }
        public string RelativePath { get; private set; }
        public string AbsolutePath { get; private set; }
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

    public sealed class ValidatedDeviceBuffer
    {
        internal ValidatedDeviceBuffer(
            int id,
            string name,
            string heapSlot,
            int logicalBytes,
            int width,
            int height,
            string elementType,
            string publicAccess
        )
        {
            Id = id;
            Name = name;
            HeapSlot = heapSlot;
            LogicalBytes = logicalBytes;
            Width = width;
            Height = height;
            ElementType = elementType;
            PublicAccess = publicAccess;
        }

        public int Id { get; private set; }
        public string Name { get; private set; }
        public string HeapSlot { get; private set; }
        public int LogicalBytes { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public string ElementType { get; private set; }
        public string PublicAccess { get; private set; }
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
            if (!manifest.ready_for_unity || !IsSafeIdentifier(manifest.artifact_stage))
                throw new InvalidDataException(
                    "Importer requires a ready_for_unity artifact with a safe artifact_stage."
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

            string bufferCopyShaderPath = null;
            if (!string.IsNullOrEmpty(manifest.program.buffer_copy_shader))
            {
                string bufferCopyRelativePath = ValidateRelativeFilePath(
                    manifest.program.buffer_copy_shader,
                    ".shader"
                );
                bufferCopyShaderPath = ResolveExistingFile(root, bufferCopyRelativePath);
                if (new FileInfo(bufferCopyShaderPath).Length == 0)
                    throw new InvalidDataException(
                        "Buffer copy shader artifact is empty: " + bufferCopyRelativePath
                    );
                ReadUtf8(bufferCopyShaderPath);
            }

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

            List<ValidatedDeviceBuffer> deviceBuffers = ValidateDeviceBuffers(
                manifest.device == null ? null : manifest.device.buffers
            );

            List<ValidatedResource> resources = new List<ValidatedResource>();
            HashSet<string> resourceNames = new HashSet<string>(StringComparer.Ordinal);
            if (manifest.resources != null)
            {
                foreach (ArtifactResourceInfo resource in manifest.resources)
                {
                    if (
                        resource == null
                        || !IsSafeIdentifier(resource.name)
                        || !IsSafeIdentifier(resource.kind)
                        || !resourceNames.Add(resource.name)
                    )
                        throw new InvalidDataException(
                            "Each resource needs a unique safe name and kind."
                        );
                    string extension;
                    if (resource.kind == "shader")
                        extension = ".shader";
                    else if (resource.kind == "bytes")
                        extension = ".bytes";
                    else
                        throw new InvalidDataException(
                            "Unsupported artifact resource kind: " + resource.kind
                        );
                    string relativePath = ValidateRelativeFilePath(resource.path, extension);
                    string absolutePath = ResolveExistingFile(root, relativePath);
                    if (new FileInfo(absolutePath).Length == 0)
                        throw new InvalidDataException(
                            "Artifact resource is empty: " + relativePath
                        );
                    resources.Add(
                        new ValidatedResource(
                            resource.name,
                            resource.kind,
                            relativePath,
                            absolutePath
                        )
                    );
                }
            }
            ValidateOnnxResources(resources);
            List<ValidatedOnnxStaticInput> onnxStaticInputs = ValidateOnnxStaticInputs(
                root,
                manifest.static_inputs
            );
            List<ValidatedOnnxInt4Call> onnxInt4Calls = ValidateOnnxInt4Calls(
                root,
                manifest.call_artifacts
            );
            List<ValidatedOnnxIo> onnxInputs = ValidateOnnxIo(
                manifest.model_inputs,
                manifest.texture_arena,
                "input"
            );
            List<ValidatedOnnxIo> onnxOutputs = ValidateOnnxIo(
                manifest.model_outputs,
                manifest.texture_arena,
                "output"
            );
            if (
                manifest.artifact_stage == "unity-udon-onnx-model"
                && (
                    onnxInputs.Count == 0
                    || onnxOutputs.Count == 0
                    || string.IsNullOrEmpty(manifest.program.buffer_copy_shader)
                )
            )
                throw new InvalidDataException(
                    "Unity ONNX models require public inputs, outputs, and a buffer copy shader."
                );

            return new ValidatedArtifact(
                root,
                manifestPath,
                assemblyPath,
                displayShaderPath,
                bufferCopyShaderPath,
                manifest.artifact_stage,
                manifest.program.name,
                kernels,
                deviceBuffers,
                resources,
                onnxStaticInputs,
                onnxInt4Calls,
                onnxInputs,
                onnxOutputs
            );
        }

        private static List<ValidatedOnnxIo> ValidateOnnxIo(
            UdonOnnxIoInfo[] entries,
            UdonOnnxTextureArenaInfo arena,
            string kind
        )
        {
            List<ValidatedOnnxIo> validated = new List<ValidatedOnnxIo>();
            HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
            UdonOnnxTextureSlotInfo[] slots = arena?.slots;
            foreach (UdonOnnxIoInfo entry in entries ?? Array.Empty<UdonOnnxIoInfo>())
            {
                if (
                    entry == null
                    || entry.index != validated.Count
                    || string.IsNullOrWhiteSpace(entry.name)
                    || !names.Add(entry.name)
                    || (entry.data_type != "float32" && entry.data_type != "int32")
                    || entry.words <= 0
                    || slots == null
                    || entry.slot < 0
                    || entry.slot >= slots.Length
                    || slots[entry.slot] == null
                    || slots[entry.slot].index != entry.slot
                    || entry.words > slots[entry.slot].capacity_words
                )
                    throw new InvalidDataException("ONNX model " + kind + " binding is invalid.");
                validated.Add(
                    new ValidatedOnnxIo(
                        entry.index,
                        entry.name,
                        entry.data_type,
                        entry.words,
                        entry.slot
                    )
                );
            }
            return validated;
        }

        private static List<ValidatedDeviceBuffer> ValidateDeviceBuffers(DeviceBufferInfo[] buffers)
        {
            List<ValidatedDeviceBuffer> validated = new List<ValidatedDeviceBuffer>();
            HashSet<int> ids = new HashSet<int>();
            HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
            HashSet<string> heapSlots = new HashSet<string>(StringComparer.Ordinal);
            foreach (DeviceBufferInfo buffer in buffers ?? Array.Empty<DeviceBufferInfo>())
            {
                if (buffer == null)
                    throw new InvalidDataException("Device buffer entry is null.");
                long expectedPixels = ((long)buffer.logical_bytes + 3) / 4;
                int expectedWidth = (int)Math.Min(expectedPixels, 8192);
                int expectedHeight =
                    expectedWidth == 0
                        ? 0
                        : (int)((expectedPixels + expectedWidth - 1) / expectedWidth);
                bool hasPublicBinding =
                    !string.IsNullOrEmpty(buffer.element_type)
                    || !string.IsNullOrEmpty(buffer.public_access);
                bool validPublicBinding =
                    !hasPublicBinding
                    || (
                        (buffer.element_type == "float32" || buffer.element_type == "int32")
                        && (buffer.public_access == "input" || buffer.public_access == "output")
                    );
                if (
                    buffer.id <= 0
                    || !ids.Add(buffer.id)
                    || !IsSafeIdentifier(buffer.name)
                    || !names.Add(buffer.name)
                    || !IsSafeIdentifier(buffer.heap_slot)
                    || !heapSlots.Add(buffer.heap_slot)
                    || buffer.heap_slot != "__udoncl_buffer_" + buffer.id
                    || buffer.logical_bytes <= 0
                    || buffer.width != expectedWidth
                    || buffer.height != expectedHeight
                    || buffer.height > 8192
                    || buffer.format != "rgba8_unorm"
                    || buffer.srgb
                    || buffer.mipmaps
                    || buffer.filter != "point"
                    || !validPublicBinding
                )
                    throw new InvalidDataException(
                        "Device buffers require unique ids/names/slots and a canonical linear RGBA8 layout."
                    );
                validated.Add(
                    new ValidatedDeviceBuffer(
                        buffer.id,
                        buffer.name,
                        buffer.heap_slot,
                        buffer.logical_bytes,
                        buffer.width,
                        buffer.height,
                        buffer.element_type,
                        buffer.public_access
                    )
                );
            }
            return validated;
        }

        private static List<ValidatedOnnxInt4Call> ValidateOnnxInt4Calls(
            string root,
            UdonOnnxCallArtifactInfo[] calls
        )
        {
            List<ValidatedOnnxInt4Call> validated = new List<ValidatedOnnxInt4Call>();
            HashSet<int> callIndexes = new HashSet<int>();
            foreach (
                UdonOnnxCallArtifactInfo call in calls ?? Array.Empty<UdonOnnxCallArtifactInfo>()
            )
            {
                if (call == null || call.call_index < 0 || !callIndexes.Add(call.call_index))
                    throw new InvalidDataException(
                        "ONNX INT4 call index is invalid or duplicated."
                    );
                string manifestRelativePath = ValidateRelativeFilePath(
                    call.path + "/manifest.json",
                    ".json"
                );
                string manifestPath = ResolveExistingFile(root, manifestRelativePath);
                UdonOnnxInt4CallManifest child;
                try
                {
                    child = JsonUtility.FromJson<UdonOnnxInt4CallManifest>(ReadUtf8(manifestPath));
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException("ONNX INT4 child manifest is invalid JSON.", ex);
                }
                if (
                    child == null
                    || child.schema_version != 1
                    || child.artifact_stage != "udon-onnx-int4-call"
                    || child.call_index != call.call_index
                    || child.function != "udonatenWeightInt4packMm"
                    || child.result_layout == null
                    || child.result_layout.width <= 0
                    || child.result_layout.width > 8192
                    || child.result_layout.height <= 0
                )
                    throw new InvalidDataException("ONNX INT4 child manifest contract is invalid.");
                string childRoot = Path.GetDirectoryName(manifestPath);
                string assemblyRelativePath = ValidateRelativeFilePath(child.assembly, ".uasm");
                string assemblyPath = ResolveExistingFile(childRoot, assemblyRelativePath);
                ReadUtf8(assemblyPath);
                List<ValidatedOnnxInt4Atlas> atlases = new List<ValidatedOnnxInt4Atlas>();
                HashSet<string> atlasNames = new HashSet<string>(StringComparer.Ordinal);
                foreach (
                    ArtifactResourceInfo resource in child.resources
                        ?? Array.Empty<ArtifactResourceInfo>()
                )
                {
                    if (
                        resource == null
                        || !IsSafeIdentifier(resource.name)
                        || !atlasNames.Add(resource.name)
                        || resource.kind != "bytes"
                        || resource.encoding != "q4-bf16-row80-rgba32"
                        || resource.width != 1280
                        || resource.height <= 0
                        || resource.height > 8192
                    )
                        throw new InvalidDataException("ONNX INT4 atlas contract is invalid.");
                    string relativePath = ValidateRelativeFilePath(resource.path, ".rgba");
                    string absolutePath = ResolveExistingFile(childRoot, relativePath);
                    long expectedBytes = (long)resource.width * resource.height * 4;
                    if (
                        resource.bytes != expectedBytes
                        || new FileInfo(absolutePath).Length != expectedBytes
                    )
                        throw new InvalidDataException("ONNX INT4 atlas byte size is invalid.");
                    atlases.Add(
                        new ValidatedOnnxInt4Atlas(
                            resource.name,
                            absolutePath,
                            resource.width,
                            resource.height
                        )
                    );
                }
                if (atlases.Count == 0)
                    throw new InvalidDataException("ONNX INT4 child has no weight atlas.");
                validated.Add(
                    new ValidatedOnnxInt4Call(
                        call.call_index,
                        assemblyPath,
                        child.result_layout.width,
                        child.result_layout.height,
                        atlases
                    )
                );
            }
            validated.Sort((left, right) => left.CallIndex.CompareTo(right.CallIndex));
            return validated;
        }

        private static List<ValidatedOnnxStaticInput> ValidateOnnxStaticInputs(
            string root,
            UdonOnnxStaticInputInfo[] inputs
        )
        {
            List<ValidatedOnnxStaticInput> validated = new List<ValidatedOnnxStaticInput>();
            HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
            foreach (
                UdonOnnxStaticInputInfo input in inputs ?? Array.Empty<UdonOnnxStaticInputInfo>()
            )
            {
                UdonOnnxStaticTextureInfo texture = input?.texture;
                if (
                    input == null
                    || string.IsNullOrWhiteSpace(input.name)
                    || !names.Add(input.name)
                    || texture == null
                    || texture.abi != "linear-rgba32-words"
                    || texture.words <= 0
                    || texture.width != Math.Min(8192, texture.words)
                    || texture.height != (texture.words + 8191) / 8192
                )
                    throw new InvalidDataException(
                        "ONNX static input texture contract is invalid."
                    );
                string relativePath = ValidateRelativeFilePath(input.resource, ".bytes");
                string absolutePath = ResolveExistingFile(root, relativePath);
                long expectedBytes = (long)texture.width * texture.height * 4;
                if (new FileInfo(absolutePath).Length != expectedBytes)
                    throw new InvalidDataException(
                        "ONNX static input byte size does not match its texture dimensions."
                    );
                validated.Add(
                    new ValidatedOnnxStaticInput(
                        input.name,
                        absolutePath,
                        texture.words,
                        texture.width,
                        texture.height
                    )
                );
            }
            return validated;
        }

        private static void ValidateOnnxResources(IList<ValidatedResource> resources)
        {
            ValidatedResource model = null;
            ValidatedResource planResource = null;
            foreach (ValidatedResource resource in resources)
            {
                if (resource.Name == "onnx_model" && resource.Kind == "bytes")
                    model = resource;
                else if (resource.Name == "onnx_execution_plan" && resource.Kind == "bytes")
                    planResource = resource;
            }
            if (model == null && planResource == null)
                return;
            if (model == null || planResource == null)
                throw new InvalidDataException(
                    "ONNX artifacts require paired onnx_model and onnx_execution_plan byte resources."
                );
            UdonOnnxExecutionPlan plan;
            try
            {
                plan = JsonUtility.FromJson<UdonOnnxExecutionPlan>(
                    ReadUtf8(planResource.AbsolutePath)
                );
            }
            catch (Exception ex)
            {
                throw new InvalidDataException("ONNX execution plan is invalid JSON.", ex);
            }
            if (
                plan == null
                || plan.schema_version != 1
                || plan.runtime != "udon-onnx"
                || !IsSafeIdentifier(plan.execution_provider)
                || plan.passes == null
                || plan.passes.Length == 0
            )
                throw new InvalidDataException(
                    "ONNX execution plan requires schema 1, runtime udon-onnx, a safe execution provider, and Device passes."
                );
            HashSet<int> passIds = new HashSet<int>();
            foreach (UdonOnnxPassInfo pass in plan.passes)
            {
                if (
                    pass == null
                    || pass.id < 0
                    || !passIds.Add(pass.id)
                    || !IsSafeIdentifier(pass.kernel)
                    || pass.fragments <= 0
                    || pass.providers == null
                    || pass.providers.Length == 0
                )
                    throw new InvalidDataException(
                        "Every ONNX Device pass needs a unique id, safe kernel, providers, and positive fragment count."
                    );
                foreach (string provider in pass.providers)
                    if (!IsSafeIdentifier(provider))
                        throw new InvalidDataException(
                            "ONNX Device pass provider names must be safe identifiers."
                        );
            }
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
