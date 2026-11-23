using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
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
            string bindingSourceAssetPath,
            string bindingNamespace,
            string bindingClass,
            string staticTensorTableSlot,
            string artifactStage,
            string programName,
            IList<ValidatedKernel> kernels,
            IList<ValidatedDispatchMaterial> dispatchMaterials,
            IList<ValidatedDeviceBuffer> deviceBuffers,
            IList<ValidatedResource> resources,
            IList<ValidatedStaticTensor> staticTensors
        )
        {
            RootDirectory = root;
            ManifestAssetPath = manifestAssetPath;
            AssemblyAssetPath = assemblyAssetPath;
            DisplayShaderAssetPath = displayShaderAssetPath;
            BufferCopyShaderAssetPath = bufferCopyShaderAssetPath;
            BindingSourceAssetPath = bindingSourceAssetPath;
            BindingNamespace = bindingNamespace;
            BindingClass = bindingClass;
            StaticTensorTableSlot = staticTensorTableSlot;
            ArtifactStage = artifactStage;
            ProgramName = programName;
            Kernels = kernels;
            DispatchMaterials = dispatchMaterials;
            DeviceBuffers = deviceBuffers;
            Resources = resources;
            StaticTensors = staticTensors;
        }

        public string RootDirectory { get; private set; }
        public string ManifestAssetPath { get; private set; }
        public string AssemblyAssetPath { get; private set; }
        public string DisplayShaderAssetPath { get; private set; }
        public string BufferCopyShaderAssetPath { get; private set; }
        public string BindingSourceAssetPath { get; private set; }
        public string BindingNamespace { get; private set; }
        public string BindingClass { get; private set; }
        public string StaticTensorTableSlot { get; private set; }
        public string ArtifactStage { get; private set; }
        public string ProgramName { get; private set; }
        public IList<ValidatedKernel> Kernels { get; private set; }

        /// <summary>
        /// One Material per dispatch site: the program's public material slot
        /// and the kernel whose shader the material renders.
        /// </summary>
        public IList<ValidatedDispatchMaterial> DispatchMaterials { get; private set; }
        public IList<ValidatedDeviceBuffer> DeviceBuffers { get; private set; }

        /// <summary>
        /// The Device texel format of every buffer and immutable tensor:
        /// "rgba8_unorm" (one word per RGBA8 texel) or "rgba32_float" (four
        /// words per RGBA32 float texel).
        /// </summary>
        public string TexelFormat { get; internal set; }

        /// <summary>
        /// The scratch texture a word-texel program's partial transfers
        /// snapshot their destination into, or null.
        /// </summary>
        public ValidatedTransferScratch TransferScratch { get; internal set; }
        public IList<ValidatedResource> Resources { get; private set; }
        public IList<ValidatedStaticTensor> StaticTensors { get; private set; }
    }

    public sealed class ValidatedStaticTensor
    {
        internal ValidatedStaticTensor(
            int id,
            string name,
            string absolutePath,
            string dataType,
            long[] shape,
            int alignment,
            long[] byteStrides,
            int byteLength,
            string sha256,
            int width,
            int height,
            string format = "rgba32"
        )
        {
            Id = id;
            Name = name;
            AbsolutePath = absolutePath;
            DataType = dataType;
            Shape = shape;
            Alignment = alignment;
            ByteStrides = byteStrides;
            ByteLength = byteLength;
            Sha256 = sha256;
            Width = width;
            Height = height;
            Format = format;
        }

        /// <summary>
        /// "rgba32" (one word per RGBA8 texel) or "rgba32_float" (four words
        /// per RGBA32 float texel).
        /// </summary>
        public string Format { get; private set; }

        public int TexelBytes
        {
            get { return Format == "rgba32_float" ? 16 : 4; }
        }

        public int Id { get; private set; }
        public string Name { get; private set; }
        public string AbsolutePath { get; private set; }
        public string DataType { get; private set; }
        public long[] Shape { get; private set; }
        public int Alignment { get; private set; }
        public long[] ByteStrides { get; private set; }
        public int ByteLength { get; private set; }
        public string Sha256 { get; private set; }
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

    public sealed class ValidatedDispatchMaterial
    {
        internal ValidatedDispatchMaterial(string slot, ValidatedKernel kernel)
        {
            Slot = slot;
            Kernel = kernel;
        }

        public string Slot { get; private set; }
        public ValidatedKernel Kernel { get; private set; }

        /// <summary>The material asset name: the slot without its prefix.</summary>
        public string AssetName
        {
            get { return Slot.Substring("__udoncl_".Length); }
        }
    }

    public sealed class ValidatedTransferScratch
    {
        internal ValidatedTransferScratch(
            string slot,
            int width,
            int height,
            string format,
            bool programSlot
        )
        {
            Slot = slot;
            Width = width;
            Height = height;
            Format = format;
            ProgramSlot = programSlot;
        }

        public string Slot { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public string Format { get; private set; }
        public bool ProgramSlot { get; private set; }
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
            string publicAccess,
            string format = "rgba8_unorm"
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
            Format = format;
        }

        /// <summary>
        /// The texel format: "rgba8_unorm" (one word per texel) or
        /// "rgba32_float" (four words per texel; Width counts texels).
        /// </summary>
        public string Format { get; private set; }

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
        private static readonly Regex CSharpIdentifier = new Regex(
            @"^[A-Za-z_][A-Za-z0-9_]*$",
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

            string manifestSource = ReadUtf8(manifestPath);

            UdonclManifest manifest;
            try
            {
                manifest = JsonUtility.FromJson<UdonclManifest>(manifestSource);
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
            bool requiresValidationAudit = manifest.artifact_stage == "unity-udoncl-library";
            if (
                requiresValidationAudit
                && (
                    manifestSource.IndexOf("\"validation\"", StringComparison.Ordinal) < 0
                    || manifestSource.IndexOf("\"read_write_alias\"", StringComparison.Ordinal) < 0
                )
            )
                throw new InvalidDataException("manifest.validation.read_write_alias is required.");
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
            if (requiresValidationAudit && manifest.validation == null)
                throw new InvalidDataException("manifest.validation.read_write_alias is required.");

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

            string bindingSourcePath = null;
            if (!string.IsNullOrEmpty(manifest.program.binding_source))
            {
                if (!IsSha256(manifest.public_abi_fingerprint))
                    throw new InvalidDataException(
                        "Compiler binding requires a lowercase SHA-256 public ABI fingerprint."
                    );
                if (
                    !string.Equals(
                        manifest.physical_export_abi,
                        "udonsharp-worlds-sdk-3.10.5",
                        StringComparison.Ordinal
                    )
                )
                    throw new InvalidDataException(
                        "Compiler binding uses an unsupported physical export ABI."
                    );
                string bindingRelativePath = ValidateRelativeFilePath(
                    manifest.program.binding_source,
                    ".cs"
                );
                if (
                    !IsDottedIdentifier(manifest.program.binding_namespace)
                    || !IsCSharpIdentifier(manifest.program.binding_class)
                )
                    throw new InvalidDataException(
                        "Compiler binding requires valid namespace and class metadata."
                    );
                ValidateCompilerBindingFingerprint(manifest);
                bindingSourcePath = ResolveExistingFile(root, bindingRelativePath);
                if (new FileInfo(bindingSourcePath).Length == 0)
                    throw new InvalidDataException("Compiler binding source is empty.");
                ValidateCompilerBindingSource(manifest, ReadUtf8(bindingSourcePath));
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

            List<ValidatedDispatchMaterial> dispatchMaterials = ValidateDispatchMaterials(
                manifest.dispatch_materials,
                kernels
            );

            List<ValidatedDeviceBuffer> deviceBuffers = ValidateDeviceBuffers(
                manifest.device == null ? null : manifest.device.buffers
            );
            if (
                deviceBuffers.Count != 0
                && manifest.artifact_stage == "unity-udoncl-library"
                && manifest.program.buffer_pool_slot != "__udoncl_buffer_pool"
            )
                throw new InvalidDataException(
                    "Device buffers require the compiler-owned program buffer pool slot."
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
                    else if (resource.kind == "json")
                        extension = ".json";
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
                    if (resource.kind == "json")
                        ReadUtf8(absolutePath);
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
            if (!string.IsNullOrEmpty(manifest.program.raw_heap_initializer))
            {
                ArtifactResourceInfo rawHeap =
                    manifest.resources == null
                        ? null
                        : Array.Find(
                            manifest.resources,
                            resource =>
                                resource != null
                                && resource.name == "raw_heap_initializer"
                                && resource.kind == "bytes"
                        );
                if (
                    rawHeap == null
                    || rawHeap.path != manifest.program.raw_heap_initializer
                    || string.IsNullOrEmpty(manifest.program.raw_heap_slot)
                    || manifest.program.raw_heap_words <= 0
                    || (rawHeap.bytes + 3) / 4 != manifest.program.raw_heap_words
                )
                    throw new InvalidDataException(
                        "Raw heap initializer metadata and byte resource must match."
                    );
            }
            List<ValidatedStaticTensor> staticTensors = ValidateStaticTensors(
                manifest.static_tensors,
                resources
            );
            if (
                !string.IsNullOrEmpty(manifest.program.static_tensor_table_slot)
                && (
                    !IsSafeIdentifier(manifest.program.static_tensor_table_slot)
                    || staticTensors.Count == 0
                )
            )
                throw new InvalidDataException(
                    "Static tensor table slot requires static tensors and a safe Udon symbol."
                );
            string texelFormat = string.IsNullOrEmpty(manifest.program.texel_format)
                ? "rgba8_unorm"
                : manifest.program.texel_format;
            if (DeviceTexelBytes(texelFormat) == 0)
                throw new InvalidDataException("Unsupported Device texel format: " + texelFormat);
            foreach (ValidatedDeviceBuffer buffer in deviceBuffers)
                if (buffer.Format != texelFormat)
                    throw new InvalidDataException(
                        "Device buffer " + buffer.Name + " does not use the program texel format."
                    );
            foreach (ValidatedStaticTensor tensor in staticTensors)
                if ((tensor.Format == "rgba32_float") != (texelFormat == "rgba32_float"))
                    throw new InvalidDataException(
                        "Static tensor " + tensor.Name + " does not use the program texel format."
                    );
            ValidatedTransferScratch transferScratch = null;
            TransferScratchInfo scratch = manifest.program.transfer_scratch;
            bool hasScratch = scratch != null && !string.IsNullOrEmpty(scratch.slot);
            if (hasScratch != (texelFormat == "rgba32_float" && deviceBuffers.Count != 0))
                throw new InvalidDataException(
                    "A word-texel program with Device buffers requires exactly one transfer scratch texture."
                );
            if (hasScratch)
            {
                if (
                    scratch.slot != "__udoncl_transfer_scratch"
                    || scratch.format != texelFormat
                    || scratch.width <= 0
                    || scratch.width > 8192
                    || scratch.height <= 0
                    || scratch.height > 8192
                )
                    throw new InvalidDataException("Transfer scratch texture metadata is invalid.");
                foreach (ValidatedDeviceBuffer buffer in deviceBuffers)
                    if (buffer.Width > scratch.width || buffer.Height > scratch.height)
                        throw new InvalidDataException(
                            "Transfer scratch texture is smaller than Device buffer "
                                + buffer.Name
                                + "."
                        );
                transferScratch = new ValidatedTransferScratch(
                    scratch.slot,
                    scratch.width,
                    scratch.height,
                    scratch.format,
                    scratch.program_slot
                );
            }

            ValidatedArtifact artifact = new ValidatedArtifact(
                root,
                manifestPath,
                assemblyPath,
                displayShaderPath,
                bufferCopyShaderPath,
                bindingSourcePath,
                manifest.program.binding_namespace,
                manifest.program.binding_class,
                manifest.program.static_tensor_table_slot,
                manifest.artifact_stage,
                manifest.program.name,
                kernels,
                dispatchMaterials,
                deviceBuffers,
                resources,
                staticTensors
            );
            artifact.TexelFormat = texelFormat;
            artifact.TransferScratch = transferScratch;
            return artifact;
        }

        private static readonly Regex DispatchMaterialSlot = new Regex(
            @"^__udoncl_(dispatch_material_[0-9]+|kernel_material_[0-9]+)$",
            RegexOptions.Compiled
        );

        private static List<ValidatedDispatchMaterial> ValidateDispatchMaterials(
            DispatchMaterialInfo[] materials,
            IList<ValidatedKernel> kernels
        )
        {
            List<ValidatedDispatchMaterial> result = new List<ValidatedDispatchMaterial>();
            if (materials == null)
                return result;
            HashSet<string> slots = new HashSet<string>(StringComparer.Ordinal);
            foreach (DispatchMaterialInfo material in materials)
            {
                if (
                    material == null
                    || string.IsNullOrEmpty(material.slot)
                    || !DispatchMaterialSlot.IsMatch(material.slot)
                )
                    throw new InvalidDataException(
                        "Each dispatch material needs a compiler-owned material slot."
                    );
                if (!slots.Add(material.slot))
                    throw new InvalidDataException(
                        "Duplicate dispatch material slot: " + material.slot
                    );
                ValidatedKernel kernel = null;
                foreach (ValidatedKernel candidate in kernels)
                    if (candidate.Id == material.kernel_id)
                        kernel = candidate;
                if (kernel == null)
                    throw new InvalidDataException(
                        "Dispatch material "
                            + material.slot
                            + " references missing kernel "
                            + material.kernel_id
                    );
                result.Add(new ValidatedDispatchMaterial(material.slot, kernel));
            }
            return result;
        }

        private static bool IsDottedIdentifier(string value)
        {
            if (string.IsNullOrEmpty(value))
                return false;
            string[] components = value.Split('.');
            return Array.TrueForAll(components, IsCSharpIdentifier);
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
                int texelBytes = DeviceTexelBytes(buffer.format);
                long textureBytes = (long)buffer.width * buffer.height * texelBytes;
                bool hasPublicBinding =
                    !string.IsNullOrEmpty(buffer.element_type)
                    || !string.IsNullOrEmpty(buffer.public_access);
                bool validPublicBinding =
                    !hasPublicBinding
                    || (
                        (
                            buffer.element_type == "float32"
                            || buffer.element_type == "int32"
                            || buffer.element_type == "int64"
                        )
                        && (
                            buffer.public_access == "input"
                            || buffer.public_access == "output"
                            || buffer.public_access == "inout"
                        )
                    );
                if (
                    buffer.id <= 0
                    || !ids.Add(buffer.id)
                    || !IsSafeIdentifier(buffer.name)
                    || !names.Add(buffer.name)
                    || !IsSafeIdentifier(buffer.heap_slot)
                    || !heapSlots.Add(buffer.heap_slot)
                    || buffer.heap_slot != "__udoncl_buffer_" + buffer.id
                    || buffer.logical_bytes < 0
                    || buffer.width <= 0
                    || buffer.width > 8192
                    || buffer.height <= 0
                    || buffer.height > 8192
                    || textureBytes < buffer.logical_bytes
                    || texelBytes == 0
                    || buffers[0] == null
                    || buffer.format != buffers[0].format
                    || buffer.srgb
                    || buffer.mipmaps
                    || buffer.filter != "point"
                    || !validPublicBinding
                )
                    throw new InvalidDataException(
                        "Device buffers require unique ids/names/slots and a bounded RGBA8 or RGBA32 float layout of one format with sufficient capacity."
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
                        buffer.public_access,
                        buffer.format
                    )
                );
            }
            return validated;
        }

        /// <summary>Bytes per texel of a Device buffer texel format, 0 if unsupported.</summary>
        internal static int DeviceTexelBytes(string format)
        {
            return format == "rgba8_unorm" ? 4
                : format == "rgba32_float" ? 16
                : 0;
        }

        private static List<ValidatedStaticTensor> ValidateStaticTensors(
            StaticTensorInfo[] tensors,
            IList<ValidatedResource> resources
        )
        {
            List<ValidatedStaticTensor> validated = new List<ValidatedStaticTensor>();
            HashSet<int> ids = new HashSet<int>();
            HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
            foreach (StaticTensorInfo tensor in tensors ?? Array.Empty<StaticTensorInfo>())
            {
                StaticTensorTextureInfo texture = tensor?.texture;
                ValidatedResource resource = null;
                if (tensor != null)
                    foreach (ValidatedResource candidate in resources)
                        if (candidate.Name == tensor.resource)
                        {
                            resource = candidate;
                            break;
                        }
                int elementBytes = StaticTensorElementBytes(tensor?.data_type);
                if (
                    tensor == null
                    || tensor.id < 0
                    || !ids.Add(tensor.id)
                    || string.IsNullOrWhiteSpace(tensor.name)
                    || !names.Add(tensor.name)
                    || resource == null
                    || resource.Kind != "bytes"
                    || elementBytes == 0
                    || tensor.shape == null
                    || tensor.layout != "row_major_contiguous"
                    || tensor.alignment != elementBytes
                    || tensor.byte_strides == null
                    || tensor.byte_strides.Length != tensor.shape.Length
                    || tensor.byte_length <= 0
                    || !IsSha256(tensor.sha256)
                    || texture == null
                    || !(
                        (texture.abi == "udoncl-static-rgba32-v1" && texture.format == "rgba32")
                        || (
                            texture.abi == "udoncl-static-words-v1"
                            && texture.format == "rgba32_float"
                        )
                    )
                    || texture.width <= 0
                    || texture.width > 8192
                    || texture.height <= 0
                    || texture.height > 8192
                    || texture.srgb
                    || texture.mipmaps
                    || texture.filter != "point"
                )
                    throw new InvalidDataException("Immutable static tensor metadata is invalid.");
                long elements = 1;
                long expectedStride = elementBytes;
                for (int dimension = tensor.shape.Length - 1; dimension >= 0; --dimension)
                {
                    long size = tensor.shape[dimension];
                    if (tensor.byte_strides[dimension] != expectedStride)
                        throw new InvalidDataException(
                            "Immutable static tensor byte strides are invalid."
                        );
                    if (
                        size <= 0
                        || expectedStride > long.MaxValue / size
                        || elements > long.MaxValue / size
                    )
                        throw new InvalidDataException("Immutable static tensor shape is invalid.");
                    expectedStride *= size;
                    elements *= size;
                }
                // Words fill rows of up to 8192; a texel holds texelWords
                // consecutive words of its row.
                int texelWords = texture.format == "rgba32_float" ? 4 : 1;
                long texelBytes = 4L * texelWords;
                long pixels = (tensor.byte_length + 3L) / 4;
                long rowWords = Math.Min(pixels, 8192L);
                int expectedWidth = (int)((rowWords + texelWords - 1) / texelWords);
                int expectedHeight = (int)((pixels + rowWords - 1) / rowWords);
                if (
                    texture.width != expectedWidth
                    || texture.height != expectedHeight
                    || (long)texture.width * texture.height * texelBytes < tensor.byte_length
                    || (long)texture.width * texture.height * texelBytes - tensor.byte_length
                        >= (long)texture.width * texelBytes
                    || elements > long.MaxValue / elementBytes
                    || elements * elementBytes != tensor.byte_length
                    || new FileInfo(resource.AbsolutePath).Length != tensor.byte_length
                    || !string.Equals(
                        FileSha256(resource.AbsolutePath),
                        tensor.sha256,
                        StringComparison.Ordinal
                    )
                )
                    throw new InvalidDataException(
                        "Immutable static tensor bytes, shape, or hash do not match."
                    );
                validated.Add(
                    new ValidatedStaticTensor(
                        tensor.id,
                        tensor.name,
                        resource.AbsolutePath,
                        tensor.data_type,
                        (long[])tensor.shape.Clone(),
                        tensor.alignment,
                        (long[])tensor.byte_strides.Clone(),
                        tensor.byte_length,
                        tensor.sha256,
                        texture.width,
                        texture.height,
                        texture.format
                    )
                );
            }
            validated.Sort((left, right) => left.Id.CompareTo(right.Id));
            for (int index = 0; index < validated.Count; ++index)
                if (validated[index].Id != index)
                    throw new InvalidDataException(
                        "Immutable static tensor ids must be contiguous from zero."
                    );
            return validated;
        }

        private static int StaticTensorElementBytes(string dataType)
        {
            if (dataType == "int8")
                return 1;
            if (dataType == "float16" || dataType == "bfloat16")
                return 2;
            if (dataType == "int32" || dataType == "float32")
                return 4;
            if (dataType == "int64")
                return 8;
            return 0;
        }

        private static string FileSha256(string path)
        {
            byte[] digest;
            using (FileStream stream = File.OpenRead(path))
            using (SHA256 sha256 = SHA256.Create())
                digest = sha256.ComputeHash(stream);
            StringBuilder result = new StringBuilder(64);
            foreach (byte value in digest)
                result.Append(value.ToString("x2"));
            return result.ToString();
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

        private static bool IsCSharpIdentifier(string value)
        {
            return !string.IsNullOrWhiteSpace(value) && CSharpIdentifier.IsMatch(value);
        }

        private static bool IsSha256(string value)
        {
            if (value == null || value.Length != 64)
                return false;
            for (int index = 0; index < value.Length; ++index)
            {
                char current = value[index];
                if ((current < '0' || current > '9') && (current < 'a' || current > 'f'))
                    return false;
            }
            return true;
        }

        private static void ValidateCompilerBindingSource(UdonclManifest manifest, string source)
        {
            string normalized = source.Replace("\r\n", "\n");
            string header =
                "// <auto-generated />\n"
                + "// udoncl-public-abi: "
                + manifest.public_abi_fingerprint
                + "\n// udoncl-binding-namespace: "
                + manifest.program.binding_namespace
                + "\n// udoncl-binding-class: "
                + manifest.program.binding_class
                + "\n";
            if (!normalized.StartsWith(header, StringComparison.Ordinal))
                throw new InvalidDataException(
                    "Compiler binding source identity does not match its manifest."
                );
        }

        private static void ValidateCompilerBindingFingerprint(UdonclManifest manifest)
        {
            ExportInfo[] exports = manifest.exports;
            if (exports == null || exports.Length == 0)
                throw new InvalidDataException(
                    "Compiler binding requires at least one typed export."
                );

            ExportInfo[] ordered = (ExportInfo[])exports.Clone();
            Array.Sort(ordered, (left, right) => left.id.CompareTo(right.id));
            HashSet<int> ids = new HashSet<int>();
            HashSet<string> names = new HashSet<string>(StringComparer.Ordinal);
            StringBuilder canonical = new StringBuilder();
            canonical.Append("udoncl-public-abi-v2\n");
            canonical.Append("runtime-abi|1\n");
            canonical.Append("physical-abi|udonsharp-worlds-sdk-3.10.5\n");

            foreach (ExportInfo export in ordered)
            {
                if (
                    export == null
                    || export.id <= 0
                    || !ids.Add(export.id)
                    || !IsCSharpIdentifier(export.public_name)
                    || !names.Add(export.public_name)
                    || !IsCSharpIdentifier(export.udon_event)
                    || !IsCSharpIdentifier(export.status_slot)
                )
                    throw new InvalidDataException("Compiler binding export metadata is invalid.");

                canonical.Append(export.id);
                canonical.Append('|');
                canonical.Append(export.public_name);
                canonical.Append("|event=");
                canonical.Append(export.udon_event);
                canonical.Append('|');
                ExportParameterInfo[] parameters =
                    export.parameters ?? Array.Empty<ExportParameterInfo>();
                HashSet<string> parameterNames = new HashSet<string>(StringComparer.Ordinal);
                for (int index = 0; index < parameters.Length; ++index)
                {
                    ExportParameterInfo parameter = parameters[index];
                    if (
                        parameter == null
                        || parameter.index != index
                        || !IsCSharpIdentifier(parameter.name)
                        || !parameterNames.Add(parameter.name)
                        || !IsBindingKind(parameter.kind)
                        || !IsCSharpIdentifier(parameter.heap_slot)
                    )
                        throw new InvalidDataException(
                            "Compiler binding parameter metadata is invalid."
                        );
                    canonical.Append(index);
                    canonical.Append(':');
                    canonical.Append(parameter.name);
                    canonical.Append(':');
                    canonical.Append(parameter.kind);
                    canonical.Append(':');
                    canonical.Append(parameter.heap_slot);
                    canonical.Append(',');
                }
                canonical.Append("->");
                if (export.result == null)
                {
                    canonical.Append("void");
                }
                else
                {
                    if (
                        !IsBindingKind(export.result.kind)
                        || !IsCSharpIdentifier(export.result.heap_slot)
                    )
                        throw new InvalidDataException(
                            "Compiler binding result metadata is invalid."
                        );
                    canonical.Append(export.result.kind);
                    canonical.Append(':');
                    canonical.Append(export.result.heap_slot);
                }
                canonical.Append("|status=");
                canonical.Append(export.status_slot);
                canonical.Append(export.async ? "|async\n" : "|sync\n");
            }

            DeviceBufferInfo[] buffers =
                manifest.device?.buffers ?? Array.Empty<DeviceBufferInfo>();
            DeviceBufferInfo[] publicBuffers = Array.FindAll(
                buffers,
                buffer =>
                    buffer != null
                    && !string.IsNullOrEmpty(buffer.element_type)
                    && !string.IsNullOrEmpty(buffer.public_access)
            );
            Array.Sort(publicBuffers, (left, right) => left.id.CompareTo(right.id));
            // A buffer's optional `source` (the Host variable or ONNX tensor
            // it holds) documents the binding and is not public ABI; udonclc
            // leaves it out of the fingerprint as well (Boundary.cpp).
            foreach (DeviceBufferInfo buffer in publicBuffers)
            {
                canonical.Append("buffer|");
                canonical.Append(buffer.id);
                canonical.Append('|');
                canonical.Append(buffer.name);
                canonical.Append("|bytes=");
                canonical.Append(buffer.logical_bytes);
                canonical.Append("|type=");
                canonical.Append(buffer.element_type);
                canonical.Append("|access=");
                canonical.Append(buffer.public_access);
                canonical.Append('\n');
            }
            canonical.Append("binding-namespace|");
            canonical.Append(manifest.program.binding_namespace);
            canonical.Append('\n');
            canonical.Append("binding-class|");
            canonical.Append(manifest.program.binding_class);
            canonical.Append('\n');

            byte[] digest;
            using (SHA256 sha256 = SHA256.Create())
                digest = sha256.ComputeHash(Encoding.UTF8.GetBytes(canonical.ToString()));
            StringBuilder actual = new StringBuilder(64);
            foreach (byte value in digest)
                actual.Append(value.ToString("x2"));
            if (
                !string.Equals(
                    manifest.public_abi_fingerprint,
                    actual.ToString(),
                    StringComparison.Ordinal
                )
            )
                throw new InvalidDataException(
                    "Compiler binding public ABI fingerprint does not match its typed exports."
                );
        }

        private static bool IsBindingKind(string kind)
        {
            return kind == "i32"
                || kind == "f32"
                || kind == "string"
                || kind == "i32_array"
                || kind == "f32_array";
        }
    }
}
