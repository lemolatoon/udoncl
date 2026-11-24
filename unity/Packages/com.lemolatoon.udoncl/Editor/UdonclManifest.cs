using System;

namespace Lemolatoon.Udoncl.Editor
{
    [Serializable]
    internal sealed class UdonclManifest
    {
        public int schema_version;
        public string compiler_version;
        public int runtime_abi_version;
        public string public_abi_fingerprint;
        public string physical_export_abi;
        public string artifact_stage;
        public bool ready_for_unity;
        public SdkCompatibility sdk_compatibility;
        public TargetInfo target;
        public ProgramInfo program;
        public ExportInfo[] exports;
        public DeviceInfo device;
        public DispatchSiteInfo[] dispatch_sites;
        public DispatchMaterialInfo[] dispatch_materials;
        public ValidationInfo validation;
        public ArtifactResourceInfo[] resources;
        public StaticTensorInfo[] static_tensors;
        public UdonOnnxBindingInfo onnx_binding;
    }

    [Serializable]
    internal sealed class ValidationInfo
    {
        public bool read_write_alias;
    }

    [Serializable]
    internal sealed class DispatchSiteInfo
    {
        public int kernel_id;
    }

    [Serializable]
    internal sealed class DispatchMaterialInfo
    {
        public string slot;
        public int kernel_id;
    }

    [Serializable]
    internal sealed class SdkCompatibility
    {
        public string sdk_version;
        public string unity_version;
        public string extern_db_hash;
    }

    [Serializable]
    internal sealed class TargetInfo
    {
        public int host_pointer_bits;
        public int device_pointer_bits;
        public string endianness;
        public string platform;
    }

    [Serializable]
    internal sealed class ProgramInfo
    {
        public string name;
        public string assembly;
        public string display_shader;
        public string buffer_copy_shader;
        public bool buffer_copy_program_slot;
        public string buffer_pool_slot;
        public string raw_heap_initializer;
        public string raw_heap_slot;
        public int raw_heap_words;
        public string static_tensor_table_slot;
        public string binding_source;
        public string binding_namespace;
        public string binding_class;
        public AsyncCommandInfo async_commands;
        public string texel_format;
        public TransferScratchInfo transfer_scratch;
    }

    /// <summary>
    /// The texture a word-texel program's partial transfers snapshot their
    /// destination into (see the generated buffer-copy shader).
    /// </summary>
    [Serializable]
    internal sealed class TransferScratchInfo
    {
        public string slot;
        public int width;
        public int height;
        public string format;
        public bool program_slot;
    }

    [Serializable]
    internal sealed class AsyncCommandInfo
    {
        public string resume_state_slot;
        public string completion_status_slot;
        public string pending_object_slot;
        public string completion_object_slot;
        public string completion_event;
        public string command_kind_slot;
        public string command_buffer_slot;
        public string command_offset_slot;
        public string command_size_slot;
        public string command_pointer_slot;
        public string raw_heap_slot;
        public int upload_kind;
        public int readback_kind;
    }

    [Serializable]
    internal sealed class ExportInfo
    {
        public int id;
        public string c_symbol;
        public string public_name;
        public string udon_event;
        public ExportParameterInfo[] parameters;
        public ExportResultInfo result;
        public string status_slot;
        public bool async;
    }

    [Serializable]
    internal sealed class ExportParameterInfo
    {
        public int index;
        public string name;
        public string kind;
        public TypeSchemaInfo type_schema;
        public string heap_slot;
    }

    [Serializable]
    internal sealed class ExportResultInfo
    {
        public string kind;
        public TypeSchemaInfo type_schema;
        public string heap_slot;
    }

    [Serializable]
    internal sealed class TypeSchemaInfo
    {
        public string category;
        public int bits;
        public string element_category;
        public int element_bits;
        public int rank;
        public int elements;
    }

    [Serializable]
    internal sealed class DeviceInfo
    {
        public KernelInfo[] kernels;
        public DeviceBufferInfo[] buffers;
    }

    [Serializable]
    internal sealed class DeviceBufferInfo
    {
        public int id;
        public string name;
        public string heap_slot;
        public int logical_bytes;
        public int width;
        public int height;
        public string format;
        public bool srgb;
        public bool mipmaps;
        public string filter;
        public string element_type;
        public string public_access;
    }

    [Serializable]
    internal sealed class KernelInfo
    {
        public int id;
        public string name;
        public string shader;
    }

    [Serializable]
    internal sealed class ArtifactResourceInfo
    {
        public string name;
        public string kind;
        public string path;
        public string encoding;
        public int width;
        public int height;
        public int bytes;
    }

    [Serializable]
    internal sealed class StaticTensorInfo
    {
        public int id;
        public string name;
        public string resource;
        public string data_type;
        public long[] shape;
        public string layout;
        public int alignment;
        public long[] byte_strides;
        public int byte_length;
        public string sha256;
        public StaticTensorTextureInfo texture;
    }

    [Serializable]
    internal sealed class StaticTensorTextureInfo
    {
        public string abi;
        public int width;
        public int height;
        public string format;
        public bool srgb;
        public bool mipmaps;
        public string filter;
    }

    [Serializable]
    internal sealed class UdonOnnxBindingInfo
    {
        public string abi;
        public int model_index;
        public string model_name;
        public UdonOnnxBindingTensorInfo[] inputs;
        public UdonOnnxBindingTensorInfo[] outputs;

        // Graph outputs a session may bind to the buffer of their Concat's
        // first operand, a graph input (absent when there is none).
        public UdonOnnxInPlaceConcatInfo[] in_place_concats;
    }

    [Serializable]
    internal sealed class UdonOnnxInPlaceConcatInfo
    {
        public int input;
        public int output;
        public int axis;
    }

    [Serializable]
    internal sealed class UdonOnnxBindingTensorInfo
    {
        public int index;
        public string name;
        public string data_type;
        public int rank;

        // -1 on the axis of a bounded dimension, whose largest extent is in
        // dimension_bounds and whose name is in dimension_names (both absent
        // for a static tensor).
        public long[] dimensions;
        public long[] dimension_bounds;
        public string[] dimension_names;
    }
}
