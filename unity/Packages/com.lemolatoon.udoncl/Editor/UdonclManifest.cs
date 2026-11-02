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
        public ArtifactResourceInfo[] resources;
        public UdonOnnxStaticInputInfo[] static_inputs;
        public UdonOnnxCallArtifactInfo[] call_artifacts;
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
    internal sealed class UdonOnnxCallArtifactInfo
    {
        public int call_index;
        public string path;
    }

    [Serializable]
    internal sealed class UdonOnnxInt4CallManifest
    {
        public int schema_version;
        public string artifact_stage;
        public int call_index;
        public string function;
        public string assembly;
        public UdonOnnxTextureLayout result_layout;
        public ArtifactResourceInfo[] resources;
    }

    [Serializable]
    internal sealed class UdonOnnxTextureLayout
    {
        public int width;
        public int height;
    }

    [Serializable]
    internal sealed class UdonOnnxExecutionPlan
    {
        public int schema_version;
        public string runtime;
        public string profile;
        public string execution_provider;
        public UdonOnnxPassInfo[] passes;
        public UdonOnnxTextureArenaInfo texture_arena;
    }

    [Serializable]
    internal sealed class UdonOnnxPassInfo
    {
        public int id;
        public string kernel;
        public string shader;
        public string[] providers;
        public int fragments;
        public UdonOnnxIntMaterialBinding[] material_int_bindings;
        public UdonOnnxFloatMaterialBinding[] material_float_bindings;
        public UdonOnnxPassStaticInput[] static_inputs;
        public bool has_host_material_index;
        public int host_material_index;
    }

    [Serializable]
    internal sealed class UdonOnnxIntMaterialBinding
    {
        public string property;
        public int value;
    }

    [Serializable]
    internal sealed class UdonOnnxFloatMaterialBinding
    {
        public string property;
        public float value;
    }

    [Serializable]
    internal sealed class UdonOnnxPassStaticInput
    {
        public string property;
        public string initializer;
    }

    [Serializable]
    internal sealed class UdonOnnxStaticInputInfo
    {
        public string name;
        public string resource;
        public UdonOnnxStaticTextureInfo texture;
    }

    [Serializable]
    internal sealed class UdonOnnxStaticTextureInfo
    {
        public string abi;
        public int words;
        public int width;
        public int height;
    }

    [Serializable]
    internal sealed class UdonOnnxTextureArenaInfo
    {
        public string abi;
        public int slot_count;
        public int total_words;
        public int total_bytes;
        public UdonOnnxTextureSlotInfo[] slots;
    }

    [Serializable]
    internal sealed class UdonOnnxTextureSlotInfo
    {
        public int index;
        public int capacity_words;
        public int width;
        public int height;
    }
}
