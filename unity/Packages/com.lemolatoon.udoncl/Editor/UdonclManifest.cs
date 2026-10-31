using System;

namespace Lemolatoon.Udoncl.Editor
{
    [Serializable]
    internal sealed class UdonclManifest
    {
        public int schema_version;
        public string compiler_version;
        public int runtime_abi_version;
        public string artifact_stage;
        public bool ready_for_unity;
        public SdkCompatibility sdk_compatibility;
        public TargetInfo target;
        public ProgramInfo program;
        public ExportInfo[] exports;
        public DeviceInfo device;
        public ArtifactResourceInfo[] resources;
    }

    [Serializable]
    internal sealed class SdkCompatibility
    {
        public string sdk_version;
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
        public string udon_event;
        public string status_slot;
        public bool async;
    }

    [Serializable]
    internal sealed class DeviceInfo
    {
        public KernelInfo[] kernels;
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
    }
}
