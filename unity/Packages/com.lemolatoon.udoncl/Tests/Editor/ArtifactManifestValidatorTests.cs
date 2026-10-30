using System;
using System.IO;
using System.Text;
using NUnit.Framework;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class ArtifactManifestValidatorTests
    {
        private string _directory;

        [SetUp]
        public void SetUp()
        {
            _directory = Path.Combine(Path.GetTempPath(), "udoncl-fixture-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_directory);
            File.WriteAllText(Path.Combine(_directory, "host.uasm"), ".data_start\n.data_end\n.code_start\n.code_end\n");
            File.WriteAllText(Path.Combine(_directory, "kernel.shader"), "Shader \"Hidden/udoncl/test\" { SubShader { Pass {} } }");
            File.WriteAllText(Path.Combine(_directory, "display.shader"), "Shader \"Hidden/udoncl/display\" { SubShader { Pass {} } }");
            WriteManifest("host.uasm", "kernel.shader", "3.7.0");
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_directory))
                Directory.Delete(_directory, true);
        }

        [Test]
        public void ValidateAcceptsWorldSdkManifestAndShader()
        {
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual("lj_demo", artifact.ProgramName);
            Assert.AreEqual(1, artifact.Kernels.Count);
            Assert.AreEqual("kernel.shader", artifact.Kernels[0].RelativeShaderPath);
        }

        [Test]
        public void ValidateRejectsTraversalShaderPath()
        {
            WriteManifest("host.uasm", "../escape.shader", "3.7.0");
            Assert.Throws<InvalidDataException>(() => ArtifactManifestValidator.Validate(_directory));
        }

        [Test]
        public void ValidateRejectsAvatarOrNonWorldSdkVersion()
        {
            WriteManifest("host.uasm", "kernel.shader", "2.0.0");
            Assert.Throws<InvalidDataException>(() => ArtifactManifestValidator.Validate(_directory));
        }

        private void WriteManifest(string assemblyPath, string shaderPath, string sdkVersion)
        {
            string json = "{" +
                "\"schema_version\":1," +
                "\"compiler_version\":\"0.1.0\"," +
                "\"runtime_abi_version\":1," +
                "\"sdk_compatibility\":{\"sdk_version\":\"" + sdkVersion + "\"}," +
                "\"target\":{\"host_pointer_bits\":32,\"device_pointer_bits\":32,\"endianness\":\"little\",\"platform\":\"VRChat-PC\"}," +
                "\"program\":{\"name\":\"lj_demo\",\"assembly\":\"" + assemblyPath +
                "\",\"display_shader\":\"display.shader\"}," +
                "\"device\":{\"kernels\":[{\"id\":1,\"name\":\"lj_step\",\"shader\":\"" + shaderPath + "\"}]}" +
                "}";
            File.WriteAllText(Path.Combine(_directory, "manifest.json"), json, new UTF8Encoding(false));
        }
    }
}
