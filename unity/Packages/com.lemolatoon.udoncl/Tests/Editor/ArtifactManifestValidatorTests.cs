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
            _directory = Path.Combine(
                Path.GetTempPath(),
                "udoncl-fixture-" + Guid.NewGuid().ToString("N")
            );
            Directory.CreateDirectory(_directory);
            File.WriteAllText(
                Path.Combine(_directory, "host.uasm"),
                ".data_start\n.data_end\n.code_start\n.code_end\n"
            );
            File.WriteAllText(
                Path.Combine(_directory, "kernel.shader"),
                "Shader \"Hidden/udoncl/test\" { SubShader { Pass {} } }"
            );
            File.WriteAllText(
                Path.Combine(_directory, "display.shader"),
                "Shader \"Hidden/udoncl/display\" { SubShader { Pass {} } }"
            );
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
            Assert.AreEqual("unity-lj-prototype", artifact.ArtifactStage);
            Assert.AreEqual(1, artifact.Kernels.Count);
            Assert.AreEqual("kernel.shader", artifact.Kernels[0].RelativeShaderPath);
        }

        [Test]
        public void ValidateAcceptsOptionalBufferCopyShader()
        {
            File.WriteAllText(
                Path.Combine(_directory, "buffer-copy.shader"),
                "Shader \"Hidden/udoncl/copy\" { SubShader { Pass {} } }"
            );
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"display_shader\":\"display.shader\"",
                    "\"display_shader\":\"display.shader\","
                        + "\"buffer_copy_shader\":\"buffer-copy.shader\""
                );
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));

            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(
                Path.GetFullPath(Path.Combine(_directory, "buffer-copy.shader")),
                artifact.BufferCopyShaderAssetPath
            );
        }

        [Test]
        public void ValidateRejectsMissingBufferCopyShader()
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"display_shader\":\"display.shader\"",
                    "\"display_shader\":\"display.shader\","
                        + "\"buffer_copy_shader\":\"missing.shader\""
                );
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));
            Assert.Throws<FileNotFoundException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateRejectsTraversalShaderPath()
        {
            WriteManifest("host.uasm", "../escape.shader", "3.7.0");
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateRejectsAvatarOrNonWorldSdkVersion()
        {
            WriteManifest("host.uasm", "kernel.shader", "2.0.0");
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateRejectsUnknownResourceKind()
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath);
            json =
                json.Substring(0, json.Length - 1)
                + ",\"resources\":[{\"name\":\"weights\",\"kind\":\"executable\",\"path\":\"weights.exe\"}]}";
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateAcceptsGenericOnnxModelAndExecutionPlanPair()
        {
            File.WriteAllBytes(Path.Combine(_directory, "model.onnx.bytes"), new byte[] { 8, 1 });
            File.WriteAllText(
                Path.Combine(_directory, "plan.bytes"),
                "{\"schema_version\":1,\"runtime\":\"udon-onnx\",\"execution_provider\":\"FixtureProvider\",\"passes\":[{\"id\":7,\"kernel\":\"fixture_kernel\",\"providers\":[\"udonblasSgemm\"],\"fragments\":16}]}"
            );
            AddResources(
                "[{\"name\":\"onnx_model\",\"kind\":\"bytes\",\"path\":\"model.onnx.bytes\"},{\"name\":\"onnx_execution_plan\",\"kind\":\"bytes\",\"path\":\"plan.bytes\"}]"
            );
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(2, artifact.Resources.Count);
        }

        [Test]
        public void ValidateRejectsUnpairedOnnxResource()
        {
            File.WriteAllBytes(Path.Combine(_directory, "model.onnx.bytes"), new byte[] { 8, 1 });
            AddResources(
                "[{\"name\":\"onnx_model\",\"kind\":\"bytes\",\"path\":\"model.onnx.bytes\"}]"
            );
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateAcceptsOnnxStaticInputTexture()
        {
            File.WriteAllBytes(Path.Combine(_directory, "scale.bytes"), new byte[8]);
            AddManifestProperty(
                "\"static_inputs\":[{\"name\":\"model.scale\",\"resource\":\"scale.bytes\","
                    + "\"texture\":{\"abi\":\"linear-rgba32-words\",\"words\":2,\"width\":2,\"height\":1}}]"
            );
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.OnnxStaticInputs.Count);
            Assert.AreEqual("model.scale", artifact.OnnxStaticInputs[0].Name);
            Assert.AreEqual(2, artifact.OnnxStaticInputs[0].Width);
        }

        [Test]
        public void ValidateAcceptsOnnxInt4ChildCall()
        {
            string child = Path.Combine(_directory, "calls", "call-0003");
            Directory.CreateDirectory(Path.Combine(child, "atlases"));
            File.WriteAllText(
                Path.Combine(child, "host.uasm"),
                ".data_start\n.data_end\n.code_start\n.code_end\n"
            );
            File.WriteAllBytes(Path.Combine(child, "atlases", "atlas-0000.rgba"), new byte[5120]);
            File.WriteAllText(
                Path.Combine(child, "manifest.json"),
                "{\"schema_version\":1,\"artifact_stage\":\"udon-onnx-int4-call\","
                    + "\"call_index\":3,\"function\":\"udonatenWeightInt4packMm\","
                    + "\"result_layout\":{\"width\":70,\"height\":1},"
                    + "\"assembly\":\"host.uasm\",\"resources\":[{\"name\":\"pageAtlas0\","
                    + "\"kind\":\"bytes\",\"path\":\"atlases/atlas-0000.rgba\","
                    + "\"encoding\":\"q4-bf16-row80-rgba32\",\"width\":1280,"
                    + "\"height\":1,\"bytes\":5120}]}"
            );
            AddManifestProperty(
                "\"call_artifacts\":[{\"call_index\":3,\"path\":\"calls/call-0003\"}]"
            );
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.OnnxInt4Calls.Count);
            Assert.AreEqual(3, artifact.OnnxInt4Calls[0].CallIndex);
            Assert.AreEqual(1, artifact.OnnxInt4Calls[0].Atlases.Count);
            Assert.AreEqual(70, artifact.OnnxInt4Calls[0].ResultWidth);
            Assert.AreEqual(1280, artifact.OnnxInt4Calls[0].Atlases[0].Width);
        }

        [Test]
        public void ValidateRejectsOnnxInt4ChildTraversal()
        {
            AddManifestProperty("\"call_artifacts\":[{\"call_index\":0,\"path\":\"../outside\"}]");
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [TestCase("hello-dispatch.zip.bytes", "hello_dispatch", "unity-udoncl-library", 1)]
        [TestCase("saxpy.zip.bytes", "saxpy_demo", "unity-udoncl-library", 1)]
        [TestCase("average-pooling.zip.bytes", "average_pooling", "unity-udoncl-library", 1)]
        [TestCase("onnx-arena-add.zip.bytes", "onnx_arena_add_example", "unity-udon-onnx-model", 1)]
        [TestCase(
            "onnx-tiny-llama.zip.bytes",
            "onnx_tiny_llama_example",
            "unity-udon-onnx-model",
            5
        )]
        public void PackagedExampleExtractsAndValidates(
            string archiveName,
            string programName,
            string artifactStage,
            int kernelCount
        )
        {
            string extracted = UdonclPackagedExampleImporter.Extract(archiveName);
            try
            {
                ValidatedArtifact artifact = ArtifactManifestValidator.Validate(extracted);
                Assert.AreEqual(programName, artifact.ProgramName);
                Assert.AreEqual(artifactStage, artifact.ArtifactStage);
                Assert.AreEqual(kernelCount, artifact.Kernels.Count);
                Assert.IsTrue(File.Exists(Path.Combine(extracted, "host.uasm")));
                if (artifactStage == "unity-udon-onnx-model")
                {
                    Assert.IsTrue(File.Exists(Path.Combine(extracted, "model.onnx.bytes")));
                    Assert.IsTrue(
                        File.Exists(Path.Combine(extracted, "execution-plan.json.bytes"))
                    );
                }
            }
            finally
            {
                Directory.Delete(extracted, true);
            }
        }

        [Test]
        public void PackagedHelloUsageCallsGeneratedTypedBinding()
        {
            PackagedExampleUsage usage = PackagedExampleUsageGenerator.ForProgram("hello_dispatch");
            Assert.NotNull(usage);
            Assert.AreEqual("UdonclHelloDispatchExample", usage.ClassName);
            StringAssert.Contains("UdonclHelloDispatchBinding binding", usage.Source);
            StringAssert.Contains("binding.TryDispatchSquare(input)", usage.Source);
            StringAssert.Contains("binding.StartReadOutput(output", usage.Source);
            StringAssert.DoesNotContain("SendCustomEvent", usage.Source);
        }

        [Test]
        public void PackagedSaxpyUsageWritesRunsAndReadsGeneratedBinding()
        {
            PackagedExampleUsage usage = PackagedExampleUsageGenerator.ForProgram("saxpy_demo");
            Assert.NotNull(usage);
            Assert.AreEqual("UdonclSaxpyDemoExample", usage.ClassName);
            StringAssert.Contains("binding.TryWriteInput(input)", usage.Source);
            StringAssert.Contains("binding.TryRunSaxpy(count, alpha)", usage.Source);
            StringAssert.Contains("binding.StartReadOutput(output", usage.Source);
            StringAssert.DoesNotContain("SetProgramVariable", usage.Source);
        }

        [Test]
        public void PackagedPoolingUsageWritesRunsAndReadsGeneratedBinding()
        {
            PackagedExampleUsage usage = PackagedExampleUsageGenerator.ForProgram(
                "average_pooling"
            );
            Assert.NotNull(usage);
            Assert.AreEqual("UdonclAveragePoolingExample", usage.ClassName);
            StringAssert.Contains("binding.TryWriteInput(input)", usage.Source);
            StringAssert.Contains("binding.TryRunAveragePool(72)", usage.Source);
            StringAssert.Contains("binding.StartReadOutput(output", usage.Source);
            StringAssert.DoesNotContain("SetProgramVariable", usage.Source);
        }

        [Test]
        public void OtherArtifactsDoNotReceivePackagedUsageCode()
        {
            Assert.IsNull(PackagedExampleUsageGenerator.ForProgram("user_library"));
        }

        private void AddResources(string resources)
        {
            AddManifestProperty("\"resources\":" + resources);
        }

        private void AddManifestProperty(string property)
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath);
            File.WriteAllText(
                manifestPath,
                json.Substring(0, json.Length - 1) + "," + property + "}",
                new UTF8Encoding(false)
            );
        }

        private void WriteManifest(string assemblyPath, string shaderPath, string sdkVersion)
        {
            string json =
                "{"
                + "\"schema_version\":1,"
                + "\"compiler_version\":\"0.1.0\","
                + "\"runtime_abi_version\":1,"
                + "\"artifact_stage\":\"unity-lj-prototype\","
                + "\"ready_for_unity\":true,"
                + "\"sdk_compatibility\":{\"sdk_version\":\""
                + sdkVersion
                + "\"},"
                + "\"target\":{\"host_pointer_bits\":32,\"device_pointer_bits\":32,\"endianness\":\"little\",\"platform\":\"VRChat-PC\"},"
                + "\"program\":{\"name\":\"lj_demo\",\"assembly\":\""
                + assemblyPath
                + "\",\"display_shader\":\"display.shader\"},"
                + "\"device\":{\"kernels\":[{\"id\":1,\"name\":\"lj_step\",\"shader\":\""
                + shaderPath
                + "\"}]}"
                + "}";
            File.WriteAllText(
                Path.Combine(_directory, "manifest.json"),
                json,
                new UTF8Encoding(false)
            );
        }
    }
}
