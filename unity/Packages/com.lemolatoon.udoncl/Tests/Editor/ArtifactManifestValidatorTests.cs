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
        public void ValidateRejectsMissingRuntimeValidationAudit()
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"artifact_stage\":\"unity-lj-prototype\"",
                    "\"artifact_stage\":\"unity-udoncl-library\""
                )
                .Replace("\"validation\":{\"read_write_alias\":false},", string.Empty);
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));

            InvalidDataException exception = Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
            StringAssert.Contains("validation.read_write_alias", exception.Message);
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
        public void ValidateAcceptsJsonDiagnosticResource()
        {
            File.WriteAllText(
                Path.Combine(_directory, "storage-report.json"),
                "{\"version\":2}",
                new UTF8Encoding(false)
            );
            AddResources(
                "[{\"name\":\"host_storage_report\",\"kind\":\"json\","
                    + "\"path\":\"storage-report.json\",\"bytes\":13}]"
            );

            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.Resources.Count);
            Assert.AreEqual("json", artifact.Resources[0].Kind);
        }

        [Test]
        public void ValidateAcceptsCompilerBindingAbiMetadata()
        {
            WriteCompilerBindingManifest(
                "89dd22ba50e6713d0758a5c6aae2dfeeeeb64f6bc0c3754a92302c092a20c751",
                "udonsharp-worlds-sdk-3.10.5"
            );
            Assert.DoesNotThrow(() => ArtifactManifestValidator.Validate(_directory));
        }

        [Test]
        public void ValidateRejectsInvalidCompilerBindingAbiMetadata()
        {
            WriteCompilerBindingManifest(new string('A', 64), "udonsharp-worlds-sdk-3.10.5");
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );

            WriteManifest("host.uasm", "kernel.shader", "3.7.0");
            WriteCompilerBindingManifest(new string('a', 64), "unsupported-abi");
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateRejectsCompilerBindingSourceIdentityMismatch()
        {
            const string fingerprint =
                "89dd22ba50e6713d0758a5c6aae2dfeeeeb64f6bc0c3754a92302c092a20c751";
            WriteCompilerBindingManifest(fingerprint, "udonsharp-worlds-sdk-3.10.5");
            string bindingPath = Path.Combine(_directory, "binding.cs");
            File.WriteAllText(
                bindingPath,
                CompilerBindingSource(fingerprint, "Fixture", "WrongBinding"),
                new UTF8Encoding(false)
            );

            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateFingerprintsCompilerGeneratedPublicBufferApi()
        {
            WriteCompilerBindingManifest(
                "a8300ca82e5f5b684d9fdbc28ab0c30a7a53dc924e73bccecff3216c3090607e",
                "udonsharp-worlds-sdk-3.10.5"
            );
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"device\":{\"kernels\"",
                    "\"device\":{\"buffers\":[{\"id\":7,\"name\":\"input\","
                        + "\"heap_slot\":\"__udoncl_buffer_7\",\"logical_bytes\":16,"
                        + "\"width\":4,\"height\":1,\"format\":\"rgba8_unorm\","
                        + "\"srgb\":false,\"mipmaps\":false,\"filter\":\"point\","
                        + "\"element_type\":\"float32\",\"public_access\":\"input\"}],"
                        + "\"kernels\""
                );
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));
            Assert.DoesNotThrow(() => ArtifactManifestValidator.Validate(_directory));

            File.WriteAllText(
                manifestPath,
                json.Replace("\"logical_bytes\":16", "\"logical_bytes\":12"),
                new UTF8Encoding(false)
            );
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateAcceptsZeroLengthPublicBufferWithPhysicalCapacity()
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"device\":{\"kernels\"",
                    "\"device\":{\"buffers\":[{\"id\":7,\"name\":\"empty_input\","
                        + "\"heap_slot\":\"__udoncl_buffer_7\",\"logical_bytes\":0,"
                        + "\"width\":1,\"height\":1,\"format\":\"rgba8_unorm\","
                        + "\"srgb\":false,\"mipmaps\":false,\"filter\":\"point\","
                        + "\"element_type\":\"float32\",\"public_access\":\"input\"}],"
                        + "\"kernels\""
                );
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));

            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.DeviceBuffers.Count);
            Assert.AreEqual(0, artifact.DeviceBuffers[0].LogicalBytes);
        }

        [Test]
        public void ValidateAcceptsWordTexelBuffersWithTransferScratch()
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string buffers =
                "\"device\":{\"buffers\":[{\"id\":7,\"name\":\"words\","
                + "\"heap_slot\":\"__udoncl_buffer_7\",\"logical_bytes\":64,"
                + "\"width\":4,\"height\":1,\"format\":\"rgba32_float\","
                + "\"srgb\":false,\"mipmaps\":false,\"filter\":\"point\"}],"
                + "\"kernels\"";
            string program =
                "\"display_shader\":\"display.shader\",\"texel_format\":\"rgba32_float\","
                + "\"transfer_scratch\":{\"slot\":\"__udoncl_transfer_scratch\","
                + "\"width\":4,\"height\":1,\"format\":\"rgba32_float\",\"program_slot\":true}";
            string original = File.ReadAllText(manifestPath);
            string json = original
                .Replace("\"device\":{\"kernels\"", buffers)
                .Replace("\"display_shader\":\"display.shader\"", program);
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));

            // Four words per texel: 4 texels hold the 64 bytes.
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual("rgba32_float", artifact.TexelFormat);
            Assert.AreEqual("rgba32_float", artifact.DeviceBuffers[0].Format);
            Assert.IsNotNull(artifact.TransferScratch);
            Assert.IsTrue(artifact.TransferScratch.ProgramSlot);

            // A word-texel program needs its transfer scratch texture.
            File.WriteAllText(
                manifestPath,
                original
                    .Replace("\"device\":{\"kernels\"", buffers)
                    .Replace(
                        "\"display_shader\":\"display.shader\"",
                        "\"display_shader\":\"display.shader\",\"texel_format\":\"rgba32_float\""
                    ),
                new UTF8Encoding(false)
            );
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );

            // Every buffer uses the program's texel format.
            File.WriteAllText(
                manifestPath,
                original.Replace("\"device\":{\"kernels\"", buffers),
                new UTF8Encoding(false)
            );
            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
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
            File.WriteAllBytes(Path.Combine(_directory, "scale.bytes"), new byte[8192 * 4]);
            AddManifestProperty(
                "\"static_inputs\":[{\"name\":\"model.scale\",\"resource\":\"scale.bytes\","
                    + "\"texture\":{\"abi\":\"linear-rgba32-words\",\"words\":2,\"width\":8192,\"height\":1}}]"
            );
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.OnnxStaticInputs.Count);
            Assert.AreEqual("model.scale", artifact.OnnxStaticInputs[0].Name);
            Assert.AreEqual(8192, artifact.OnnxStaticInputs[0].Width);
        }

        [Test]
        public void ValidateAcceptsImmutableStaticTensor()
        {
            File.WriteAllBytes(Path.Combine(_directory, "weight.bytes"), new byte[] { 1, 0, 0, 0 });
            AddResources(
                "[{\"name\":\"static_tensor_0\",\"kind\":\"bytes\","
                    + "\"path\":\"weight.bytes\",\"bytes\":4}]"
            );
            AddManifestProperty(
                "\"static_tensors\":[{\"id\":0,\"name\":\"weight\","
                    + "\"resource\":\"static_tensor_0\",\"data_type\":\"int32\","
                    + "\"shape\":[1],\"layout\":\"row_major_contiguous\","
                    + "\"alignment\":4,\"byte_strides\":[4],\"byte_length\":4,"
                    + "\"sha256\":\"67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450\","
                    + "\"texture\":{\"abi\":\"udoncl-static-rgba32-v1\","
                    + "\"width\":1,\"height\":1,\"format\":\"rgba32\","
                    + "\"srgb\":false,\"mipmaps\":false,\"filter\":\"point\"}}]"
            );
            AddProgramProperty("\"static_tensor_table_slot\":\"__udoncl_static_tensors\"");

            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.StaticTensors.Count);
            Assert.AreEqual("weight", artifact.StaticTensors[0].Name);
            Assert.AreEqual("int32", artifact.StaticTensors[0].DataType);
            Assert.AreEqual(4, artifact.StaticTensors[0].ByteLength);
            Assert.AreEqual(4, artifact.StaticTensors[0].Alignment);
            CollectionAssert.AreEqual(new long[] { 4 }, artifact.StaticTensors[0].ByteStrides);
            Assert.AreEqual(1, artifact.StaticTensors[0].Width);
            Assert.AreEqual(1, artifact.StaticTensors[0].Height);
        }

        [Test]
        public void ValidateAcceptsMultiRowImmutableStaticTensor()
        {
            File.WriteAllBytes(Path.Combine(_directory, "weight.bytes"), new byte[33792]);
            AddResources(
                "[{\"name\":\"static_tensor_0\",\"kind\":\"bytes\","
                    + "\"path\":\"weight.bytes\",\"bytes\":33792}]"
            );
            AddManifestProperty(
                "\"static_tensors\":[{\"id\":0,\"name\":\"weight\","
                    + "\"resource\":\"static_tensor_0\",\"data_type\":\"int8\","
                    + "\"shape\":[33,64,16],\"layout\":\"row_major_contiguous\","
                    + "\"alignment\":1,\"byte_strides\":[1024,16,1],"
                    + "\"byte_length\":33792,\"sha256\":"
                    + "\"904cc818da7f8339cc5ea26ae49ec7f7d33bbca04f63902ea9c2bcd6012532a7\","
                    + "\"texture\":{\"abi\":\"udoncl-static-rgba32-v1\","
                    + "\"width\":8192,\"height\":2,\"format\":\"rgba32\","
                    + "\"srgb\":false,\"mipmaps\":false,\"filter\":\"point\"}}]"
            );
            AddProgramProperty("\"static_tensor_table_slot\":\"__udoncl_static_tensors\"");

            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(_directory);
            Assert.AreEqual(1, artifact.StaticTensors.Count);
            Assert.AreEqual(33792, artifact.StaticTensors[0].ByteLength);
            Assert.AreEqual(8192, artifact.StaticTensors[0].Width);
            Assert.AreEqual(2, artifact.StaticTensors[0].Height);
        }

        [Test]
        public void ValidateRejectsStaticTensorTableSlotWithoutTensors()
        {
            AddProgramProperty("\"static_tensor_table_slot\":\"__udoncl_static_tensors\"");

            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
        }

        [Test]
        public void ValidateRejectsImmutableStaticTensorHashMismatch()
        {
            File.WriteAllBytes(Path.Combine(_directory, "weight.bytes"), new byte[] { 1, 0, 0, 0 });
            AddResources(
                "[{\"name\":\"static_tensor_0\",\"kind\":\"bytes\","
                    + "\"path\":\"weight.bytes\",\"bytes\":4}]"
            );
            AddManifestProperty(
                "\"static_tensors\":[{\"id\":0,\"name\":\"weight\","
                    + "\"resource\":\"static_tensor_0\",\"data_type\":\"int32\","
                    + "\"shape\":[1],\"layout\":\"row_major_contiguous\","
                    + "\"alignment\":4,\"byte_strides\":[4],"
                    + "\"byte_length\":4,\"sha256\":\""
                    + new string('0', 64)
                    + "\",\"texture\":{"
                    + "\"abi\":\"udoncl-static-rgba32-v1\",\"width\":1,"
                    + "\"height\":1,\"format\":\"rgba32\",\"srgb\":false,"
                    + "\"mipmaps\":false,\"filter\":\"point\"}}]"
            );

            Assert.Throws<InvalidDataException>(() =>
                ArtifactManifestValidator.Validate(_directory)
            );
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
            StringAssert.Contains("binding.TryWriteX(x)", usage.Source);
            StringAssert.Contains("binding.TryWriteY(y)", usage.Source);
            StringAssert.Contains("binding.StartRunSaxpy(count, alpha, this", usage.Source);
            StringAssert.Contains("binding.GetRunSaxpyStatus()", usage.Source);
            StringAssert.Contains("binding.TryGetRunSaxpyResult(out openclStatus)", usage.Source);
            StringAssert.Contains("binding.StartReadResult(result", usage.Source);
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
        public void PackagedUdonBlasUsageAwaitsAndReadsGeneratedBinding()
        {
            PackagedExampleUsage usage = PackagedExampleUsageGenerator.ForProgram(
                "udonblas_sgemm_m1n10k144"
            );
            Assert.NotNull(usage);
            Assert.AreEqual("UdonclUdonBlasSgemmExample", usage.ClassName);
            StringAssert.Contains("binding.StartRunUdonBlasSgemm(this", usage.Source);
            StringAssert.Contains("binding.GetRunUdonBlasSgemmStatus()", usage.Source);
            StringAssert.Contains(
                "binding.TryGetRunUdonBlasSgemmResult(out libraryStatus)",
                usage.Source
            );
            StringAssert.Contains(
                "binding.TryGetUdonBlasSgemmPhase(out libraryPhase)",
                usage.Source
            );
            StringAssert.Contains(
                "binding.TryGetUdonBlasSgemmResult(column, out value)",
                usage.Source
            );
            StringAssert.Contains("1224.0f * (column + 1)", usage.Source);
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

        private void AddProgramProperty(string property)
        {
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"display_shader\":\"display.shader\"",
                    "\"display_shader\":\"display.shader\"," + property
                );
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));
        }

        private void WriteCompilerBindingManifest(string fingerprint, string physicalAbi)
        {
            File.WriteAllText(
                Path.Combine(_directory, "binding.cs"),
                CompilerBindingSource(fingerprint, "Fixture", "Binding"),
                new UTF8Encoding(false)
            );
            string manifestPath = Path.Combine(_directory, "manifest.json");
            string json = File.ReadAllText(manifestPath)
                .Replace(
                    "\"runtime_abi_version\":1,",
                    "\"runtime_abi_version\":1,"
                        + "\"public_abi_fingerprint\":\""
                        + fingerprint
                        + "\",\"physical_export_abi\":\""
                        + physicalAbi
                        + "\","
                )
                .Replace(
                    "\"display_shader\":\"display.shader\"",
                    "\"display_shader\":\"display.shader\","
                        + "\"binding_source\":\"binding.cs\","
                        + "\"binding_namespace\":\"Fixture\","
                        + "\"binding_class\":\"Binding\""
                )
                .Replace(
                    "\"device\":",
                    "\"exports\":[{"
                        + "\"id\":1,\"public_name\":\"Probe\","
                        + "\"udon_event\":\"Probe\",\"parameters\":[],"
                        + "\"result\":{\"kind\":\"i32\","
                        + "\"heap_slot\":\"__0_Probe__ret\"},"
                        + "\"status_slot\":\"__udoncl_status_1\","
                        + "\"async\":false}],\"device\":"
                );
            File.WriteAllText(manifestPath, json, new UTF8Encoding(false));
        }

        private static string CompilerBindingSource(
            string fingerprint,
            string namespaceName,
            string className
        )
        {
            return "// <auto-generated />\n"
                + "// udoncl-public-abi: "
                + fingerprint
                + "\n// udoncl-binding-namespace: "
                + namespaceName
                + "\n// udoncl-binding-class: "
                + className
                + "\nnamespace "
                + namespaceName
                + " { public sealed class "
                + className
                + " {} }";
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
                + "\"validation\":{\"read_write_alias\":false},"
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
