using System;
using System.IO;
using Lemolatoon.Udoncl.Editor;
using NUnit.Framework;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class OnnxOracleSidecarTests
    {
        private string directory;

        [SetUp]
        public void SetUp()
        {
            directory = Path.Combine(Path.GetTempPath(), "udoncl-onnx-oracle-" + Guid.NewGuid());
            Directory.CreateDirectory(directory);
            File.WriteAllBytes(
                Path.Combine(directory, "output-000.f32.bytes"),
                Floats(1.0f, -2.0f, 3.5f, 4.0f)
            );
            File.WriteAllBytes(
                Path.Combine(directory, "output-001.f32.bytes"),
                Floats(0.25f, 0.5f)
            );
            File.WriteAllText(
                Path.Combine(directory, "oracle.json"),
                "{\n"
                    + "  \"schema\": \"udoncl.onnx-oracle.v1\",\n"
                    + "  \"model_sha256\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\n"
                    + "  \"inputs\": [{\"dtype\":\"int64\",\"name\":\"ids\",\"shape\":[1,2],\"values\":[4,5]}],\n"
                    + "  \"outputs\": [\n"
                    + "    {\"dtype\":\"float32\",\"element_count\":4,\"file\":\"output-000.f32.bytes\",\"index\":0,\"name\":\"logits\",\"shape\":[1,2,2]},\n"
                    + "    {\"dtype\":\"float32\",\"element_count\":2,\"file\":\"output-001.f32.bytes\",\"index\":1,\"name\":\"present\",\"shape\":[2]}\n"
                    + "  ]\n"
                    + "}\n"
            );
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, true);
        }

        [Test]
        public void LoadsTypedInputsAndOrderedFloatOutputs()
        {
            ValidatedOnnxOracle oracle = OnnxOracleSidecar.Load(directory);
            Assert.AreEqual(1, oracle.Inputs.Count);
            Assert.AreEqual("ids", oracle.Inputs[0].Name);
            CollectionAssert.AreEqual(new long[] { 4, 5 }, oracle.Inputs[0].ReadInt64Values());
            Assert.AreEqual(2, oracle.Outputs.Count);
            Assert.AreEqual("logits", oracle.Outputs[0].Name);
            CollectionAssert.AreEqual(
                new float[] { 1.0f, -2.0f, 3.5f, 4.0f },
                oracle.ReadFloat32Output(0)
            );
            oracle.RequireFloat32Close(
                1,
                new float[] { 99.0f, 0.25001f, 0.49999f },
                1,
                0.00002f,
                0.0f
            );
        }

        [Test]
        public void RejectsUnorderedOutputsAndMismatchedValues()
        {
            string manifest = File.ReadAllText(Path.Combine(directory, "oracle.json"));
            File.WriteAllText(
                Path.Combine(directory, "oracle.json"),
                manifest.Replace("\"index\":1", "\"index\":7")
            );
            Assert.Throws<InvalidDataException>(() => OnnxOracleSidecar.Load(directory));

            File.WriteAllText(Path.Combine(directory, "oracle.json"), manifest);
            ValidatedOnnxOracle oracle = OnnxOracleSidecar.Load(directory);
            Assert.Throws<InvalidDataException>(() =>
                oracle.RequireFloat32Close(1, new float[] { 0.25f, 0.75f }, 0.0001f, 0.0f)
            );
        }

        [Test]
        public void FindsPublicBuffersByNameNotPosition()
        {
            // Device allocation order, not ONNX order.
            ValidatedDeviceBuffer[] buffers =
            {
                Buffer(1, "input_ids", "int64", "input"),
                Buffer(2, "past_key_values_0_key", "float32", "input"),
                Buffer(3, "position_ids", "int64", "input"),
                Buffer(4, "private_4", null, null),
                Buffer(5, "present_0_key", "float32", "output"),
                Buffer(6, "tensor_0", "float32", "output"),
                Buffer(7, "a_b", "float32", "input"),
                Buffer(8, "a_b_input_5", "float32", "input"),
            };
            Assert.AreEqual(
                "past_key_values_0_key",
                OnnxPublicBuffers.BufferName("past_key_values.0.key")
            );
            Assert.AreEqual("tensor_0", OnnxPublicBuffers.BufferName("0"));
            Assert.AreEqual("_x_y", OnnxPublicBuffers.BufferName("_x-y"));
            Assert.AreEqual(3, OnnxPublicBuffers.Find(buffers, "input", 1, "position_ids").Id);
            Assert.AreEqual(
                2,
                OnnxPublicBuffers.Find(buffers, "input", 2, "past_key_values.0.key").Id
            );
            Assert.AreEqual(5, OnnxPublicBuffers.Find(buffers, "output", 1, "present.0.key").Id);
            Assert.AreEqual(6, OnnxPublicBuffers.Find(buffers, "output", 0, "0").Id);
            // A colliding name carries the access and ONNX index.
            Assert.AreEqual(7, OnnxPublicBuffers.Find(buffers, "input", 4, "a.b").Id);
            Assert.AreEqual(8, OnnxPublicBuffers.Find(buffers, "input", 5, "a_b").Id);
            // The access is part of the match.
            Assert.Throws<InvalidDataException>(() =>
                OnnxPublicBuffers.Find(buffers, "output", 0, "input_ids")
            );
            Assert.Throws<InvalidDataException>(() =>
                OnnxPublicBuffers.Find(buffers, "input", 9, "attention_mask")
            );
        }

        private static ValidatedDeviceBuffer Buffer(
            int id,
            string name,
            string elementType,
            string access
        )
        {
            return new ValidatedDeviceBuffer(
                id,
                name,
                "__udoncl_buffer_" + id,
                4,
                1,
                1,
                elementType,
                access
            );
        }

        private static byte[] Floats(params float[] values)
        {
            byte[] bytes = new byte[values.Length * sizeof(float)];
            for (int index = 0; index < values.Length; ++index)
                Array.Copy(
                    BitConverter.GetBytes(values[index]),
                    0,
                    bytes,
                    index * sizeof(float),
                    4
                );
            return bytes;
        }
    }
}
