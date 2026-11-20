using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor
{
    internal sealed class ValidatedOnnxOracle
    {
        private readonly string rootDirectory;

        internal ValidatedOnnxOracle(
            string rootDirectory,
            string modelSha256,
            IList<ValidatedOnnxOracleInput> inputs,
            IList<ValidatedOnnxOracleOutput> outputs
        )
        {
            this.rootDirectory = rootDirectory;
            ModelSha256 = modelSha256;
            Inputs = inputs;
            Outputs = outputs;
        }

        internal string ModelSha256 { get; private set; }
        internal IList<ValidatedOnnxOracleInput> Inputs { get; private set; }
        internal IList<ValidatedOnnxOracleOutput> Outputs { get; private set; }

        internal float[] ReadFloat32Output(int index)
        {
            if (index < 0 || index >= Outputs.Count)
                throw new ArgumentOutOfRangeException("index");
            ValidatedOnnxOracleOutput output = Outputs[index];
            byte[] bytes = File.ReadAllBytes(Path.Combine(rootDirectory, output.File));
            float[] values = new float[output.ElementCount];
            for (int element = 0; element < values.Length; ++element)
                values[element] = BitConverter.ToSingle(bytes, element * sizeof(float));
            return values;
        }

        internal void RequireFloat32Close(
            int index,
            float[] actual,
            float absoluteTolerance,
            float relativeTolerance
        )
        {
            RequireFloat32Close(index, actual, 0, absoluteTolerance, relativeTolerance);
        }

        internal void RequireFloat32Close(
            int index,
            float[] actual,
            int actualOffset,
            float absoluteTolerance,
            float relativeTolerance
        )
        {
            if (actual == null)
                throw new ArgumentNullException("actual");
            if (actualOffset < 0)
                throw new ArgumentOutOfRangeException("actualOffset");
            if (absoluteTolerance < 0.0f || relativeTolerance < 0.0f)
                throw new ArgumentOutOfRangeException("absoluteTolerance");
            float[] expected = ReadFloat32Output(index);
            if (actualOffset > actual.Length - expected.Length)
                throw new InvalidDataException(
                    "ONNX oracle output "
                        + index
                        + " has "
                        + expected.Length
                        + " elements, but the requested actual range is unavailable."
                );
            for (int element = 0; element < expected.Length; ++element)
            {
                float expectedValue = expected[element];
                float actualValue = actual[actualOffset + element];
                float tolerance = absoluteTolerance + relativeTolerance * Math.Abs(expectedValue);
                if (
                    float.IsNaN(actualValue)
                    || float.IsInfinity(actualValue)
                    || Math.Abs(actualValue - expectedValue) > tolerance
                )
                    throw new InvalidDataException(
                        "ONNX oracle output "
                            + index
                            + " differs at element "
                            + element
                            + ": expected "
                            + expectedValue
                            + ", actual "
                            + actualValue
                            + "."
                    );
            }
        }
    }

    internal sealed class ValidatedOnnxOracleInput
    {
        internal ValidatedOnnxOracleInput(
            string name,
            string dataType,
            long[] shape,
            double[] values
        )
        {
            Name = name;
            DataType = dataType;
            Shape = shape;
            Values = values;
        }

        internal string Name { get; private set; }
        internal string DataType { get; private set; }
        internal long[] Shape { get; private set; }
        internal double[] Values { get; private set; }

        internal long[] ReadInt64Values()
        {
            if (DataType != "int64")
                throw new InvalidDataException("The ONNX oracle input is not INT64.");
            long[] result = new long[Values.Length];
            for (int index = 0; index < Values.Length; ++index)
            {
                double value = Values[index];
                long converted = (long)value;
                if (double.IsNaN(value) || double.IsInfinity(value) || converted != value)
                    throw new InvalidDataException("An ONNX oracle INT64 input value is invalid.");
                result[index] = converted;
            }
            return result;
        }
    }

    internal sealed class ValidatedOnnxOracleOutput
    {
        internal ValidatedOnnxOracleOutput(
            int index,
            string name,
            string file,
            int elementCount,
            long[] shape
        )
        {
            Index = index;
            Name = name;
            File = file;
            ElementCount = elementCount;
            Shape = shape;
        }

        internal int Index { get; private set; }
        internal string Name { get; private set; }
        internal string File { get; private set; }
        internal int ElementCount { get; private set; }
        internal long[] Shape { get; private set; }
    }

    // Pairs ONNX tensors with the public Device buffers of an artifact by
    // name. The manifest lists Device buffers in allocation order, which is
    // not the ONNX input/output order (a banked sequence state moves its
    // past inputs, for one), so a caller never pairs them by position.
    internal static class OnnxPublicBuffers
    {
        // Mirrors onnxPublicBufferName in tools/vrccl/vrccl.cpp: every
        // character outside [A-Za-z0-9_] becomes '_', and a name that does not
        // start with a letter or '_' gets the prefix "tensor_".
        internal static string BufferName(string tensorName)
        {
            if (tensorName == null)
                throw new ArgumentNullException("tensorName");
            char[] characters = tensorName.ToCharArray();
            for (int index = 0; index < characters.Length; ++index)
            {
                char character = characters[index];
                if (!IsAsciiLetter(character) && !IsAsciiDigit(character) && character != '_')
                    characters[index] = '_';
            }
            string name = new string(characters);
            if (name.Length == 0 || !(IsAsciiLetter(name[0]) || name[0] == '_'))
                name = "tensor_" + name;
            return name;
        }

        // The public buffer of the ONNX `access` ("input" or "output") tensor
        // `index` named `tensorName`. vrccl gives a name that collides with an
        // earlier public buffer the suffix "_<access>_<index>".
        internal static ValidatedDeviceBuffer Find(
            IList<ValidatedDeviceBuffer> buffers,
            string access,
            int index,
            string tensorName
        )
        {
            if (buffers == null)
                throw new ArgumentNullException("buffers");
            if (access != "input" && access != "output")
                throw new ArgumentException("access must be input or output.", "access");
            string name = BufferName(tensorName);
            string suffixed = name + "_" + access + "_" + index;
            ValidatedDeviceBuffer plain = null;
            foreach (ValidatedDeviceBuffer buffer in buffers)
            {
                if (buffer == null || buffer.PublicAccess != access)
                    continue;
                if (buffer.Name == suffixed)
                    return buffer;
                if (buffer.Name == name)
                    plain = buffer;
            }
            if (plain == null)
                throw new InvalidDataException(
                    "The artifact has no public "
                        + access
                        + " buffer '"
                        + name
                        + "' for ONNX tensor '"
                        + tensorName
                        + "'."
                );
            return plain;
        }

        private static bool IsAsciiLetter(char character)
        {
            return (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z');
        }

        private static bool IsAsciiDigit(char character)
        {
            return character >= '0' && character <= '9';
        }
    }

    internal static class OnnxOracleSidecar
    {
        internal static ValidatedOnnxOracle Load(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
                throw new ArgumentException("An ONNX oracle directory is required.", "directory");
            string root = Path.GetFullPath(directory);
            string manifestPath = Path.Combine(root, "oracle.json");
            if (!File.Exists(manifestPath))
                throw new FileNotFoundException(
                    "The ONNX oracle manifest is missing.",
                    manifestPath
                );
            OnnxOracleManifest manifest = JsonUtility.FromJson<OnnxOracleManifest>(
                File.ReadAllText(manifestPath)
            );
            if (manifest == null || manifest.schema != "udoncl.onnx-oracle.v1")
                throw new InvalidDataException("Unsupported ONNX oracle schema.");
            if (!IsSha256(manifest.model_sha256))
                throw new InvalidDataException("The ONNX oracle model hash is invalid.");
            if (manifest.inputs == null || manifest.outputs == null)
                throw new InvalidDataException("The ONNX oracle input/output tables are required.");

            List<ValidatedOnnxOracleInput> inputs = new List<ValidatedOnnxOracleInput>();
            foreach (OnnxOracleInput input in manifest.inputs)
            {
                if (input == null || string.IsNullOrEmpty(input.name) || input.shape == null)
                    throw new InvalidDataException("An ONNX oracle input is incomplete.");
                if (string.IsNullOrEmpty(input.dtype) || input.values == null)
                    throw new InvalidDataException(
                        "A typed ONNX oracle input value table is required."
                    );
                RequireElementCount(input.shape, input.values.LongLength, "input " + input.name);
                inputs.Add(
                    new ValidatedOnnxOracleInput(input.name, input.dtype, input.shape, input.values)
                );
            }

            List<ValidatedOnnxOracleOutput> outputs = new List<ValidatedOnnxOracleOutput>();
            for (int index = 0; index < manifest.outputs.Length; ++index)
            {
                OnnxOracleOutput output = manifest.outputs[index];
                if (
                    output == null
                    || output.index != index
                    || string.IsNullOrEmpty(output.name)
                    || output.dtype != "float32"
                    || output.shape == null
                    || output.element_count < 0
                )
                    throw new InvalidDataException(
                        "An ONNX oracle output is incomplete or unordered."
                    );
                if (
                    string.IsNullOrEmpty(output.file)
                    || output.file != Path.GetFileName(output.file)
                )
                    throw new InvalidDataException(
                        "An ONNX oracle output file must be a leaf name."
                    );
                RequireElementCount(output.shape, output.element_count, "output " + output.name);
                string outputPath = Path.Combine(root, output.file);
                if (!File.Exists(outputPath))
                    throw new FileNotFoundException(
                        "An ONNX oracle output is missing.",
                        outputPath
                    );
                long expectedBytes = (long)output.element_count * sizeof(float);
                if (new FileInfo(outputPath).Length != expectedBytes)
                    throw new InvalidDataException(
                        "An ONNX oracle output has an invalid byte length."
                    );
                outputs.Add(
                    new ValidatedOnnxOracleOutput(
                        index,
                        output.name,
                        output.file,
                        output.element_count,
                        output.shape
                    )
                );
            }
            return new ValidatedOnnxOracle(root, manifest.model_sha256, inputs, outputs);
        }

        private static void RequireElementCount(long[] shape, long expected, string label)
        {
            long count = 1;
            foreach (long dimension in shape)
            {
                if (dimension < 0 || (dimension != 0 && count > long.MaxValue / dimension))
                    throw new InvalidDataException(
                        "The ONNX oracle " + label + " shape is invalid."
                    );
                count *= dimension;
            }
            if (count != expected)
                throw new InvalidDataException(
                    "The ONNX oracle " + label + " shape is inconsistent."
                );
        }

        private static bool IsSha256(string value)
        {
            if (value == null || value.Length != 64)
                return false;
            foreach (char character in value)
            {
                if ((character < '0' || character > '9') && (character < 'a' || character > 'f'))
                    return false;
            }
            return true;
        }

        [Serializable]
        private sealed class OnnxOracleManifest
        {
            public string schema;
            public string model_sha256;
            public OnnxOracleInput[] inputs;
            public OnnxOracleOutput[] outputs;
        }

        [Serializable]
        private sealed class OnnxOracleInput
        {
            public string dtype;
            public string name;
            public long[] shape;
            public double[] values;
        }

        [Serializable]
        private sealed class OnnxOracleOutput
        {
            public string dtype;
            public int element_count;
            public string file;
            public int index;
            public string name;
            public long[] shape;
        }
    }
}
