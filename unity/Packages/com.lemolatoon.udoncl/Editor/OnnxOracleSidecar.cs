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
            if (actual == null)
                throw new ArgumentNullException("actual");
            if (absoluteTolerance < 0.0f || relativeTolerance < 0.0f)
                throw new ArgumentOutOfRangeException("absoluteTolerance");
            float[] expected = ReadFloat32Output(index);
            if (actual.Length != expected.Length)
                throw new InvalidDataException(
                    "ONNX oracle output "
                        + index
                        + " has "
                        + expected.Length
                        + " elements, but the actual output has "
                        + actual.Length
                        + "."
                );
            for (int element = 0; element < expected.Length; ++element)
            {
                float expectedValue = expected[element];
                float actualValue = actual[element];
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
