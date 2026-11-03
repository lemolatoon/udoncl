using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.Editor.ProgramSources;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    /// <summary>
    /// Executes the public ONNX MLIR -> UdonDNN/UdonBLAS MNIST artifact on the
    /// SDK VM, including its real texture uploads, Device dispatches, and
    /// readbacks. This intentionally does not use the legacy execution-plan
    /// scheduler.
    /// </summary>
    public sealed class PublicMnistOnnxRuntimeLifecycleTests
    {
        private const string GeneratedRoot = "Assets/udoncl-generated/onnx_mnist";
        private const string RunEntry = "RunMnist";
        private string repository;
        private string artifactDirectory;
        private string serializedProgramPath;
        private HashSet<string> serializedProgramsBefore;

        [OneTimeSetUp]
        public void ImportPublicMnistArtifact()
        {
            repository = ResolveRepository();
            serializedProgramsBefore = SerializedProgramAssets();
            AssetDatabase.DeleteAsset(GeneratedRoot);
            artifactDirectory = ResolvePath(
                "UDONCL_PUBLIC_MNIST_UNITY_BUNDLE",
                Path.Combine(
                    "build",
                    "udoncl",
                    "tests",
                    "cmake-package",
                    "onnx-mnist-consumer",
                    "udoncl",
                    "onnx-mnist",
                    "artifact"
                )
            );
            Assert.That(File.Exists(Path.Combine(artifactDirectory, "manifest.json")));
            Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));

            string programPath = GeneratedRoot + "/Programs/host_program.asset";
            string programGuid = AssetDatabase.AssetPathToGUID(programPath);
            Assert.IsFalse(string.IsNullOrEmpty(programGuid));
            serializedProgramPath = "Assets/SerializedUdonPrograms/" + programGuid + ".asset";
        }

        [Test]
        public void PublicOnnxRuntimeInfersMnistAndMatchesReference()
        {
            UdonAssemblyProgramAsset asset = Load<UdonAssemblyProgramAsset>(
                "Programs/host_program.asset"
            );
            IUdonProgram program = asset.SerializedProgramAsset.RetrieveProgram();
            Assert.NotNull(program);
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(vm);
            Assert.IsTrue(vm.LoadProgram(program));
            IUdonHeap heap = vm.InspectHeap();

            byte[] rawHeapBytes = File.ReadAllBytes(
                Path.Combine(artifactDirectory, "raw-heap.bytes")
            );
            int[] rawHeap = new int[rawHeapBytes.Length];
            for (int index = 0; index < rawHeapBytes.Length; ++index)
                rawHeap[index] = rawHeapBytes[index];
            Set(heap, program, "__vrc_raw_heap", rawHeap);

            Dictionary<int, RenderTexture> buffers = new Dictionary<int, RenderTexture>();
            string[] bufferNames =
            {
                "dnn_pointwise_self",
                "dnn_pointwise_other",
                "dnn_pointwise_output",
                "dnn_rms_arena_partial",
                "blas_input",
                "blas_output",
            };
            for (int index = 0; index < bufferNames.Length; ++index)
            {
                int id = index + 1;
                RenderTexture buffer = Load<RenderTexture>(
                    "RenderTextures/"
                        + UdonclArtifactImporter.DeviceBufferFileName(id, bufferNames[index])
                );
                buffers.Add(id, buffer);
                Set(heap, program, "__udoncl_buffer_" + id, buffer);
                Clear(buffer);
            }

            int[] kernelIds = { 232, 234, 233, 231, 1 };
            string[] kernelNames =
            {
                "udondnn_large_convolution_forward",
                "udondnn_large_op_tensor",
                "udondnn_large_activation_forward",
                "udondnn_large_pooling_forward",
                "udonblas_sgemm",
            };
            Dictionary<int, Material> materials = new Dictionary<int, Material>();
            for (int index = 0; index < kernelIds.Length; ++index)
            {
                Material material = Load<Material>(
                    "Materials/kernel_" + kernelIds[index] + "_" + kernelNames[index] + ".mat"
                );
                materials.Add(kernelIds[index], material);
                Set(heap, program, "__udoncl_kernel_material_" + kernelIds[index], material);
            }
            Material copyMaterial = Load<Material>("Materials/buffer-copy.mat");

            float[] input = ReadFloats(
                ResolvePath(
                    "UDONCL_ONNX_MNIST_INPUT",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-input.bytes")
                )
            );
            Assert.AreEqual(28 * 28, input.Length);
            for (int index = 0; index < input.Length; ++index)
            {
                Set(heap, program, "__0_index__param", index);
                Set(heap, program, "__0_value__param", input[index]);
                Execute(vm, program, "__0_WriteMnistInput");
                Assert.AreEqual(0, Get<int>(heap, program, "__udoncl_status_1"));
                Assert.AreEqual(0, Get<int>(heap, program, "__0___0_WriteMnistInput__ret"));
            }

            Set(heap, program, "__udoncl_async_resume_state", 0);
            Set(heap, program, "__udoncl_async_command_kind", 0);
            int commands = RunAsyncExport(
                vm,
                program,
                heap,
                rawHeap,
                buffers,
                materials,
                copyMaterial
            );
            int runStatus = Get<int>(heap, program, "__udoncl_status_2");
            int runResult = Get<int>(heap, program, "__0_RunMnist__ret");
            TestContext.WriteLine(
                $"commands={commands} status={runStatus} result={runResult} resume={Get<int>(heap, program, "__udoncl_async_resume_state")}"
            );
            TestContext.WriteLine("run_i32_slots=" + DumpRunIntSlots(heap, program));
            Assert.AreEqual(0, runStatus, "MNIST inference returned an OpenCL error status.");
            Assert.AreEqual(0, runResult, "RunMnist returned an error.");
            Assert.Greater(commands, 0, "MNIST inference emitted no transfer commands.");

            float[] actual = new float[10];
            for (int index = 0; index < actual.Length; ++index)
            {
                Set(heap, program, "__1_index__param", index);
                Execute(vm, program, "__0_ReadMnistOutput");
                Assert.AreEqual(0, Get<int>(heap, program, "__udoncl_status_3"));
                actual[index] = Get<float>(heap, program, "__0___0_ReadMnistOutput__ret");
            }

            float[] expected = ReadFloats(
                ResolvePath(
                    "UDONCL_ONNX_MNIST_REFERENCE",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-reference.bytes")
                )
            );
            Assert.AreEqual(actual.Length, expected.Length);
            float maximumError = 0.0f;
            for (int index = 0; index < actual.Length; ++index)
                maximumError = Mathf.Max(maximumError, Mathf.Abs(actual[index] - expected[index]));
            TestContext.WriteLine("actual_logits=[" + string.Join(",", actual) + "]");
            TestContext.WriteLine("expected_logits=[" + string.Join(",", expected) + "]");
            TestContext.WriteLine(
                $"max_error={maximumError:R} actual_argmax={ArgMax(actual)} expected_argmax={ArgMax(expected)}"
            );
            Assert.That(maximumError, Is.LessThan(0.002f));
            Assert.AreEqual(8, ArgMax(expected));
            Assert.AreEqual(ArgMax(expected), ArgMax(actual));
            TestContext.Progress.WriteLine(
                "public_mnist commands={0} max_error={1:R} argmax={2} logits=[{3}]",
                commands,
                maximumError,
                ArgMax(actual),
                string.Join(",", actual)
            );
        }

        [OneTimeTearDown]
        public void RemoveGeneratedAssets()
        {
            RenderTexture.active = null;
            AssetDatabase.DeleteAsset(GeneratedRoot);
            if (!string.IsNullOrEmpty(serializedProgramPath))
                AssetDatabase.DeleteAsset(serializedProgramPath);
            if (serializedProgramsBefore != null)
                foreach (string path in SerializedProgramAssets())
                    if (!serializedProgramsBefore.Contains(path))
                        AssetDatabase.DeleteAsset(path);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Assert.IsNull(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(GeneratedRoot));
            if (!string.IsNullOrEmpty(serializedProgramPath))
                Assert.IsNull(
                    AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(serializedProgramPath)
                );
            if (serializedProgramsBefore != null)
                CollectionAssert.AreEquivalent(
                    serializedProgramsBefore,
                    SerializedProgramAssets(),
                    "Public MNIST test leaked a serialized Udon program."
                );
        }

        private static int RunAsyncExport(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            int[] rawHeap,
            IReadOnlyDictionary<int, RenderTexture> buffers,
            IReadOnlyDictionary<int, Material> materials,
            Material copyMaterial
        )
        {
            int commands = 0;
            for (int iteration = 0; iteration < 128; ++iteration)
            {
                Execute(vm, program, RunEntry);
                int kind = Get<int>(heap, program, "__udoncl_async_command_kind");
                int resume = Get<int>(heap, program, "__udoncl_async_resume_state");
                if (kind == 0)
                {
                    if (resume != 0)
                    {
                        TestContext.WriteLine(
                            $"async_step iteration={iteration} resume={resume} kind=immediate"
                        );
                        LogDispatchState(resume, buffers, materials);
                        Set(heap, program, "__udoncl_async_completion_status", 0);
                        continue;
                    }
                    return commands;
                }

                int bufferId = Get<int>(heap, program, "__udoncl_async_command_buffer");
                int offset = Get<int>(heap, program, "__udoncl_async_command_offset");
                int size = Get<int>(heap, program, "__udoncl_async_command_size");
                int pointer = Get<int>(heap, program, "__udoncl_async_command_pointer");
                Assert.IsTrue(buffers.TryGetValue(bufferId, out RenderTexture buffer));
                Assert.That(offset, Is.GreaterThanOrEqualTo(0));
                Assert.That(size, Is.GreaterThan(0));
                Assert.AreEqual(0, offset & 3);
                Assert.AreEqual(0, size & 3);
                Assert.AreEqual(0, pointer & 3);
                Assert.That(pointer, Is.LessThanOrEqualTo(rawHeap.Length - size));
                TestContext.WriteLine(
                    $"async_step iteration={iteration} resume={resume} kind={kind} buffer={bufferId} offset={offset} size={size} pointer={pointer} host_first={HeapFloat(rawHeap, pointer):R}"
                );
                if (kind == 1)
                {
                    Upload(buffer, offset, size, pointer, rawHeap, copyMaterial);
                    TestContext.WriteLine(
                        $"async_upload iteration={iteration} buffer={bufferId} {TextureStats(buffer, offset / 4, size / 4)}"
                    );
                }
                else if (kind == 2)
                {
                    Readback(buffer, offset, size, pointer, rawHeap);
                    TestContext.WriteLine(
                        $"async_readback iteration={iteration} buffer={bufferId} first={HeapFloat(rawHeap, pointer):R}"
                    );
                }
                else
                    Assert.Fail("Unknown MNIST async command kind " + kind);
                ++commands;
                Set(heap, program, "__udoncl_async_completion_status", 0);
                Set(heap, program, "__udoncl_async_command_kind", 0);
            }
            Assert.Fail("MNIST async export did not complete within 128 commands.");
            return -1;
        }

        private static void Upload(
            RenderTexture destination,
            int offset,
            int size,
            int pointer,
            int[] rawHeap,
            Material copyMaterial
        )
        {
            Color32[] pixels = ReadPixels(destination);
            int offsetWords = offset / 4;
            int words = size / 4;
            for (int word = 0; word < words; ++word)
            {
                int byteIndex = pointer + word * 4;
                pixels[offsetWords + word] = new Color32(
                    (byte)rawHeap[byteIndex],
                    (byte)rawHeap[byteIndex + 1],
                    (byte)rawHeap[byteIndex + 2],
                    (byte)rawHeap[byteIndex + 3]
                );
            }
            Texture2D staging = new Texture2D(
                destination.width,
                destination.height,
                TextureFormat.RGBA32,
                false,
                true
            );
            try
            {
                staging.filterMode = FilterMode.Point;
                staging.wrapMode = TextureWrapMode.Clamp;
                staging.SetPixels32(pixels);
                staging.Apply(false, false);
                Graphics.Blit(staging, destination, copyMaterial);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(staging);
            }
        }

        private static void Readback(
            RenderTexture source,
            int offset,
            int size,
            int pointer,
            int[] rawHeap
        )
        {
            Color32[] pixels = ReadPixels(source);
            int offsetWords = offset / 4;
            int words = size / 4;
            for (int word = 0; word < words; ++word)
            {
                int byteIndex = pointer + word * 4;
                Color32 pixel = pixels[offsetWords + word];
                rawHeap[byteIndex] = pixel.r;
                rawHeap[byteIndex + 1] = pixel.g;
                rawHeap[byteIndex + 2] = pixel.b;
                rawHeap[byteIndex + 3] = pixel.a;
            }
        }

        private static void Clear(RenderTexture destination)
        {
            RenderTexture previous = RenderTexture.active;
            try
            {
                RenderTexture.active = destination;
                GL.Clear(false, true, Color.clear);
            }
            finally
            {
                RenderTexture.active = previous;
            }
        }

        private static Color32[] ReadPixels(RenderTexture source)
        {
            RenderTexture previous = RenderTexture.active;
            Texture2D staging = new Texture2D(
                source.width,
                source.height,
                TextureFormat.RGBA32,
                false,
                true
            );
            try
            {
                RenderTexture.active = source;
                staging.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
                staging.Apply(false, false);
                return staging.GetPixels32();
            }
            finally
            {
                RenderTexture.active = previous;
                UnityEngine.Object.DestroyImmediate(staging);
            }
        }

        private static float[] ReadFloats(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            Assert.AreEqual(0, bytes.Length % 4);
            float[] result = new float[bytes.Length / 4];
            for (int index = 0; index < result.Length; ++index)
                result[index] = BitConverter.ToSingle(bytes, index * 4);
            return result;
        }

        private static int ArgMax(float[] values)
        {
            int result = 0;
            for (int index = 1; index < values.Length; ++index)
                if (values[index] > values[result])
                    result = index;
            return result;
        }

        private string ResolvePath(string environmentName, string relativeFallback)
        {
            string configured = Environment.GetEnvironmentVariable(environmentName);
            return !string.IsNullOrWhiteSpace(configured)
                ? Path.GetFullPath(configured)
                : Path.Combine(repository, relativeFallback);
        }

        private static string ResolveRepository()
        {
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(PublicMnistOnnxRuntimeLifecycleTests).Assembly
                );
            Assert.NotNull(package);
            return Path.GetFullPath(Path.Combine(package.resolvedPath, "..", "..", ".."));
        }

        private static HashSet<string> SerializedProgramAssets()
        {
            HashSet<string> result = new HashSet<string>(StringComparer.Ordinal);
            foreach (
                string guid in AssetDatabase.FindAssets(
                    string.Empty,
                    new[] { "Assets/SerializedUdonPrograms" }
                )
            )
                result.Add(AssetDatabase.GUIDToAssetPath(guid));
            return result;
        }

        private static T Load<T>(string relativePath)
            where T : UnityEngine.Object
        {
            T asset = AssetDatabase.LoadAssetAtPath<T>(GeneratedRoot + "/" + relativePath);
            Assert.NotNull(asset, "Missing generated asset " + relativePath);
            return asset;
        }

        private static void Execute(IUdonVM vm, IUdonProgram program, string entry)
        {
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol(entry));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM failed at " + entry);
        }

        private static T Get<T>(IUdonHeap heap, IUdonProgram program, string symbol)
        {
            return heap.GetHeapVariable<T>(program.SymbolTable.GetAddressFromSymbol(symbol));
        }

        private static string DumpRunIntSlots(IUdonHeap heap, IUdonProgram program)
        {
            List<string> values = new List<string>();
            for (int index = 0; index <= 44; ++index)
            {
                string symbol = "__vrc_f2_r_SystemInt32_" + index;
                object value = heap.GetHeapVariable<object>(
                    program.SymbolTable.GetAddressFromSymbol(symbol)
                );
                values.Add(index + "=" + (value == null ? "null" : value.ToString()));
            }
            return string.Join(",", values);
        }

        private static float HeapFloat(int[] heap, int pointer)
        {
            int bits =
                (heap[pointer] & 255)
                | ((heap[pointer + 1] & 255) << 8)
                | ((heap[pointer + 2] & 255) << 16)
                | ((heap[pointer + 3] & 255) << 24);
            return BitConverter.Int32BitsToSingle(bits);
        }

        private static void LogDispatchState(
            int resume,
            IReadOnlyDictionary<int, RenderTexture> buffers,
            IReadOnlyDictionary<int, Material> materials
        )
        {
            if (resume == 4)
            {
                Material material = materials[232];
                TestContext.WriteLine(
                    "conv_material="
                        + string.Join(
                            ",",
                            new[]
                            {
                                material.GetInteger("_VRC_Arg4"),
                                material.GetInteger("_VRC_Arg5"),
                                material.GetInteger("_VRC_Arg6"),
                                material.GetInteger("_VRC_Arg7"),
                                material.GetInteger("_VRC_Arg8"),
                                material.GetInteger("_VRC_Arg9"),
                                material.GetInteger("_VRC_Arg10"),
                                material.GetInteger("_VRC_Arg11"),
                                material.GetInteger("_VRC_Arg12"),
                                material.GetInteger("_VRC_Arg13"),
                            }
                        )
                        + $",alpha={material.GetFloat("_VRC_Arg14"):R},beta={material.GetFloat("_VRC_Arg15"):R}"
                );
                TestContext.WriteLine(
                    $"conv_textures={ReferenceEquals(material.GetTexture("_VRC_Input"), buffers[1])},{ReferenceEquals(material.GetTexture("_VRC_Input1"), buffers[2])},{ReferenceEquals(material.GetTexture("_VRC_Input2"), buffers[4])}"
                );
            }
            if (resume == 20)
            {
                Material material = materials[1];
                TestContext.WriteLine(
                    "dense_material="
                        + string.Join(
                            ",",
                            new[]
                            {
                                material.GetInteger("_VRC_Arg2"),
                                material.GetInteger("_VRC_Arg3"),
                                material.GetInteger("_VRC_Arg4"),
                                material.GetInteger("_VRC_Arg5"),
                                material.GetInteger("_VRC_Arg6"),
                                material.GetInteger("_VRC_Arg7"),
                                material.GetInteger("_VRC_Arg8"),
                                material.GetInteger("_VRC_Arg9"),
                            }
                        )
                        + $",alpha={material.GetFloat("_VRC_Arg10"):R},beta={material.GetFloat("_VRC_Arg11"):R}"
                );
                TestContext.WriteLine(
                    $"dense_texture={ReferenceEquals(material.GetTexture("_VRC_Input"), buffers[5])}"
                );
            }
            int bufferId = resume == 20 ? 6 : 3;
            int offsetWords = resume == 20 ? 768 : 0;
            int words =
                resume == 17 ? 144
                : resume == 20 ? 10
                : 9216;
            TestContext.WriteLine(
                $"dispatch_output resume={resume} {TextureStats(buffers[bufferId], offsetWords, words)}"
            );
        }

        private static string TextureStats(RenderTexture texture, int offsetWords, int words)
        {
            Color32[] pixels = ReadPixels(texture);
            int nonzero = 0;
            float maximumAbsolute = 0.0f;
            for (int index = 0; index < words; ++index)
            {
                Color32 pixel = pixels[offsetWords + index];
                int bits = pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
                float value = BitConverter.Int32BitsToSingle(bits);
                if (bits != 0)
                    ++nonzero;
                maximumAbsolute = Mathf.Max(maximumAbsolute, Mathf.Abs(value));
            }
            return $"nonzero={nonzero}/{words} max_abs={maximumAbsolute:R}";
        }

        private static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
