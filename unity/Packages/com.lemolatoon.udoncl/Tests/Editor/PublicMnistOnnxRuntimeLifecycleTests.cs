using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UdonSharp;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using VRC.Udon;
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
        private const string GeneratedRoot = "Assets/udoncl-generated/generic_onnx_mnist";
        private const string BindingClass = "UdonclGenericOnnxMnistBinding";
        private const string ControllerClass = "UdonclGenericOnnxMnistTestController";
        private const string GeneratedNamespace = "Lemolatoon.Udoncl.Generated.";
        private const string RunEntry = "Run";
        private const string ControllerSource =
            "Assets/UdonclSampleValidation/" + ControllerClass + ".cs";
        private const string ControllerProgram =
            GeneratedRoot + "/Tests/" + ControllerClass + ".asset";
        private string repository;
        private string artifactDirectory;
        private ValidatedArtifact artifact;

        [OneTimeSetUp]
        public void ImportPublicMnistArtifact()
        {
            repository = ResolveRepository();
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
            artifact = ArtifactManifestValidator.Validate(artifactDirectory);
            Assert.AreEqual("generic_onnx_mnist", artifact.ProgramName);
            if (!AssetDatabase.IsValidFolder(GeneratedRoot))
                Assert.AreEqual(GeneratedRoot, UdonclArtifactImporter.Import(artifactDirectory));
            CollectionAssert.AreEqual(
                File.ReadAllBytes(Path.Combine(artifactDirectory, "binding.cs")),
                File.ReadAllBytes(GeneratedRoot + "/Bindings/" + BindingClass + ".cs"),
                "Importer must install compiler-generated binding source byte-for-byte."
            );

            string programPath = GeneratedRoot + "/Programs/host_program.asset";
            string programGuid = AssetDatabase.AssetPathToGUID(programPath);
            Assert.IsFalse(string.IsNullOrEmpty(programGuid));

            Assert.NotNull(
                AssetDatabase.LoadAssetAtPath<MonoScript>(ControllerSource),
                "The pre-test MNIST U# controller source was not imported."
            );
            if (!AssetDatabase.IsValidFolder(GeneratedRoot + "/Tests"))
                AssetDatabase.CreateFolder(GeneratedRoot, "Tests");
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                GeneratedRoot + "/Bindings/" + BindingClass + ".cs",
                GeneratedRoot + "/Bindings/" + BindingClass + ".asset"
            );
            UdonclArtifactImporter.EnsureCompiledUdonSharpProgramAsset(
                ControllerSource,
                ControllerProgram
            );
            GeneratedBindingFinalizer.FinalizeAvailableBindings();
        }

        [Test]
        public void ImportedStaticTensorsPreserveBytesAndBindingTable()
        {
            Assert.AreEqual(4, artifact.StaticTensors.Count);
            Texture2D[] imported = new Texture2D[artifact.StaticTensors.Count];
            foreach (ValidatedStaticTensor tensor in artifact.StaticTensors)
            {
                Texture2D texture = Load<Texture2D>(
                    "Textures/"
                        + UdonclArtifactImporter.StaticTensorFileName(tensor.Id, tensor.Name)
                );
                Assert.AreEqual(tensor.Width, texture.width);
                Assert.AreEqual(tensor.Height, texture.height);
                Assert.AreEqual(TextureFormat.RGBA32, texture.format);
                byte[] expected = File.ReadAllBytes(tensor.AbsolutePath);
                var actual = texture.GetRawTextureData<byte>();
                Assert.AreEqual(tensor.Width * tensor.Height * 4, actual.Length);
                for (int index = 0; index < expected.Length; ++index)
                    Assert.AreEqual(expected[index], actual[index], "Static tensor byte " + index);
                for (int index = expected.Length; index < actual.Length; ++index)
                    Assert.AreEqual(0, actual[index], "Static tensor padding byte " + index);
                imported[tensor.Id] = texture;
            }

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/" + BindingClass + ".prefab"
            );
            Assert.NotNull(prefab);
            UdonSharpBehaviour proxy = null;
            foreach (UdonSharpBehaviour candidate in prefab.GetComponents<UdonSharpBehaviour>())
                if (candidate.GetType().FullName == GeneratedNamespace + BindingClass)
                    proxy = candidate;
            Assert.NotNull(proxy);
            SerializedProperty table = new SerializedObject(proxy).FindProperty("staticTensors");
            Assert.NotNull(table);
            Assert.AreEqual(imported.Length, table.arraySize);
            for (int index = 0; index < imported.Length; ++index)
                Assert.AreEqual(
                    imported[index],
                    table.GetArrayElementAtIndex(index).objectReferenceValue
                );
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
            int[] rawHeap = PackRawHeap(rawHeapBytes);
            Set(heap, program, "__vrc_raw_heap", rawHeap);

            Texture2D[] staticTensors = new Texture2D[artifact.StaticTensors.Count];
            foreach (ValidatedStaticTensor tensor in artifact.StaticTensors)
                staticTensors[tensor.Id] = Load<Texture2D>(
                    "Textures/"
                        + UdonclArtifactImporter.StaticTensorFileName(tensor.Id, tensor.Name)
                );
            Set(heap, program, artifact.StaticTensorTableSlot, staticTensors);

            Dictionary<int, RenderTexture> buffers = new Dictionary<int, RenderTexture>();
            RenderTexture[] bufferPool = new RenderTexture[artifact.DeviceBuffers.Count];
            for (int index = 0; index < artifact.DeviceBuffers.Count; ++index)
            {
                ValidatedDeviceBuffer descriptor = artifact.DeviceBuffers[index];
                RenderTexture buffer = Load<RenderTexture>(
                    "RenderTextures/"
                        + UdonclArtifactImporter.DeviceBufferFileName(
                            descriptor.Id,
                            descriptor.Name
                        )
                );
                buffers.Add(descriptor.Id, buffer);
                bufferPool[index] = buffer;
                Clear(buffer);
            }
            Set(heap, program, "__udoncl_buffer_pool", bufferPool);

            Dictionary<int, Material> materials = new Dictionary<int, Material>();
            foreach (ValidatedKernel kernel in artifact.Kernels)
            {
                Material material = Load<Material>(
                    "Materials/kernel_" + kernel.Id + "_" + kernel.Name + ".mat"
                );
                materials.Add(kernel.Id, material);
                Set(heap, program, "__udoncl_kernel_material_" + kernel.Id, material);
            }
            // 4096 is the synthesized kernel of the fused bias Add and Relu.
            CollectionAssert.AreEquivalent(new[] { 221, 231, 232, 233, 234, 4096 }, materials.Keys);
            Material copyMaterial = Load<Material>("Materials/buffer-copy.mat");

            float[] input = ReadFloats(
                ResolvePath(
                    "UDONCL_ONNX_MNIST_INPUT",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-input.bytes")
                )
            );
            Assert.AreEqual(28 * 28, input.Length);
            int initializeCommands = RunAsyncExport(
                vm,
                program,
                heap,
                rawHeap,
                buffers,
                materials,
                copyMaterial,
                "Initialize"
            );
            int initializeStatus = Get<int>(heap, program, "__udoncl_status_562720017");
            int initializeResult = Get<int>(heap, program, "__0_Initialize__ret");
            Assert.AreEqual(0, initializeStatus, "MNIST initialization returned an ABI error.");
            Assert.AreEqual(0, initializeResult, "MNIST initialization failed.");
            Assert.AreEqual(
                0,
                initializeCommands,
                "Static-tensor MNIST initialization unexpectedly emitted an async transfer."
            );

            UploadFloats(buffers[1], input, copyMaterial);
            int commands = RunAsyncExport(
                vm,
                program,
                heap,
                rawHeap,
                buffers,
                materials,
                copyMaterial,
                RunEntry
            );
            int runStatus = Get<int>(heap, program, "__udoncl_status_483499420");
            int runResult = Get<int>(heap, program, "__0_Run__ret");
            TestContext.WriteLine(
                $"initialize_commands={initializeCommands} commands={commands} status={runStatus} result={runResult}"
            );
            Assert.AreEqual(0, runStatus, "MNIST inference returned an OpenCL error status.");
            Assert.AreEqual(0, runResult, "RunMnist returned an error.");
            // Device dispatches are enqueue-only and the six MNIST commands fit
            // in one Device work slice, so inference completes without a
            // suspension; the output is still read back and compared below.
            Assert.AreEqual(
                0,
                commands,
                "Enqueue-only MNIST inference within one work slice must not suspend."
            );

            float[] actual = ReadFloats(buffers[2], 10);

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

        [UnityTest]
        public IEnumerator GeneratedUSharpBindingInfersMnistAndMatchesReference()
        {
            yield return new EnterPlayMode();

            float[] input = ReadFloats(
                ResolvePath(
                    "UDONCL_ONNX_MNIST_INPUT",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-input.bytes")
                )
            );
            float[] expected = ReadFloats(
                ResolvePath(
                    "UDONCL_ONNX_MNIST_REFERENCE",
                    Path.Combine("build", "precommit", "onnx-aot", "unity-mnist-reference.bytes")
                )
            );
            Assert.AreEqual(28 * 28, input.Length);
            Assert.AreEqual(10, expected.Length);

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(
                GeneratedRoot + "/Prefabs/" + BindingClass + ".prefab"
            );
            Assert.NotNull(prefab);
            GameObject instance = UnityEngine.Object.Instantiate(prefab);
            UdonBehaviour controller = UdonclArtifactImporter.AddCompiledUdonSharpBehaviour(
                instance,
                ControllerProgram
            );
            UdonclArtifactImporter.InitializeUdonBehaviours(instance);
            yield return null;

            UdonBehaviour binding = UdonclArtifactImporter.FindUdonSharpBacking(
                instance,
                GeneratedNamespace + BindingClass
            );
            Assert.NotNull(binding);
            Assert.NotNull(controller);
            UdonBehaviour program = (UdonBehaviour)
                UdonclArtifactImporter.GetUdonProgramVariable(binding, "program");
            Assert.NotNull(program, "Generated U# binding did not retain its Host program.");
            int[] rawHeap = (int[])
                UdonclArtifactImporter.GetUdonProgramVariable(program, "__vrc_raw_heap");
            byte[] expectedRawHeap = File.ReadAllBytes(
                Path.Combine(artifactDirectory, "raw-heap.bytes")
            );
            Assert.NotNull(rawHeap, "Generated Host program has no initialized C raw heap.");
            Assert.AreEqual((expectedRawHeap.Length + 3) / 4, rawHeap.Length);
            for (int index = 0; index < expectedRawHeap.Length; ++index)
                Assert.AreEqual(
                    expectedRawHeap[index],
                    (byte)(rawHeap[index / 4] >> ((index & 3) * 8)),
                    "Generated Host raw heap differs at byte " + index + "."
                );
            float[] logits = new float[expected.Length];
            UdonclArtifactImporter.SetUdonProgramVariable(controller, "binding", binding);
            UdonclArtifactImporter.SetUdonProgramVariable(controller, "input", input);
            UdonclArtifactImporter.SetUdonProgramVariable(controller, "logits", logits);
            Assert.AreSame(
                input,
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "input"),
                "The generated U# controller did not retain the assigned ONNX input buffer."
            );
            Assert.AreSame(
                logits,
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "logits"),
                "The generated U# controller did not retain the assigned ONNX output buffer."
            );
            UdonclArtifactImporter.SendUdonCustomEvent(controller, "Run");

            const int timeoutFrames = 1200;
            int frame = 0;
            bool completed = false;
            while (frame < timeoutFrames && !completed)
            {
                ++frame;
                yield return null;
                completed = (bool)
                    UdonclArtifactImporter.GetUdonProgramVariable(controller, "completed");
            }
            int status = (int)UdonclArtifactImporter.GetUdonProgramVariable(controller, "status");
            int libraryStatus = (int)
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "libraryStatus");
            int phase = (int)UdonclArtifactImporter.GetUdonProgramVariable(controller, "phase");
            float[] actual = (float[])
                UdonclArtifactImporter.GetUdonProgramVariable(controller, "logits");
            yield return new ExitPlayMode();

            TestContext.Progress.WriteLine(
                "generated_usharp_mnist frames={0} completed={1} status={2} library_status={3} phase={4} logits=[{5}]",
                frame,
                completed,
                status,
                libraryStatus,
                phase,
                string.Join(",", actual)
            );

            Assert.IsTrue(
                completed,
                "Generated U# MNIST execution timed out after "
                    + frame
                    + " frames; status="
                    + status
                    + "; libraryStatus="
                    + libraryStatus
            );
            Assert.AreEqual(
                0,
                status,
                "Generated U# binding returned an API error; phase="
                    + phase
                    + "; libraryStatus="
                    + libraryStatus
            );
            Assert.AreEqual(
                0,
                libraryStatus,
                "MNIST host program returned an error at controller phase " + phase + "."
            );
            Assert.AreEqual(expected.Length, actual.Length);
            float maximumError = 0.0f;
            for (int index = 0; index < actual.Length; ++index)
                maximumError = Mathf.Max(maximumError, Mathf.Abs(actual[index] - expected[index]));
            TestContext.Progress.WriteLine(
                "generated_usharp_mnist max_error={0:R} argmax={1}",
                maximumError,
                ArgMax(actual)
            );
            Assert.That(maximumError, Is.LessThan(0.002f));
            Assert.AreEqual(8, ArgMax(expected));
            Assert.AreEqual(ArgMax(expected), ArgMax(actual));
        }

        [OneTimeTearDown]
        public void RemoveGeneratedAssets()
        {
            RenderTexture.active = null;
            // Generated U# sources remain valid until Unity exits. Deleting them from a
            // fixture teardown races UdonSharp's asynchronous whole-assembly compiler.
            // scripts/precommit.py restores the generated/source/serialized snapshots
            // after the Editor process has stopped.
        }

        internal static int RunAsyncExport(
            IUdonVM vm,
            IUdonProgram program,
            IUdonHeap heap,
            int[] rawHeap,
            IReadOnlyDictionary<int, RenderTexture> buffers,
            IReadOnlyDictionary<int, Material> materials,
            Material copyMaterial,
            string entry = RunEntry
        )
        {
            int commands = 0;
            Set(heap, program, "__udoncl_async_pending_object", 0);
            Set(heap, program, "__udoncl_async_completion_object", 0);
            Set(heap, program, "__udoncl_async_command_kind", 0);
            Execute(vm, program, entry);
            for (int iteration = 0; iteration < 128; ++iteration)
            {
                int kind = Get<int>(heap, program, "__udoncl_async_command_kind");
                int pending = Get<int>(heap, program, "__udoncl_async_pending_object");
                if (pending == 0)
                {
                    Assert.AreEqual(0, kind, "Completed async export retained a command kind.");
                    return commands;
                }
                Set(heap, program, "__udoncl_async_pending_object", 0);

                TestContext.WriteLine(
                    $"async_step iteration={iteration} pending={pending} kind={kind}"
                );
                if (kind == 0)
                {
                    ++commands;
                    Set(heap, program, "__udoncl_async_completion_object", pending);
                    Set(heap, program, "__udoncl_async_completion_status", 0);
                    Execute(vm, program, "__udoncl_async_complete");
                    Set(heap, program, "__udoncl_async_completion_object", 0);
                    continue;
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
                Assert.That(pointer, Is.LessThanOrEqualTo(rawHeap.Length * 4 - size));
                TestContext.WriteLine(
                    $"async_transfer iteration={iteration} pending={pending} kind={kind} buffer={bufferId} offset={offset} size={size} pointer={pointer} host_first={HeapFloat(rawHeap, pointer):R}"
                );
                if (kind == 1)
                {
                    Upload(buffer, offset, size, pointer, rawHeap, copyMaterial);
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
                Set(heap, program, "__udoncl_async_command_kind", 0);
                Set(heap, program, "__udoncl_async_completion_object", pending);
                Set(heap, program, "__udoncl_async_completion_status", 0);
                Execute(vm, program, "__udoncl_async_complete");
                Set(heap, program, "__udoncl_async_completion_object", 0);
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
                int value = rawHeap[pointer / 4 + word];
                pixels[offsetWords + word] = new Color32(
                    (byte)value,
                    (byte)(value >> 8),
                    (byte)(value >> 16),
                    (byte)(value >> 24)
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

        internal static void UploadFloats(
            RenderTexture destination,
            float[] values,
            Material copyMaterial
        )
        {
            Assert.That(
                values.Length,
                Is.LessThanOrEqualTo(destination.width * destination.height)
            );
            Color32[] pixels = new Color32[destination.width * destination.height];
            for (int index = 0; index < values.Length; ++index)
            {
                int word = BitConverter.SingleToInt32Bits(values[index]);
                pixels[index] = new Color32(
                    (byte)word,
                    (byte)(word >> 8),
                    (byte)(word >> 16),
                    (byte)(word >> 24)
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

        internal static void UploadInts(
            RenderTexture destination,
            int[] values,
            Material copyMaterial
        )
        {
            Assert.That(
                values.Length,
                Is.LessThanOrEqualTo(destination.width * destination.height)
            );
            Color32[] pixels = new Color32[destination.width * destination.height];
            for (int index = 0; index < values.Length; ++index)
            {
                int word = values[index];
                pixels[index] = new Color32(
                    (byte)word,
                    (byte)(word >> 8),
                    (byte)(word >> 16),
                    (byte)(word >> 24)
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

        internal static void UploadLongs(
            RenderTexture destination,
            long[] values,
            Material copyMaterial
        )
        {
            Assert.That(
                values.Length * 2,
                Is.LessThanOrEqualTo(destination.width * destination.height)
            );
            Color32[] pixels = new Color32[destination.width * destination.height];
            for (int index = 0; index < values.Length; ++index)
            {
                long value = values[index];
                int low = (int)value;
                int high = (int)(value >> 32);
                pixels[index * 2] = new Color32(
                    (byte)low,
                    (byte)(low >> 8),
                    (byte)(low >> 16),
                    (byte)(low >> 24)
                );
                pixels[index * 2 + 1] = new Color32(
                    (byte)high,
                    (byte)(high >> 8),
                    (byte)(high >> 16),
                    (byte)(high >> 24)
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
                Color32 pixel = pixels[offsetWords + word];
                rawHeap[pointer / 4 + word] =
                    pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
            }
        }

        internal static void Clear(RenderTexture destination)
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

        internal static Color32[] ReadPixels(RenderTexture source)
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

        private static float[] ReadFloats(RenderTexture source, int count)
        {
            Color32[] pixels = ReadPixels(source);
            Assert.That(count, Is.LessThanOrEqualTo(pixels.Length));
            float[] result = new float[count];
            for (int index = 0; index < count; ++index)
            {
                Color32 pixel = pixels[index];
                int word = pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
                result[index] = BitConverter.Int32BitsToSingle(word);
            }
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

        internal static string ResolveRepository()
        {
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(PublicMnistOnnxRuntimeLifecycleTests).Assembly
                );
            Assert.NotNull(package);
            return Path.GetFullPath(Path.Combine(package.resolvedPath, "..", "..", ".."));
        }

        internal static HashSet<string> SerializedProgramAssets()
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

        internal static void Execute(IUdonVM vm, IUdonProgram program, string entry)
        {
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol(entry));
            try
            {
                Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM failed at " + entry);
            }
            catch (Exception exception)
            {
                IUdonHeap heap = vm.InspectHeap();
                int[] rawHeap = Get<int[]>(heap, program, "__vrc_raw_heap");
                Assert.Fail(
                    "SDK IUdonVM threw at "
                        + entry
                        + " PC "
                        + vm.GetProgramCounter()
                        + "; raw_word_index="
                        + Get<int>(heap, program, "__vrc_raw_byte_load_word_index")
                        + "; raw_heap_length="
                        + rawHeap.Length
                        + ": "
                        + exception
                );
            }
        }

        internal static T Get<T>(IUdonHeap heap, IUdonProgram program, string symbol)
        {
            return heap.GetHeapVariable<T>(program.SymbolTable.GetAddressFromSymbol(symbol));
        }

        private static float HeapFloat(int[] heap, int pointer)
        {
            return BitConverter.Int32BitsToSingle(heap[pointer / 4]);
        }

        internal static int[] PackRawHeap(byte[] bytes)
        {
            int[] words = new int[(bytes.Length + 3) / 4];
            for (int index = 0; index < bytes.Length; ++index)
                words[index / 4] |= bytes[index] << ((index & 3) * 8);
            return words;
        }

        internal static void Set<T>(IUdonHeap heap, IUdonProgram program, string symbol, T value)
        {
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(symbol), value);
        }
    }
}
