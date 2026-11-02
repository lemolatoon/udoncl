using System;
using System.IO;
using NUnit.Framework;
using UnityEditor.PackageManager;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    /// <summary>
    /// End-to-end Unity fixture for MCP-driven validation. OneTimeSetUp stages
    /// the emitted lowering artifact in Library, tests assemble and execute it
    /// on the SDK IUdonVM, and OneTimeTearDown removes every staged file. The
    /// World bundle test separately covers the SDK .uasm Asset importer.
    /// </summary>
    public sealed class UdonVmLoweringLifecycleTests
    {
        private const string TemporaryDirectory = "Library/UdonclLoweringIntegration";
        private const string TemporaryAssembly = TemporaryDirectory + "/scalar.uasm";
        private string assemblyText;

        [OneTimeSetUp]
        public void SetupLoweredAssemblyAsset()
        {
            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UdonVmLoweringLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string fixture = Environment.GetEnvironmentVariable("UDONCL_LOWERED_UASM");
            if (string.IsNullOrWhiteSpace(fixture))
            {
                string repository = Path.GetFullPath(
                    Path.Combine(package.resolvedPath, "..", "..", "..")
                );
                string generated = Path.Combine(
                    repository,
                    "build",
                    "udoncl",
                    "tests",
                    "udon-scalar",
                    "scalar.uasm"
                );
                fixture = File.Exists(generated)
                    ? generated
                    : Path.Combine(
                        package.resolvedPath,
                        "Tests",
                        "Editor",
                        "Fixtures",
                        "scalar.uasm"
                    );
            }
            else
                fixture = Path.GetFullPath(fixture);
            Assert.That(File.Exists(fixture), "Missing compiler-emitted scalar.uasm fixture.");
            assemblyText = File.ReadAllText(fixture);
            if (Directory.Exists(TemporaryDirectory))
                Directory.Delete(TemporaryDirectory, true);
            Directory.CreateDirectory(TemporaryDirectory);
            File.WriteAllText(TemporaryAssembly, assemblyText);
            Assert.That(File.Exists(TemporaryAssembly));
        }

        [TestCase("__0_add", "__0_arg0__param", 17, "__0_arg1__param", 25, "__0___0_add__ret", 42)]
        [TestCase("__0_sum_to", "__2_arg0__param", 5, null, 0, "__0___0_sum_to__ret", 10)]
        public void IntegerLoweringExecutesOnSdkUdonVm(
            string entry,
            string arg0,
            int value0,
            string arg1,
            int value1,
            string resultSymbol,
            int expected
        )
        {
            IUdonProgram program = LoadProgram();
            IUdonVM vm = CreateVm(program);
            IUdonHeap heap = vm.InspectHeap();
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(arg0), value0);
            if (arg1 != null)
                heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol(arg1), value1);
            Execute(vm, program, entry);
            Assert.AreEqual(
                expected,
                heap.GetHeapVariable<int>(program.SymbolTable.GetAddressFromSymbol(resultSymbol))
            );
        }

        [TestCase(3.5f, -2.0f, -7.0f)]
        public void FloatMultiplicationLoweringExecutesOnSdkUdonVm(
            float value,
            float factor,
            float expected
        )
        {
            IUdonProgram program = LoadProgram();
            IUdonVM vm = CreateVm(program);
            IUdonHeap heap = vm.InspectHeap();
            heap.SetHeapVariable(
                program.SymbolTable.GetAddressFromSymbol("__1_arg0__param"),
                value
            );
            heap.SetHeapVariable(
                program.SymbolTable.GetAddressFromSymbol("__1_arg1__param"),
                factor
            );
            Execute(vm, program, "__0_scale");
            Assert.AreEqual(
                expected,
                heap.GetHeapVariable<float>(
                    program.SymbolTable.GetAddressFromSymbol("__0___0_scale__ret")
                ),
                1e-6f
            );
        }

        [TestCase(0.02f, 0)]
        [TestCase(0.30f, 2)]
        public void ControlFlowLoweringExecutesOnSdkUdonVm(float temperature, int expected)
        {
            IUdonProgram program = LoadProgram();
            IUdonVM vm = CreateVm(program);
            IUdonHeap heap = vm.InspectHeap();
            heap.SetHeapVariable(
                program.SymbolTable.GetAddressFromSymbol("__3_arg0__param"),
                temperature
            );
            Execute(vm, program, "__0_phase");
            Assert.AreEqual(
                expected,
                heap.GetHeapVariable<int>(
                    program.SymbolTable.GetAddressFromSymbol("__0___0_phase__ret")
                )
            );
        }

        [Test]
        public void FloatArrayLoadStoreLoweringExecutesOnSdkUdonVm()
        {
            IUdonProgram program = LoadProgram();
            IUdonVM vm = CreateVm(program);
            IUdonHeap heap = vm.InspectHeap();
            float[] input = { 1.0f, -2.0f, 3.5f, 2.0f };
            float[] output = { 0.0f };
            heap.SetHeapVariable(
                program.SymbolTable.GetAddressFromSymbol("__4_arg0__param"),
                input
            );
            heap.SetHeapVariable(
                program.SymbolTable.GetAddressFromSymbol("__2_arg1__param"),
                output
            );
            Execute(vm, program, "__0_array_sum");
            Assert.AreEqual(
                4.5f,
                heap.GetHeapVariable<float>(
                    program.SymbolTable.GetAddressFromSymbol("__0___0_array_sum__ret")
                ),
                1e-6f
            );
            Assert.AreEqual(4.5f, output[0], 1e-6f);
        }

        [OneTimeTearDown]
        public void CleanupLoweredAssemblyAsset()
        {
            if (Directory.Exists(TemporaryDirectory))
                Directory.Delete(TemporaryDirectory, true);
            Assert.That(
                Directory.Exists(TemporaryDirectory),
                Is.False,
                "Lowering integration test leaked its staged Library directory."
            );
        }

        private static IUdonProgram LoadProgram()
        {
            IUdonProgram program = UdonEditorManager.Instance.Assemble(
                File.ReadAllText(TemporaryAssembly)
            );
            Assert.NotNull(program, "SDK assembler rejected the lowered program.");
            return program;
        }

        private static IUdonVM CreateVm(IUdonProgram program)
        {
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(vm);
            Assert.IsTrue(vm.LoadProgram(program), "SDK VM rejected the lowered program.");
            return vm;
        }

        private static void Execute(IUdonVM vm, IUdonProgram program, string entry)
        {
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol(entry));
            Assert.AreEqual(0u, vm.Interpret(), "SDK IUdonVM reported an execution error.");
        }
    }
}
