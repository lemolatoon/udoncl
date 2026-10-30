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
            string fixture = Path.Combine(
                package.resolvedPath,
                "Tests",
                "Editor",
                "Fixtures",
                "scalar.uasm"
            );
            Assert.That(File.Exists(fixture), "Missing compiler-emitted scalar.uasm fixture.");
            assemblyText = File.ReadAllText(fixture);
            if (Directory.Exists(TemporaryDirectory))
                Directory.Delete(TemporaryDirectory, true);
            Directory.CreateDirectory(TemporaryDirectory);
            File.WriteAllText(TemporaryAssembly, assemblyText);
            Assert.That(File.Exists(TemporaryAssembly));
        }

        [TestCase("VRC_Call_add_1", "VRC_ARG_1_0", 17, "VRC_ARG_1_1", 25, "VRC_RET_1", 42)]
        [TestCase("VRC_Call_sum_to_3", "VRC_ARG_3_0", 5, null, 0, "VRC_RET_3", 10)]
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
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol("VRC_ARG_2_0"), value);
            heap.SetHeapVariable(program.SymbolTable.GetAddressFromSymbol("VRC_ARG_2_1"), factor);
            Execute(vm, program, "VRC_Call_scale_2");
            Assert.AreEqual(
                expected,
                heap.GetHeapVariable<float>(program.SymbolTable.GetAddressFromSymbol("VRC_RET_2")),
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
                program.SymbolTable.GetAddressFromSymbol("VRC_ARG_4_0"),
                temperature
            );
            Execute(vm, program, "VRC_Call_phase_4");
            Assert.AreEqual(
                expected,
                heap.GetHeapVariable<int>(program.SymbolTable.GetAddressFromSymbol("VRC_RET_4"))
            );
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
