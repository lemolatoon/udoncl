using System;
using System.IO;
using NUnit.Framework;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    /// <summary>
    /// Validates and executes the import-ready Unity library emitted by the
    /// public CMake package. Full Importer and generated-binding lifecycle
    /// validation is kept in the MCP-driven Unity acceptance check.
    /// </summary>
    public sealed class UnityLibraryArtifactLifecycleTests
    {
        [Test]
        public void CmakeUnityLibraryExecutesOnSdkUdonVm()
        {
            string artifactDirectory = ResolveArtifactDirectory();
            ValidatedArtifact artifact = ArtifactManifestValidator.Validate(artifactDirectory);
            Assert.AreEqual("unity-udoncl-library", artifact.ArtifactStage);
            Assert.AreEqual("hello_kernel", artifact.ProgramName);

            string assemblyPath = Path.Combine(artifact.RootDirectory, "host.uasm");
            Assert.That(File.Exists(assemblyPath), "Compiler output is missing host.uasm.");
            IUdonProgram program = UdonEditorManager.Instance.Assemble(
                File.ReadAllText(assemblyPath)
            );
            Assert.NotNull(program, "SDK assembler rejected the CMake-generated library.");

            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(vm);
            Assert.IsTrue(vm.LoadProgram(program), "SDK VM rejected the generated library.");
            IUdonHeap heap = vm.InspectHeap();
            heap.SetHeapVariable(
                program.SymbolTable.GetAddressFromSymbol("__0_value__param"),
                3.0f
            );
            vm.SetProgramCounter(program.EntryPoints.GetAddressFromSymbol("__0_SquarePlusOne"));
            Assert.AreEqual(0u, vm.Interpret(), "SDK VM failed to execute SquarePlusOne.");
            Assert.AreEqual(
                0,
                heap.GetHeapVariable<int>(
                    program.SymbolTable.GetAddressFromSymbol("__udoncl_status_1")
                )
            );
            Assert.AreEqual(
                10.0f,
                heap.GetHeapVariable<float>(
                    program.SymbolTable.GetAddressFromSymbol("__0___0_SquarePlusOne__ret")
                ),
                1e-6f
            );
        }

        private static string ResolveArtifactDirectory()
        {
            string configured = Environment.GetEnvironmentVariable("UDONCL_UNITY_LIBRARY_ARTIFACT");
            if (!string.IsNullOrWhiteSpace(configured))
                return Path.GetFullPath(configured);

            UnityEditor.PackageManager.PackageInfo package =
                UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                    typeof(UnityLibraryArtifactLifecycleTests).Assembly
                );
            Assert.NotNull(package, "Could not resolve the UdonCL package root.");
            string repository = Path.GetFullPath(
                Path.Combine(package.resolvedPath, "..", "..", "..")
            );
            return Path.Combine(
                repository,
                "build",
                "udoncl",
                "tests",
                "cmake-package",
                "consumer",
                "udoncl",
                "hello-unity-library",
                "artifact"
            );
        }
    }
}
