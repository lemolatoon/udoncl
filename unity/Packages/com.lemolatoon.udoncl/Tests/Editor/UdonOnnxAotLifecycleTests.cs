using System;
using System.IO;
using NUnit.Framework;
using UnityEditor.PackageManager;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class UdonOnnxAotLifecycleTests
    {
        [TestCase("UDONCL_ONNX_AOT_UASM", "generic-host.uasm", "pass202Material")]
        [TestCase("UDONCL_ONNX_TILED_AOT_UASM", "tiled-gemm-host.uasm", "pass211Material")]
        [TestCase(
            "UDONCL_ONNX_MNIST_OPERATOR_AOT_UASM",
            "mnist-operators-host.uasm",
            "pass213Material"
        )]
        [TestCase("UDONCL_ONNX_RL_OPERATOR_AOT_UASM", "rl-policy-host.uasm", "pass209Material")]
        [TestCase(
            "UDONCL_ONNX_TINY_LLAMA_OPERATOR_AOT_UASM",
            "tiny-llama-standard-host.uasm",
            "pass214Material"
        )]
        public void GeneratedOperatorPlanHostAssemblesWithSdk(
            string environmentName,
            string fallbackName,
            string finalMaterial
        )
        {
            string fixture = Environment.GetEnvironmentVariable(environmentName);
            if (string.IsNullOrWhiteSpace(fixture))
            {
                UnityEditor.PackageManager.PackageInfo package =
                    UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                        typeof(UdonOnnxAotLifecycleTests).Assembly
                    );
                Assert.NotNull(package, "Could not resolve the UdonCL package root.");
                string repository = Path.GetFullPath(
                    Path.Combine(package.resolvedPath, "..", "..", "..")
                );
                fixture = Path.Combine(repository, "build", "precommit", "onnx-aot", fallbackName);
            }
            fixture = Path.GetFullPath(fixture);
            Assert.That(File.Exists(fixture), "Missing compiler-emitted ONNX AOT Host fixture.");
            IUdonProgram program = UdonEditorManager.Instance.Assemble(File.ReadAllText(fixture));
            Assert.NotNull(program, "SDK assembler rejected the generated ONNX AOT Host.");
            Assert.DoesNotThrow(() => program.EntryPoints.GetAddressFromSymbol("_run"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("initialState"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("stateA"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("stateB"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("resultState"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol("pass200Material"));
            Assert.DoesNotThrow(() => program.SymbolTable.GetAddressFromSymbol(finalMaterial));
            IUdonVM vm = UdonEditorManager.Instance.ConstructUdonVM();
            Assert.NotNull(vm);
            Assert.IsTrue(vm.LoadProgram(program), "SDK VM rejected the generated ONNX AOT Host.");
        }
    }
}
