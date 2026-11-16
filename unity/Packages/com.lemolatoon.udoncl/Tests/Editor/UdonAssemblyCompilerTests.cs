using System.Text;
using NUnit.Framework;
using VRC.Udon.Common.Interfaces;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class UdonAssemblyCompilerTests
    {
        [Test]
        public void AssemblesProgramsLargerThanTheSdkDefaultHeap()
        {
            StringBuilder assembly = new StringBuilder(".data_start\n");
            for (int index = 0; index < 513; ++index)
                assembly.Append("    value_").Append(index).Append(": %SystemInt32, 0\n");
            assembly.Append(".data_end\n.code_start\n.code_end\n");

            IUdonProgram program = UdonAssemblyCompiler.Assemble(assembly.ToString());

            Assert.NotNull(program);
            Assert.AreEqual(513, program.Heap.GetHeapCapacity());
        }
    }
}
