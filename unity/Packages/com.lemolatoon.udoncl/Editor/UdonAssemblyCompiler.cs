using System;
using VRC.Udon.Common.Interfaces;
using VRC.Udon.Editor;
using VRC.Udon.UAssembly.Assembler;
using VRC.Udon.UAssembly.Interfaces;

namespace Lemolatoon.Udoncl.Editor
{
    internal static class UdonAssemblyCompiler
    {
        internal const uint MaximumHeapCapacity = 1_048_576;
        private const uint DefaultHeapCapacity = 512;

        internal static IUdonProgram Assemble(string assembly)
        {
            uint capacity = DefaultHeapCapacity;
            while (true)
            {
                try
                {
                    UAssemblyAssembler assembler = new UAssemblyAssembler(
                        new FixedHeapFactory(capacity),
                        new EditorTypeResolver()
                    );
                    return assembler.Assemble(assembly);
                }
                catch (IndexOutOfRangeException) when (capacity < MaximumHeapCapacity)
                {
                    capacity = Math.Min(capacity * 2, MaximumHeapCapacity);
                }
            }
        }

        private sealed class FixedHeapFactory : IUdonHeapFactory
        {
            private readonly uint capacity;

            internal FixedHeapFactory(uint capacity)
            {
                this.capacity = capacity;
            }

            public IUdonHeap ConstructUdonHeap()
            {
                return UdonEditorManager.Instance.ConstructUdonHeap(capacity);
            }

            public IUdonHeap ConstructUdonHeap(uint heapSize)
            {
                return UdonEditorManager.Instance.ConstructUdonHeap(capacity);
            }
        }

        private sealed class EditorTypeResolver : IUAssemblyTypeResolver
        {
            public Type GetTypeFromTypeString(string typeString)
            {
                return UdonEditorManager.Instance.GetTypeFromTypeString(typeString);
            }
        }
    }
}
