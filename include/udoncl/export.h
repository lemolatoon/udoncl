#ifndef UDONCL_EXPORT_H
#define UDONCL_EXPORT_H

/*
 * UdonCL Host source annotations.  Clang serializes them into
 * llvm.global.annotations, which udonclc consumes after all Host translation
 * units have been linked.
 *
 * UDONCL_EXPORT marks a Host C function as a public UdonCL export.
 * Suspension is inferred from the call graph; it is deliberately not part of
 * the source annotation.
 *
 * UDONCL_EXPORT_WHILE_BUSY marks an export that the generated U# binding also
 * accepts while another export is suspended, such as a progress or streamed
 * output getter. Such an export must not suspend itself; udonclc rejects one
 * that may.
 *
 * UDONCL_PUBLIC(type) marks a file-scope, non-static cl_mem variable as a
 * public Device buffer of the Unity library, named after the variable, whose
 * elements are float, int or long (float[], int[] or long[] in U#):
 *
 *   UDONCL_PUBLIC(float) cl_mem x, y, result;
 *
 * Its clCreateBuffer flags give its direction: CL_MEM_READ_ONLY is written by
 * U#, CL_MEM_WRITE_ONLY is read by U#. The variable must hold the buffer of
 * exactly one allocation context.
 */
#if defined(__clang__)
#if defined(__has_attribute)
#if !__has_attribute(annotate)
#error "UdonCL requires Clang function annotation support"
#endif
#endif
#define UDONCL_EXPORT __attribute__((annotate("udoncl.export")))
#define UDONCL_EXPORT_WHILE_BUSY                                               \
  __attribute__((annotate("udoncl.export"),                                    \
                 annotate("udoncl.export_while_busy")))
#define UDONCL_PUBLIC(type) __attribute__((annotate("udoncl.public:" #type)))
#else
#define UDONCL_EXPORT
#define UDONCL_EXPORT_WHILE_BUSY
#define UDONCL_PUBLIC(type)
#endif

#endif
