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
#else
#define UDONCL_EXPORT
#define UDONCL_EXPORT_WHILE_BUSY
#endif

#endif
