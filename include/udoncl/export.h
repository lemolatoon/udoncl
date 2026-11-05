#ifndef UDONCL_EXPORT_H
#define UDONCL_EXPORT_H

/*
 * Marks a Host C function as a public UdonCL export.  Clang serializes the
 * annotation into llvm.global.annotations, which udonclc consumes after all
 * Host translation units have been linked.  Suspension is inferred from the
 * call graph; it is deliberately not part of the source annotation.
 */
#if defined(__clang__)
#if defined(__has_attribute)
#if __has_attribute(annotate)
#define UDONCL_EXPORT __attribute__((annotate("udoncl.export")))
#else
#error "UdonCL requires Clang function annotation support"
#endif
#else
#define UDONCL_EXPORT __attribute__((annotate("udoncl.export")))
#endif
#else
#define UDONCL_EXPORT
#endif

#endif
