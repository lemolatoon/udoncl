#include "VRC/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include <functional>
#include <string>
#include <tuple>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_DISCOVERSTATICDEVICERESOURCESPASS
#include "VRC/Conversion/Passes.h.inc"

namespace {

constexpr StringLiteral discoveryAttribute =
    "vrc.static_device_resource_discovery";
constexpr StringLiteral allocationSiteAttribute =
    "vrc.static_device_allocation_site";
constexpr StringLiteral callSiteAttribute = "vrc.static_device_call_site";
constexpr StringLiteral publicBindingAttribute =
    "vrc.static_public_resource_binding";

bool isAllocationPrimitive(StringRef name) {
  return name == "clCreateBuffer" || name == "udonclCreateBufferForQueue" ||
         name == "udonclDeviceBufferCreate";
}

struct CallEdge {
  func::CallOp call;
  func::FuncOp callee;
  int32_t occurrence;
  bool inLoop;
};

struct AllocationSite {
  func::CallOp call;
  std::string primitive;
  int32_t occurrence;
  bool inLoop;
};

struct FunctionInfo {
  func::FuncOp function;
  SmallVector<CallEdge> calls;
  SmallVector<AllocationSite> allocations;
  SmallVector<func::CallIndirectOp> indirectCalls;
  bool producesResource = false;
};

struct CallFrame {
  std::string caller;
  std::string callee;
  int32_t occurrence;

  auto key() const { return std::tie(caller, callee, occurrence); }
};

struct AllocationContext {
  std::string root;
  SmallVector<CallFrame> path;
  std::string function;
  std::string primitive;
  int32_t occurrence;
  DictionaryAttr publicBinding;

  auto key() const {
    return std::tie(root, path, function, primitive, occurrence);
  }
};

bool operator<(const CallFrame &left, const CallFrame &right) {
  return left.key() < right.key();
}

bool operator<(const AllocationContext &left, const AllocationContext &right) {
  return left.key() < right.key();
}

bool isInsideStructuredLoop(Operation *operation, func::FuncOp function) {
  for (Operation *parent = operation->getParentOp();
       parent && parent != function; parent = parent->getParentOp())
    if (isa<LoopLikeOpInterface>(parent))
      return true;
  return false;
}

llvm::DenseSet<Block *> findCyclicBlocks(func::FuncOp function) {
  llvm::DenseSet<Block *> cyclic;
  for (Block &start : function.getBody()) {
    SmallVector<Block *> worklist;
    llvm::DenseSet<Block *> visited;
    for (Block *successor : start.getSuccessors())
      worklist.push_back(successor);
    while (!worklist.empty()) {
      Block *block = worklist.pop_back_val();
      if (block == &start) {
        cyclic.insert(&start);
        break;
      }
      if (!visited.insert(block).second)
        continue;
      for (Block *successor : block->getSuccessors())
        worklist.push_back(successor);
    }
  }
  return cyclic;
}

bool isInsideLoop(Operation *operation, func::FuncOp function,
                  const llvm::DenseSet<Block *> &cyclicBlocks) {
  if (isInsideStructuredLoop(operation, function))
    return true;
  Block *block = operation->getBlock();
  return block && block->getParent() == &function.getBody() &&
         cyclicBlocks.contains(block);
}

DictionaryAttr frameAttribute(Builder &builder, const CallFrame &frame) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr(
          "caller", FlatSymbolRefAttr::get(builder.getContext(), frame.caller)),
      builder.getNamedAttr(
          "callee", FlatSymbolRefAttr::get(builder.getContext(), frame.callee)),
      builder.getNamedAttr("occurrence",
                           builder.getI32IntegerAttr(frame.occurrence)),
  });
}

DictionaryAttr contextAttribute(Builder &builder,
                                const AllocationContext &context,
                                int32_t contextId) {
  SmallVector<Attribute> path;
  for (const CallFrame &frame : context.path)
    path.push_back(frameAttribute(builder, frame));
  SmallVector<NamedAttribute> siteFields{
      builder.getNamedAttr(
          "function",
          FlatSymbolRefAttr::get(builder.getContext(), context.function)),
      builder.getNamedAttr("occurrence",
                           builder.getI32IntegerAttr(context.occurrence)),
      builder.getNamedAttr("primitive",
                           builder.getStringAttr(context.primitive)),
  };
  if (context.primitive == "udonclDeviceBufferCreate")
    siteFields.push_back(
        builder.getNamedAttr("resource_operand", builder.getI32IntegerAttr(3)));
  else
    siteFields.push_back(
        builder.getNamedAttr("resource_result", builder.getI32IntegerAttr(0)));
  auto site = builder.getDictionaryAttr(siteFields);
  SmallVector<NamedAttribute> fields{
      builder.getNamedAttr("allocation_site", site),
      builder.getNamedAttr("call_path", builder.getArrayAttr(path)),
      builder.getNamedAttr("context_id", builder.getI32IntegerAttr(contextId)),
      builder.getNamedAttr(
          "root", FlatSymbolRefAttr::get(builder.getContext(), context.root)),
  };
  if (context.publicBinding)
    fields.push_back(
        builder.getNamedAttr("public_binding", context.publicBinding));
  return builder.getDictionaryAttr(fields);
}

struct DiscoverStaticDeviceResourcesPass
    : impl::DiscoverStaticDeviceResourcesPassBase<
          DiscoverStaticDeviceResourcesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());

    llvm::StringMap<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      functions.try_emplace(function.getSymName(), function);

    llvm::StringMap<FunctionInfo> information;
    bool invalid = false;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      FunctionInfo info;
      info.function = function;
      llvm::DenseSet<Block *> cyclicBlocks = findCyclicBlocks(function);
      llvm::StringMap<int32_t> directOccurrences;
      llvm::StringMap<int32_t> allocationOccurrences;
      function.walk([&](Operation *operation) {
        if (auto indirect = dyn_cast<func::CallIndirectOp>(operation)) {
          info.indirectCalls.push_back(indirect);
          return;
        }
        auto call = dyn_cast<func::CallOp>(operation);
        if (!call)
          return;
        call->removeAttr(allocationSiteAttribute);
        call->removeAttr(callSiteAttribute);
        StringRef calleeName = call.getCallee();
        bool inLoop = isInsideLoop(call, function, cyclicBlocks);
        if (isAllocationPrimitive(calleeName)) {
          auto primitive = functions.find(calleeName);
          if (primitive == functions.end() ||
              !primitive->getValue().isExternal()) {
            call.emitError("static device resource discovery requires an "
                           "unlinked OpenCL allocation declaration");
            invalid = true;
            return;
          }
          int32_t occurrence = allocationOccurrences[calleeName]++;
          SmallVector<NamedAttribute> siteFields{
              builder.getNamedAttr(
                  "function", FlatSymbolRefAttr::get(builder.getContext(),
                                                     function.getSymName())),
              builder.getNamedAttr("occurrence",
                                   builder.getI32IntegerAttr(occurrence)),
              builder.getNamedAttr("primitive",
                                   builder.getStringAttr(calleeName)),
          };
          if (calleeName == "udonclDeviceBufferCreate")
            siteFields.push_back(builder.getNamedAttr(
                "resource_operand", builder.getI32IntegerAttr(3)));
          else
            siteFields.push_back(builder.getNamedAttr(
                "resource_result", builder.getI32IntegerAttr(0)));
          call->setAttr(allocationSiteAttribute,
                        builder.getDictionaryAttr(siteFields));
          info.allocations.push_back(
              {call, calleeName.str(), occurrence, inLoop});
          return;
        }
        auto found = functions.find(calleeName);
        if (found == functions.end() || found->getValue().isExternal())
          return;
        int32_t occurrence = directOccurrences[calleeName]++;
        call->setAttr(
            callSiteAttribute,
            builder.getDictionaryAttr({
                builder.getNamedAttr(
                    "callee",
                    FlatSymbolRefAttr::get(builder.getContext(), calleeName)),
                builder.getNamedAttr("occurrence",
                                     builder.getI32IntegerAttr(occurrence)),
            }));
        info.calls.push_back({call, found->getValue(), occurrence, inLoop});
      });
      info.producesResource = !info.allocations.empty();
      information.try_emplace(function.getSymName(), std::move(info));
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &entry : information) {
        FunctionInfo &info = entry.getValue();
        if (info.producesResource)
          continue;
        for (CallEdge &edge : info.calls) {
          auto callee = information.find(edge.callee.getSymName());
          if (callee != information.end() &&
              callee->getValue().producesResource) {
            info.producesResource = true;
            changed = true;
            break;
          }
        }
      }
    }

    const bool hasConfiguredRoots =
        llvm::any_of(module.getOps<func::FuncOp>(), [](func::FuncOp function) {
          return !function.isExternal() && function->hasAttr("vrc.export_id");
        });
    SmallVector<func::FuncOp> roots;
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (!function.isExternal() &&
          (hasConfiguredRoots ? function->hasAttr("vrc.export_id")
                              : function->hasAttr("udoncl.export")))
        roots.push_back(function);
    llvm::sort(roots, [](func::FuncOp left, func::FuncOp right) {
      return left.getSymName() < right.getSymName();
    });

    llvm::DenseSet<Operation *> validated;
    std::function<void(func::FuncOp)> validateCallGraph =
        [&](func::FuncOp function) {
          if (invalid || !validated.insert(function).second)
            return;
          auto found = information.find(function.getSymName());
          if (found == information.end())
            return;
          FunctionInfo &info = found->getValue();
          if (!info.indirectCalls.empty()) {
            info.indirectCalls.front().emitError(
                "reachable indirect call prevents complete static device "
                "resource discovery");
            invalid = true;
            return;
          }
          for (CallEdge &edge : info.calls)
            validateCallGraph(edge.callee);
        };
    for (func::FuncOp root : roots)
      validateCallGraph(root);
    if (invalid) {
      signalPassFailure();
      return;
    }

    SmallVector<AllocationContext> contexts;
    SmallVector<func::FuncOp> active;
    SmallVector<CallFrame> path;
    SmallVector<DictionaryAttr> publicBindings;
    std::function<void(func::FuncOp, StringRef)> visit =
        [&](func::FuncOp function, StringRef root) {
          if (invalid)
            return;
          auto found = information.find(function.getSymName());
          if (found == information.end())
            return;
          FunctionInfo &info = found->getValue();
          if (llvm::is_contained(active, function)) {
            function.emitError(
                "reachable resource-producing recursion is unsupported");
            invalid = true;
            return;
          }
          active.push_back(function);
          for (AllocationSite &site : info.allocations) {
            if (site.inLoop) {
              site.call.emitError(
                  "static device allocation in an unproved loop is "
                  "unsupported");
              invalid = true;
              break;
            }
            auto siteBinding = site.call->getAttrOfType<DictionaryAttr>(
                publicBindingAttribute);
            if ((siteBinding && !publicBindings.empty()) ||
                publicBindings.size() > 1) {
              site.call.emitError(
                  "allocation context has ambiguous public resource bindings");
              invalid = true;
              break;
            }
            DictionaryAttr publicBinding =
                siteBinding ? siteBinding
                            : (publicBindings.empty() ? DictionaryAttr()
                                                      : publicBindings.back());
            contexts.push_back({root.str(), path, function.getSymName().str(),
                                site.primitive, site.occurrence,
                                publicBinding});
          }
          if (!invalid)
            for (CallEdge &edge : info.calls) {
              auto callee = information.find(edge.callee.getSymName());
              if (callee == information.end() ||
                  !callee->getValue().producesResource)
                continue;
              if (edge.inLoop) {
                edge.call.emitError(
                    "resource-producing call in an unproved loop is "
                    "unsupported");
                invalid = true;
                break;
              }
              path.push_back({function.getSymName().str(),
                              edge.callee.getSymName().str(), edge.occurrence});
              auto binding = edge.call->getAttrOfType<DictionaryAttr>(
                  publicBindingAttribute);
              if (binding)
                publicBindings.push_back(binding);
              visit(edge.callee, root);
              if (binding)
                publicBindings.pop_back();
              path.pop_back();
              if (invalid)
                break;
            }
          active.pop_back();
        };

    for (func::FuncOp root : roots) {
      auto info = information.find(root.getSymName());
      if (info != information.end() && info->getValue().producesResource)
        visit(root, root.getSymName());
      if (invalid)
        break;
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    llvm::sort(contexts);
    SmallVector<Attribute> contextAttributes;
    for (auto [index, context] : llvm::enumerate(contexts))
      contextAttributes.push_back(
          contextAttribute(builder, context, static_cast<int32_t>(index)));
    module->setAttr(
        discoveryAttribute,
        builder.getDictionaryAttr({
            builder.getNamedAttr("contexts",
                                 builder.getArrayAttr(contextAttributes)),
            builder.getNamedAttr("schema_version",
                                 builder.getI32IntegerAttr(1)),
        }));
  }
};

} // namespace
} // namespace vrc
