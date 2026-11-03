#include "VRC/Conversion/LLVMToHighLevel/Passes.h"
#include "VRC/Dialect/UdonHL/UdonHLDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"
#include <cstring>

using namespace mlir;

namespace vrc {
#define GEN_PASS_DEF_RAISELLVMRAWMEMORYPASS
#include "VRC/Conversion/LLVMToHighLevel/Passes.h.inc"

namespace {
bool containsPointer(Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return containsPointer(array.getElementType());
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type))
    return llvm::any_of(structure.getBody(), containsPointer);
  return false;
}

::vrc::udonhl::PtrType pointerType(MLIRContext *context,
                                   LLVM::LLVMPointerType type) {
  return ::vrc::udonhl::PtrType::get(context, type.getAddressSpace());
}

FailureOr<uint64_t> abiAlignment(const DataLayout &layout, Type type);
FailureOr<uint64_t> structFieldOffset(const DataLayout &layout,
                                      LLVM::LLVMStructType structure,
                                      unsigned selected);

FailureOr<uint64_t> fixedSize(const DataLayout &layout, Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return uint64_t{4};
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
    auto elementSize = fixedSize(layout, array.getElementType());
    auto elementAlignment = abiAlignment(layout, array.getElementType());
    if (failed(elementSize) || failed(elementAlignment))
      return failure();
    uint64_t stride = llvm::alignTo(*elementSize, *elementAlignment);
    if (array.getNumElements() > INT32_MAX / stride)
      return failure();
    return array.getNumElements() * stride;
  }
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structure.isOpaque())
      return failure();
    uint64_t offset = 0;
    uint64_t structureAlignment = structure.isPacked() ? 1 : 0;
    for (Type field : structure.getBody()) {
      auto fieldSize = fixedSize(layout, field);
      auto fieldAlignment = abiAlignment(layout, field);
      if (failed(fieldSize) || failed(fieldAlignment))
        return failure();
      if (!structure.isPacked()) {
        offset = llvm::alignTo(offset, *fieldAlignment);
        structureAlignment = std::max(structureAlignment, *fieldAlignment);
      }
      if (*fieldSize > INT32_MAX - offset)
        return failure();
      offset += *fieldSize;
    }
    return llvm::alignTo(offset, structureAlignment);
  }
  auto size = layout.getTypeSize(type);
  uint64_t alignment = layout.getTypeABIAlignment(type);
  if (size.isScalable() || !alignment || size.getFixedValue() > INT32_MAX)
    return failure();
  return llvm::alignTo(size.getFixedValue(), alignment);
}

FailureOr<uint64_t> abiAlignment(const DataLayout &layout, Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return uint64_t{4};
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return abiAlignment(layout, array.getElementType());
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structure.isOpaque())
      return failure();
    if (structure.isPacked())
      return uint64_t{1};
    uint64_t alignment = 1;
    for (Type field : structure.getBody()) {
      auto fieldAlignment = abiAlignment(layout, field);
      if (failed(fieldAlignment))
        return failure();
      alignment = std::max(alignment, *fieldAlignment);
    }
    return alignment;
  }
  uint64_t alignment = layout.getTypeABIAlignment(type);
  return alignment ? FailureOr<uint64_t>(alignment) : failure();
}

LogicalResult reject(Operation *operation, StringRef reason) {
  return operation->emitError() << "cannot recover raw memory for "
                                << operation->getName() << ": " << reason;
}

bool hasPointerSignature(LLVM::LLVMFuncOp function) {
  return llvm::any_of(function.getArgumentTypes(), containsPointer) ||
         llvm::any_of(function.getResultTypes(), containsPointer);
}

bool hasPointerSignature(LLVM::CallOp call) {
  return llvm::any_of(call.getOperandTypes(), containsPointer) ||
         llvm::any_of(call.getResultTypes(), containsPointer);
}

struct FunctionPattern : OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> arguments, results;
    if (failed(getTypeConverter()->convertTypes(op.getArgumentTypes(),
                                                arguments)) ||
        failed(getTypeConverter()->convertTypes(op.getResultTypes(), results)))
      return failure();
    auto function =
        func::FuncOp::create(rewriter, op.getLoc(), op.getSymName(),
                             rewriter.getFunctionType(arguments, results));
    if (op.isExternal() || op.getLinkage() != LLVM::Linkage::External)
      function.setPrivate();
    function->setAttr("vrc.llvm_function",
                      rewriter.getDictionaryAttr(op->getAttrs()));
    for (NamedAttribute attribute : op->getAttrs())
      if (attribute.getName().getValue().starts_with("vrc."))
        function->setAttr(attribute.getName(), attribute.getValue());
    rewriter.inlineRegionBefore(op.getBody(), function.getBody(),
                                function.end());
    if (failed(rewriter.convertRegionTypes(&function.getBody(),
                                           *getTypeConverter())))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct CallPattern : OpConversionPattern<LLVM::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getCallee() || op.getCConv() != LLVM::cconv::CConv::C ||
        op.getTailCallKind() != LLVM::TailCallKind::None ||
        !op.getOpBundleSizes().empty() || op.getVarCalleeType())
      return failure();
    SmallVector<Type> results;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), results)))
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, *op.getCallee(), results,
                                              adaptor.getOperands());
    return success();
  }
};

struct ReturnPattern : OpConversionPattern<LLVM::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

FailureOr<Attribute> zeroInitializer(OpBuilder &builder, Type storageType) {
  if (isa<::vrc::udonhl::PtrType>(storageType))
    return builder.getI32IntegerAttr(0);
  if (auto integer = dyn_cast<IntegerType>(storageType))
    return builder.getIntegerAttr(integer, 0);
  if (auto floating = dyn_cast<FloatType>(storageType))
    return builder.getFloatAttr(floating, 0.0);
  // Aggregate BSS is represented explicitly rather than pretending an LLVM
  // aggregate attribute has the UdonHL storage type.
  return builder.getUnitAttr();
}

LogicalResult serializeInteger(llvm::APInt value, MutableArrayRef<int8_t> out) {
  if (value.getBitWidth() > out.size() * 8)
    return failure();
  value = value.zextOrTrunc(out.size() * 8);
  for (unsigned index = 0; index < out.size(); ++index)
    out[index] =
        static_cast<int8_t>(value.extractBitsAsZExtValue(8, index * 8));
  return success();
}

LogicalResult serializeAttribute(Attribute attribute, Type type,
                                 const DataLayout &layout,
                                 MutableArrayRef<int8_t> out) {
  if (auto integer = dyn_cast<IntegerAttr>(attribute))
    return serializeInteger(integer.getValue(), out);
  if (auto floating = dyn_cast<FloatAttr>(attribute))
    return serializeInteger(floating.getValue().bitcastToAPInt(), out);
  if (isa<LLVM::ZeroAttr>(attribute)) {
    llvm::fill(out, 0);
    return success();
  }
  if (auto text = dyn_cast<StringAttr>(attribute)) {
    if (text.getValue().size() > out.size())
      return failure();
    llvm::fill(out, 0);
    std::memcpy(out.data(), text.getValue().data(), text.getValue().size());
    return success();
  }
  auto elements = dyn_cast<ElementsAttr>(attribute);
  auto array = dyn_cast<LLVM::LLVMArrayType>(type);
  if (!elements || !array ||
      elements.getNumElements() != array.getNumElements())
    return failure();
  auto elementSize = fixedSize(layout, array.getElementType());
  auto elementAlignment = abiAlignment(layout, array.getElementType());
  if (failed(elementSize) || failed(elementAlignment))
    return failure();
  uint64_t stride = llvm::alignTo(*elementSize, *elementAlignment);
  unsigned index = 0;
  for (Attribute element : elements.getValues<Attribute>()) {
    if (failed(serializeAttribute(element, array.getElementType(), layout,
                                  out.slice(index * stride, *elementSize))))
      return failure();
    ++index;
  }
  return success();
}

FailureOr<std::pair<uint64_t, Type>>
aggregateElement(Type type, ArrayRef<int64_t> position,
                 const DataLayout &layout) {
  uint64_t offset = 0;
  for (int64_t selected : position) {
    if (selected < 0)
      return failure();
    if (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
      if (static_cast<uint64_t>(selected) >= array.getNumElements())
        return failure();
      auto size = fixedSize(layout, array.getElementType());
      auto alignment = abiAlignment(layout, array.getElementType());
      if (failed(size) || failed(alignment))
        return failure();
      offset +=
          static_cast<uint64_t>(selected) * llvm::alignTo(*size, *alignment);
      type = array.getElementType();
    } else if (auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
      auto field =
          structFieldOffset(layout, structure, static_cast<unsigned>(selected));
      if (failed(field))
        return failure();
      offset += *field;
      type = structure.getBody()[selected];
    } else {
      return failure();
    }
  }
  return std::pair<uint64_t, Type>{offset, type};
}

LogicalResult serializeValue(Value value, Type type, const DataLayout &layout,
                             MutableArrayRef<int8_t> out) {
  if (isa_and_nonnull<LLVM::UndefOp>(value.getDefiningOp()) ||
      isa_and_nonnull<LLVM::ZeroOp>(value.getDefiningOp())) {
    llvm::fill(out, 0);
    return success();
  }
  if (auto constant = value.getDefiningOp<LLVM::ConstantOp>())
    return serializeAttribute(constant.getValue(), type, layout, out);
  auto insert = value.getDefiningOp<LLVM::InsertValueOp>();
  if (!insert ||
      failed(serializeValue(insert.getContainer(), type, layout, out)))
    return failure();
  auto selected = aggregateElement(type, insert.getPosition(), layout);
  if (failed(selected))
    return failure();
  auto size = fixedSize(layout, selected->second);
  if (failed(size) || selected->first > out.size() ||
      *size > out.size() - selected->first)
    return failure();
  return serializeValue(insert.getValue(), selected->second, layout,
                        out.slice(selected->first, *size));
}

FailureOr<Attribute> globalInitializer(LLVM::GlobalOp op, Type storageType,
                                       OpBuilder &builder,
                                       const DataLayout &layout) {
  bool aggregate =
      isa<LLVM::LLVMArrayType, LLVM::LLVMStructType>(op.getGlobalType());
  if (aggregate) {
    auto size = fixedSize(layout, op.getGlobalType());
    if (failed(size))
      return failure();
    SmallVector<int8_t> bytes(*size, 0);
    if (auto value = op.getValue()) {
      if (failed(serializeAttribute(*value, op.getGlobalType(), layout, bytes)))
        return failure();
    } else if (Block *block = op.getInitializerBlock()) {
      auto returned = dyn_cast<LLVM::ReturnOp>(block->getTerminator());
      if (!returned || returned.getNumOperands() != 1 ||
          failed(serializeValue(returned.getOperand(0), op.getGlobalType(),
                                layout, bytes)))
        return failure();
    }
    return builder.getDenseI8ArrayAttr(bytes);
  }
  if (auto value = op.getValue()) {
    if (isa<LLVM::ZeroAttr>(*value))
      return zeroInitializer(builder, storageType);
    if (auto typed = dyn_cast<TypedAttr>(*value))
      return Attribute(typed);
    return failure();
  }
  Block *block = op.getInitializerBlock();
  if (!block)
    return Attribute();
  auto returned = dyn_cast<LLVM::ReturnOp>(block->getTerminator());
  if (!returned || returned.getNumOperands() != 1)
    return failure();
  Operation *value = returned.getOperand(0).getDefiningOp();
  if (isa_and_nonnull<LLVM::ZeroOp>(value))
    return zeroInitializer(builder, storageType);
  if (auto constant = dyn_cast_or_null<LLVM::ConstantOp>(value))
    return constant.getValue();
  return failure();
}

struct GlobalPattern : OpConversionPattern<LLVM::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef linkage;
    if (op.getLinkage() == LLVM::Linkage::Internal ||
        op.getLinkage() == LLVM::Linkage::Private)
      linkage = "internal";
    else if (op.getLinkage() == LLVM::Linkage::External)
      linkage = "external";
    else
      return reject(op, "global linkage is not implemented");
    Type storageType = getTypeConverter()->convertType(op.getGlobalType());
    if (!storageType)
      return failure();
    DataLayout layout(op->getParentOfType<ModuleOp>());
    auto initial = globalInitializer(op, storageType, rewriter, layout);
    if (failed(initial))
      return reject(op, "global initializer is not a constant or zero value");
    auto naturalAlignment = abiAlignment(layout, op.getGlobalType());
    if (failed(naturalAlignment))
      return reject(op, "global alignment is not representable by ILP32");
    uint64_t alignment = op.getAlignment().value_or(*naturalAlignment);

    OperationState state(op.getLoc(),
                         ::vrc::udonhl::GlobalOp::getOperationName());
    state.addAttribute("sym_name", rewriter.getStringAttr(op.getSymName()));
    state.addAttribute("storage_type", TypeAttr::get(storageType));
    if (*initial)
      state.addAttribute("initial_value", *initial);
    state.addAttribute("linkage", rewriter.getStringAttr(linkage));
    state.addAttribute("address_space",
                       rewriter.getI32IntegerAttr(op.getAddrSpace()));
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(alignment));
    if (op.getConstant())
      state.addAttribute("constant", rewriter.getUnitAttr());
    rewriter.create(state);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AddressPattern : OpConversionPattern<LLVM::AddressOfOp> {
  AddressPattern(TypeConverter &converter, MLIRContext *context,
                 const llvm::StringMap<int64_t> &addressSpaces)
      : OpConversionPattern(converter, context), addressSpaces(addressSpaces) {}

  LogicalResult
  matchAndRewrite(LLVM::AddressOfOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto found = addressSpaces.find(op.getGlobalName());
    if (found == addressSpaces.end())
      return reject(op, "address_of does not resolve to a converted global");
    auto resultType = pointerType(rewriter.getContext(), op.getType());
    if (static_cast<int64_t>(resultType.getAddressSpace()) != found->second)
      return reject(op, "address_of pointer and global address spaces differ");
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::AddressOfOp::getOperationName());
    state.addTypes(resultType);
    state.addAttribute("global", op.getGlobalNameAttr());
    auto *address = rewriter.create(state);
    rewriter.replaceOp(op, address->getResults());
    return success();
  }

  const llvm::StringMap<int64_t> &addressSpaces;
};

struct AllocaPattern : OpConversionPattern<LLVM::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AllocaOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto count = op.getArraySize().getDefiningOp<LLVM::ConstantOp>();
    auto countValue =
        count ? dyn_cast<IntegerAttr>(count.getValue()) : IntegerAttr();
    DataLayout layout(op->getParentOfType<ModuleOp>());
    auto stride = fixedSize(layout, op.getElemType());
    if (!countValue || countValue.getValue().isNegative() || failed(stride))
      return reject(op, "raw alloca size must be a nonnegative constant");
    auto bytes = llvm::checkedMul(
        static_cast<int64_t>(countValue.getValue().getZExtValue()),
        static_cast<int64_t>(*stride));
    if (!bytes || *bytes > INT32_MAX)
      return reject(op, "raw alloca exceeds the 32-bit address space");
    auto naturalAlignment = abiAlignment(layout, op.getElemType());
    if (failed(naturalAlignment))
      return reject(op, "raw alloca alignment is not representable");
    uint64_t alignment = op.getAlignment().value_or(*naturalAlignment);
    auto byteSize = arith::ConstantIntOp::create(
        rewriter, op.getLoc(), static_cast<int64_t>(*bytes), 32);
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::AllocaOp::getOperationName());
    state.addOperands(byteSize.getResult());
    state.addTypes(pointerType(rewriter.getContext(), op.getType()));
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(alignment));
    auto *allocation = rewriter.create(state);
    rewriter.replaceOp(op, allocation->getResults());
    return success();
  }
};

FailureOr<uint64_t> structFieldOffset(const DataLayout &layout,
                                      LLVM::LLVMStructType structure,
                                      unsigned selected) {
  if (structure.isOpaque() || selected >= structure.getBody().size())
    return failure();
  uint64_t offset = 0;
  for (unsigned index = 0; index <= selected; ++index) {
    Type field = structure.getBody()[index];
    if (!structure.isPacked()) {
      auto alignment = abiAlignment(layout, field);
      if (failed(alignment))
        return failure();
      offset = llvm::alignTo(offset, *alignment);
    }
    if (index == selected)
      return offset;
    auto size = fixedSize(layout, field);
    if (failed(size))
      return failure();
    offset += *size;
  }
  return failure();
}

struct GEPPattern : OpConversionPattern<LLVM::GEPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GEPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DataLayout layout(op->getParentOfType<ModuleOp>());
    Type current = op.getElemType();
    int64_t staticOffset = 0;
    Value dynamicOffset;
    bool first = true;
    unsigned dynamicIndex = 1;
    for (auto index : op.getIndices()) {
      uint64_t stride = 0;
      if (first) {
        auto size = fixedSize(layout, current);
        if (failed(size))
          return reject(op, "GEP element size is not fixed and 32-bit");
        stride = *size;
        first = false;
      } else if (auto array = dyn_cast<LLVM::LLVMArrayType>(current)) {
        current = array.getElementType();
        auto size = fixedSize(layout, current);
        if (failed(size))
          return reject(op, "GEP array element size is not fixed and 32-bit");
        stride = *size;
      } else if (auto structure = dyn_cast<LLVM::LLVMStructType>(current)) {
        auto attribute = dyn_cast<IntegerAttr>(index);
        if (!attribute || attribute.getInt() < 0)
          return reject(op, "GEP struct indices must be nonnegative constants");
        unsigned field = static_cast<unsigned>(attribute.getInt());
        auto offset = structFieldOffset(layout, structure, field);
        if (failed(offset) || *offset > INT32_MAX)
          return reject(op, "GEP struct field offset is outside 32-bit memory");
        staticOffset += *offset;
        current = structure.getBody()[field];
        continue;
      } else {
        return reject(op, "GEP descent requires an array or struct type");
      }

      if (auto attribute = dyn_cast<IntegerAttr>(index)) {
        auto contribution =
            llvm::checkedMul(attribute.getInt(), static_cast<int64_t>(stride));
        if (!contribution)
          return reject(op, "GEP constant byte offset overflows");
        staticOffset += *contribution;
        continue;
      }
      Value value = adaptor.getOperands()[dynamicIndex++];
      if (!value.getType().isInteger(32))
        return reject(op, "dynamic GEP indices must use the 32-bit Udon ABI");
      if (stride != 1) {
        Value scale = arith::ConstantIntOp::create(
            rewriter, op.getLoc(), static_cast<int64_t>(stride), 32);
        value = arith::MulIOp::create(rewriter, op.getLoc(), value, scale);
      }
      dynamicOffset = dynamicOffset
                          ? arith::AddIOp::create(rewriter, op.getLoc(),
                                                  dynamicOffset, value)
                                .getResult()
                          : value;
    }
    if (staticOffset < INT32_MIN || staticOffset > INT32_MAX)
      return reject(op, "GEP static byte offset is outside signed i32");
    if (staticOffset || !dynamicOffset) {
      Value fixed =
          arith::ConstantIntOp::create(rewriter, op.getLoc(), staticOffset, 32);
      dynamicOffset = dynamicOffset
                          ? arith::AddIOp::create(rewriter, op.getLoc(),
                                                  dynamicOffset, fixed)
                                .getResult()
                          : fixed;
    }
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::PtrAddOp::getOperationName());
    state.addOperands({adaptor.getBase(), dynamicOffset});
    state.addTypes(pointerType(rewriter.getContext(),
                               cast<LLVM::LLVMPointerType>(op.getType())));
    auto *result = rewriter.create(state);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

struct LoadPattern : OpConversionPattern<LLVM::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVolatile_() ||
        op.getOrdering() != LLVM::AtomicOrdering::not_atomic)
      return reject(op, "volatile and atomic raw loads are not implemented");
    Type result = getTypeConverter()->convertType(op.getType());
    if (!result)
      return failure();
    DataLayout layout(op->getParentOfType<ModuleOp>());
    auto naturalAlignment = abiAlignment(layout, op.getType());
    if (failed(naturalAlignment))
      return reject(op, "raw load alignment is not representable");
    uint64_t alignment = op.getAlignment().value_or(*naturalAlignment);
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::LoadOp::getOperationName());
    state.addOperands(adaptor.getAddr());
    state.addTypes(result);
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(alignment));
    auto *load = rewriter.create(state);
    rewriter.replaceOp(op, load->getResults());
    return success();
  }
};

struct StorePattern : OpConversionPattern<LLVM::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVolatile_() ||
        op.getOrdering() != LLVM::AtomicOrdering::not_atomic)
      return reject(op, "volatile and atomic raw stores are not implemented");
    DataLayout layout(op->getParentOfType<ModuleOp>());
    auto naturalAlignment = abiAlignment(layout, op.getValue().getType());
    if (failed(naturalAlignment))
      return reject(op, "raw store alignment is not representable");
    uint64_t alignment = op.getAlignment().value_or(*naturalAlignment);
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::StoreOp::getOperationName());
    state.addOperands({adaptor.getValue(), adaptor.getAddr()});
    state.addAttribute("alignment", rewriter.getI64IntegerAttr(alignment));
    rewriter.create(state);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PtrToIntPattern : OpConversionPattern<LLVM::PtrToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isInteger(32))
      return reject(op, "ptrtoint result must use the 32-bit Udon ABI");
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::PtrToIntOp::getOperationName());
    state.addOperands(adaptor.getArg());
    state.addTypes(rewriter.getI32Type());
    auto *result = rewriter.create(state);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

struct IntToPtrPattern : OpConversionPattern<LLVM::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getArg().getType().isInteger(32))
      return reject(op, "inttoptr input must use the 32-bit Udon ABI");
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::IntToPtrOp::getOperationName());
    state.addOperands(adaptor.getArg());
    state.addTypes(pointerType(rewriter.getContext(),
                               cast<LLVM::LLVMPointerType>(op.getType())));
    auto *result = rewriter.create(state);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

struct NullPointerPattern : OpConversionPattern<LLVM::ZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ZeroOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pointer = dyn_cast<LLVM::LLVMPointerType>(op.getType());
    if (!pointer)
      return failure();
    Value zero = arith::ConstantIntOp::create(rewriter, op.getLoc(), 0, 32);
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::IntToPtrOp::getOperationName());
    state.addOperands(zero);
    state.addTypes(pointerType(rewriter.getContext(), pointer));
    auto *result = rewriter.create(state);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

struct PointerComparePattern : OpConversionPattern<LLVM::ICmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<::vrc::udonhl::PtrType>(adaptor.getLhs().getType()) ||
        !isa<::vrc::udonhl::PtrType>(adaptor.getRhs().getType()))
      return failure();
    arith::CmpIPredicate predicate;
    switch (op.getPredicate()) {
    case LLVM::ICmpPredicate::eq:
      predicate = arith::CmpIPredicate::eq;
      break;
    case LLVM::ICmpPredicate::ne:
      predicate = arith::CmpIPredicate::ne;
      break;
    case LLVM::ICmpPredicate::ult:
      predicate = arith::CmpIPredicate::ult;
      break;
    case LLVM::ICmpPredicate::ule:
      predicate = arith::CmpIPredicate::ule;
      break;
    case LLVM::ICmpPredicate::ugt:
      predicate = arith::CmpIPredicate::ugt;
      break;
    case LLVM::ICmpPredicate::uge:
      predicate = arith::CmpIPredicate::uge;
      break;
    default:
      return reject(op, "pointer comparison predicate is not implemented");
    }
    auto toInteger = [&](Value pointer) {
      OperationState state(op.getLoc(),
                           ::vrc::udonhl::PtrToIntOp::getOperationName());
      state.addOperands(pointer);
      state.addTypes(rewriter.getI32Type());
      return rewriter.create(state)->getResult(0);
    };
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, predicate,
                                               toInteger(adaptor.getLhs()),
                                               toInteger(adaptor.getRhs()));
    return success();
  }
};

struct PointerBranchPattern : OpConversionPattern<LLVM::BrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::BrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct PointerCondBranchPattern : OpConversionPattern<LLVM::CondBrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    return success();
  }
};

struct MemcpyPattern : OpConversionPattern<LLVM::MemcpyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::MemcpyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIsVolatile())
      return reject(op, "volatile memcpy is not implemented");
    if (!adaptor.getLen().getType().isInteger(32))
      return reject(op, "memcpy length must use the 32-bit Udon ABI");
    OperationState state(op.getLoc(),
                         ::vrc::udonhl::MemcpyOp::getOperationName());
    state.addOperands({adaptor.getDst(), adaptor.getSrc(), adaptor.getLen()});
    rewriter.create(state);
    rewriter.eraseOp(op);
    return success();
  }
};

struct RaiseLLVMRawMemoryPass
    : impl::RaiseLLVMRawMemoryPassBase<RaiseLLVMRawMemoryPass> {
  void runOnOperation() override {
    llvm::StringMap<int64_t> globalAddressSpaces;
    for (LLVM::GlobalOp global : getOperation().getOps<LLVM::GlobalOp>())
      globalAddressSpaces.try_emplace(global.getSymName(),
                                      global.getAddrSpace());
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](LLVM::LLVMPointerType type) -> Type {
      return pointerType(&getContext(), type);
    });

    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target
        .addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                         cf::ControlFlowDialect, func::FuncDialect,
                         memref::MemRefDialect, ::vrc::udonhl::UdonHLDialect>();
    target.addIllegalOp<LLVM::AllocaOp, LLVM::GEPOp, LLVM::LoadOp,
                        LLVM::StoreOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp,
                        LLVM::GlobalOp, LLVM::AddressOfOp>();
    target.addIllegalOp<LLVM::MemcpyOp>();
    target.addDynamicallyLegalOp<LLVM::ZeroOp>([](LLVM::ZeroOp op) {
      return !isa<LLVM::LLVMPointerType>(op.getType());
    });
    target.addDynamicallyLegalOp<LLVM::ICmpOp>([](LLVM::ICmpOp op) {
      return !isa<LLVM::LLVMPointerType>(op.getLhs().getType());
    });
    target.addDynamicallyLegalOp<LLVM::BrOp>([](LLVM::BrOp op) {
      return llvm::none_of(op.getOperandTypes(), containsPointer);
    });
    target.addDynamicallyLegalOp<LLVM::CondBrOp>([](LLVM::CondBrOp op) {
      return llvm::none_of(op.getOperandTypes(), containsPointer);
    });
    target.addIllegalOp<LLVM::LLVMFuncOp, LLVM::CallOp, LLVM::ReturnOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<FunctionPattern, CallPattern, ReturnPattern, GlobalPattern,
                 AllocaPattern, GEPPattern, LoadPattern, StorePattern,
                 PtrToIntPattern, IntToPtrPattern>(converter, &getContext());
    patterns.add<NullPointerPattern, PointerComparePattern,
                 PointerBranchPattern, PointerCondBranchPattern>(converter,
                                                                 &getContext());
    patterns.add<MemcpyPattern>(converter, &getContext());
    patterns.add<AddressPattern>(converter, &getContext(), globalAddressSpaces);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace vrc
