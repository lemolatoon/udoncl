#ifndef VRC_SUPPORT_HOSTLITERAL_H
#define VRC_SUPPORT_HOSTLITERAL_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace vrc {

struct HostLiteralKey {
  std::string id;
  std::string physicalType;
  std::string value;
  bool supported = false;
  llvm::SmallVector<llvm::StringRef, 4> contextFreeRecipes;
};

inline std::string hostLiteralTypeText(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  stream.flush();
  return text;
}

inline llvm::StringRef hostLiteralPhysicalType(mlir::Type type) {
  if (type.isInteger(1))
    return "SystemBoolean";
  if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
      type.isIndex())
    return "SystemInt32";
  if (type.isInteger(64))
    return "SystemInt64";
  if (type.isF32())
    return "SystemSingle";
  return "unsupported";
}

inline std::string hostLiteralPhysicalValue(mlir::Attribute value,
                                            mlir::Type type) {
  llvm::APInt bits;
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
    bits = integer.getValue();
    if (type.isInteger(1))
      bits = bits.zextOrTrunc(1);
    else if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
             type.isIndex())
      bits = bits.sextOrTrunc(32);
  } else if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(value)) {
    bits = floating.getValue().bitcastToAPInt();
  } else {
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.print(stream);
    stream.flush();
    return "attr:" + text;
  }
  llvm::SmallString<32> hexadecimal;
  bits.toStringUnsigned(hexadecimal, 16);
  return "0x" + hexadecimal.str().str();
}

inline HostLiteralKey getHostLiteralKey(mlir::Type type,
                                        mlir::Attribute value) {
  HostLiteralKey key;
  llvm::StringRef physical = hostLiteralPhysicalType(type);
  key.physicalType = physical.str();
  key.value = hostLiteralPhysicalValue(value, type);
  key.supported = physical != "unsupported";
  key.id = key.supported
               ? (physical + ":" + key.value).str()
               : (physical + ":" + hostLiteralTypeText(type) + ":" + key.value)
                     .str();

  if (key.supported)
    key.contextFreeRecipes.push_back("resident_shared");
  if (type.isInteger(1))
    key.contextFreeRecipes.push_back("rematerialize_i1");
  else if (type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
           type.isIndex())
    key.contextFreeRecipes.append({"raw_pool", "rematerialize_i32"});
  else if (type.isF32())
    key.contextFreeRecipes.append({"raw_pool", "rematerialize_f32_bits"});
  else if (type.isInteger(64))
    key.contextFreeRecipes.push_back("rematerialize_i64");
  return key;
}

} // namespace vrc

#endif // VRC_SUPPORT_HOSTLITERAL_H
