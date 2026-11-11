#include "VRC/ABI/Manifest/BindingSource.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <set>

using namespace llvm;

namespace vrc {
namespace {

Error invalid(const Twine &message) {
  return createStringError(inconvertibleErrorCode(), message);
}

bool isNamespace(StringRef value) {
  if (value.empty())
    return false;
  SmallVector<StringRef> components;
  value.split(components, '.');
  return !components.empty() &&
         llvm::all_of(components, [](StringRef component) {
           return isUdonSharpIdentifier(component);
         });
}

Expected<StringRef> csharpType(StringRef kind) {
  if (kind == "i32")
    return StringRef("int");
  if (kind == "f32")
    return StringRef("float");
  if (kind == "i32_array")
    return StringRef("int[]");
  if (kind == "f32_array")
    return StringRef("float[]");
  if (kind == "string")
    return StringRef("string");
  return invalid(Twine("unsupported U# export kind: ") + kind);
}

std::string escapeString(StringRef value) {
  std::string result;
  raw_string_ostream output(result);
  printEscapedString(value, output);
  return result;
}

struct AsyncCommands {
  std::string pendingObject;
  std::string completionObject;
  std::string completionStatus;
  std::string completionEvent;
  std::string commandKind;
  std::string commandBuffer;
  std::string commandOffset;
  std::string commandSize;
  std::string commandPointer;
  std::string rawHeap;
  int64_t uploadKind = 0;
  int64_t readbackKind = 0;
};

struct AsyncExportBinding {
  int64_t id = 0;
  std::string publicName;
  std::string statusSlot;
  std::optional<std::string> resultType;
  std::optional<std::string> resultSlot;
};

struct PublicBufferBinding {
  int64_t id = 0;
  int64_t words = 0;
  int64_t elements = 0;
  std::string suffix;
  std::string elementType;
  std::string access;
};

std::string publicIoName(StringRef name, unsigned index) {
  std::string result;
  bool capitalize = true;
  for (unsigned char character : name.bytes()) {
    bool lower = character >= 'a' && character <= 'z';
    bool upper = character >= 'A' && character <= 'Z';
    bool digit = character >= '0' && character <= '9';
    if (!lower && !upper && !digit) {
      capitalize = true;
      continue;
    }
    if (result.empty() && digit)
      result += "Value";
    result.push_back(capitalize && lower ? character - 'a' + 'A' : character);
    capitalize = false;
  }
  if (result.empty())
    result = "Value" + std::to_string(index);
  return result;
}

Expected<AsyncCommands> parseAsyncCommands(const json::Object &manifest) {
  const json::Object *program = manifest.getObject("program");
  const json::Object *commands =
      program ? program->getObject("async_commands") : nullptr;
  if (!commands)
    return invalid("async binding requires program.async_commands");
  auto pendingObject = commands->getString("pending_object_slot");
  auto completionObject = commands->getString("completion_object_slot");
  auto completionStatus = commands->getString("completion_status_slot");
  auto completionEvent = commands->getString("completion_event");
  auto commandKind = commands->getString("command_kind_slot");
  auto commandBuffer = commands->getString("command_buffer_slot");
  auto commandOffset = commands->getString("command_offset_slot");
  auto commandSize = commands->getString("command_size_slot");
  auto commandPointer = commands->getString("command_pointer_slot");
  auto rawHeap = commands->getString("raw_heap_slot");
  auto uploadKind = commands->getInteger("upload_kind");
  auto readbackKind = commands->getInteger("readback_kind");
  if (!pendingObject || !completionObject || !completionStatus ||
      !completionEvent || !commandKind || !commandBuffer || !commandOffset ||
      !commandSize || !commandPointer || !rawHeap || !uploadKind ||
      !readbackKind || *uploadKind <= 0 || *readbackKind <= 0 ||
      *uploadKind == *readbackKind)
    return invalid("async binding command ABI is incomplete");
  return AsyncCommands{pendingObject->str(),
                       completionObject->str(),
                       completionStatus->str(),
                       completionEvent->str(),
                       commandKind->str(),
                       commandBuffer->str(),
                       commandOffset->str(),
                       commandSize->str(),
                       commandPointer->str(),
                       rawHeap->str(),
                       *uploadKind,
                       *readbackKind};
}

} // namespace

Expected<std::string> buildUdonSharpBindingSource(const json::Object &manifest,
                                                  StringRef namespaceName,
                                                  StringRef className) {
  if (!isNamespace(namespaceName))
    return invalid("binding namespace is not a dotted C# identifier");
  if (!isUdonSharpIdentifier(className))
    return invalid("binding class name is not a C# identifier");
  auto physicalABI = manifest.getString("physical_export_abi");
  auto *exports = manifest.getArray("exports");
  if (!physicalABI || *physicalABI != "udonsharp-worlds-sdk-3.10.5" ||
      !exports || exports->empty())
    return invalid("binding source requires the supported physical export ABI");

  bool hasAsync = llvm::any_of(*exports, [](const json::Value &value) {
    const auto *entry = value.getAsObject();
    auto async = entry ? entry->getBoolean("async") : std::nullopt;
    return async && *async;
  });
  std::optional<AsyncCommands> asyncCommands;
  if (hasAsync) {
    auto parsed = parseAsyncCommands(manifest);
    if (!parsed)
      return parsed.takeError();
    asyncCommands = std::move(*parsed);
  }
  const json::Object *device = manifest.getObject("device");
  const json::Array *buffers = device ? device->getArray("buffers") : nullptr;
  bool hasBuffers = buffers && !buffers->empty();
  if (hasAsync && !hasBuffers)
    return invalid("async binding requires at least one Device buffer");
  SmallVector<PublicBufferBinding> publicBuffers;
  bool hasPublicInput = false;
  bool hasPublicOutput = false;
  if (buffers)
    for (auto indexed : llvm::enumerate(*buffers)) {
      const json::Value &value = indexed.value();
      const json::Object *buffer = value.getAsObject();
      auto id = buffer ? buffer->getInteger("id") : std::nullopt;
      auto bytes = buffer ? buffer->getInteger("logical_bytes") : std::nullopt;
      if (!id || !bytes || *id <= 0 || *bytes < 0 || (*bytes & 3) != 0)
        return invalid("binding manifest contains an invalid Device buffer");
      auto elementType = buffer->getString("element_type");
      auto access = buffer->getString("public_access");
      if (elementType.has_value() != access.has_value())
        return invalid(
            "Device buffer public type/access must be specified together");
      if (!elementType)
        continue;
      auto name = buffer->getString("name");
      int64_t elementBytes = *elementType == "int64" ? 8 : 4;
      if (!name ||
          (*elementType != "float32" && *elementType != "int32" &&
           *elementType != "int64") ||
          (*bytes % elementBytes) != 0 ||
          (*access != "input" && *access != "output"))
        return invalid("Device buffer has an unsupported public transfer ABI");
      publicBuffers.push_back({*id, *bytes / 4, *bytes / elementBytes,
                               publicIoName(*name, indexed.index()),
                               elementType->str(), access->str()});
      hasPublicInput |= *access == "input";
      hasPublicOutput |= *access == "output";
    }
  bool hasPublicBuffers = !publicBuffers.empty();

  std::string source;
  raw_string_ostream output(source);
  auto fingerprint = manifest.getString("public_abi_fingerprint");
  if (!fingerprint)
    return invalid("binding source requires a public ABI fingerprint");
  output << "// <auto-generated />\n"
         << "// udoncl-public-abi: " << *fingerprint << "\n"
         << "// udoncl-binding-namespace: " << namespaceName << "\n"
         << "// udoncl-binding-class: " << className
         << "\n"
            "using UdonSharp;\n"
            "using UnityEngine;\n"
            "using VRC.SDK3.Rendering;\n"
            "using VRC.SDKBase;\n"
            "using VRC.Udon;\n\n"
         << "namespace " << namespaceName << "\n{\n"
         << "    public sealed class " << className
         << " : UdonSharpBehaviour\n    {\n"
            "        public const int StatusSuccess = 0;\n"
            "        public const int StatusNotInitialized = -1;\n"
            "        public const int StatusBusy = -2;\n"
            "        public const int StatusInvalidArgument = -3;\n"
            "        public const int StatusAbiMismatch = -4;\n"
            "        public const int StatusExecutionFailure = -5;\n"
            "        public const int StatusDisposed = -6;\n\n"
            "        [SerializeField] private UdonBehaviour program;\n"
            "        [SerializeField] private Texture2D[] staticTensors;\n";
  if (hasBuffers)
    output
        << "        [SerializeField] private int[] deviceBufferIds;\n"
           "        [SerializeField] private int[] deviceBufferLogicalBytes;\n"
           "        [SerializeField] private RenderTexture[] deviceBuffers;\n"
           "        [SerializeField] private Material bufferCopyMaterial;\n"
           "        private Texture2D[] uploadTextures;\n";
  if (hasAsync)
    output << "        private UdonSharpBehaviour pendingAsyncReceiver;\n"
              "        private string pendingAsyncCompletionEvent;\n"
              "        private int pendingAsyncExportId;\n"
              "        private int pendingAsyncGeneration;\n"
              "        private int lastAsyncExportId;\n"
              "        private int lastAsyncStatus = StatusNotInitialized;\n"
              "        private int pendingAsyncObject;\n"
              "        private int pendingAsyncCompletionStatus;\n"
              "        private int asyncCommandReadbackMode;\n"
              "        private int asyncCommandBufferIndex;\n"
              "        private int asyncCommandOffsetWords;\n"
              "        private int asyncCommandSizeWords;\n"
              "        private int asyncCommandPointerBytes;\n"
              "        private int[] asyncCommandHeap;\n";
  if (hasPublicOutput)
    output
        << "        private float[] pendingFloatReadback;\n"
           "        private int[] pendingIntReadback;\n"
           "        private long[] pendingLongReadback;\n"
           "        private UdonSharpBehaviour pendingReadbackReceiver;\n"
           "        private string pendingReadbackEvent;\n"
           "        private int pendingReadbackIndex;\n"
           "        private int pendingReadbackGeneration;\n"
           "        private int lastReadbackStatus = StatusNotInitialized;\n";
  output << "        private int generation = 1;\n"
            "        private bool busy;\n"
         << (hasPublicBuffers ? "        private bool readbackBusy;\n" : "")
         << "        private bool disposed;\n\n"
            "        public int TryBindProgram(UdonBehaviour value)\n"
            "        {\n"
            "            if (value == null)\n"
            "                return StatusInvalidArgument;\n"
         << (hasPublicBuffers ? "            if (busy || readbackBusy)\n"
                              : "            if (busy)\n")
         << "                return StatusBusy;\n"
            "            generation = generation == int.MaxValue ? 1 : "
            "generation + 1;\n"
            "            program = value;\n"
            "            disposed = false;\n"
            "            return StatusSuccess;\n"
            "        }\n\n"
            "        public bool IsInitialized()\n"
            "        {\n"
            "            return program != null && !disposed;\n"
            "        }\n\n"
            "        public int TryDispose()\n"
            "        {\n"
            "            if (disposed)\n"
            "                return StatusSuccess;\n"
         << (hasPublicBuffers ? "            if (busy || readbackBusy)\n"
                              : "            if (busy)\n")
         << "                return StatusBusy;\n"
            "            generation = generation == int.MaxValue ? 1 : "
            "generation + 1;\n"
            "            disposed = true;\n"
            "            program = null;\n"
            "            return StatusSuccess;\n"
            "        }\n";

  std::set<std::string> methods{"IsInitialized", "TryBindProgram",
                                "TryDispose"};
  if (hasBuffers)
    output << R"(
        private int FindDeviceBuffer(int bufferId)
        {
            if (deviceBufferIds == null || deviceBuffers == null || deviceBufferLogicalBytes == null)
                return -1;
            for (int index = 0; index < deviceBufferIds.Length; ++index)
                if (deviceBufferIds[index] == bufferId)
                    return index;
            return -1;
        }

        private int EnsureDeviceBuffersCreated()
        {
            if (deviceBuffers == null)
                return StatusNotInitialized;
            for (int index = 0; index < deviceBuffers.Length; ++index)
            {
                RenderTexture buffer = deviceBuffers[index];
                if (buffer == null)
                    return StatusNotInitialized;
                if (!buffer.IsCreated() && !buffer.Create() && !buffer.IsCreated())
                    return StatusExecutionFailure;
            }
            return StatusSuccess;
        }

        private int UploadWords(int index, Color32[] pixels)
        {
            if (bufferCopyMaterial == null || deviceBuffers[index] == null)
                return StatusNotInitialized;
            if (uploadTextures == null)
                uploadTextures = new Texture2D[deviceBuffers.Length];
            Texture2D staging = uploadTextures[index];
            if (staging == null)
            {
                RenderTexture destination = deviceBuffers[index];
                staging = new Texture2D(destination.width, destination.height, TextureFormat.RGBA32, false, true);
                uploadTextures[index] = staging;
            }
            staging.SetPixels32(pixels);
            staging.Apply(false, false);
            VRCGraphics.Blit(staging, deviceBuffers[index], bufferCopyMaterial);
            return StatusSuccess;
        }
)";
  if (hasPublicBuffers)
    output << R"(
        private int ValidateTransfer(int bufferId, int words)
        {
            if (disposed)
                return StatusDisposed;
            if (busy || readbackBusy)
                return StatusBusy;
            if (program == null || bufferCopyMaterial == null)
                return StatusNotInitialized;
            int deviceBufferStatus = EnsureDeviceBuffersCreated();
            if (deviceBufferStatus != StatusSuccess)
                return deviceBufferStatus;
            int index = FindDeviceBuffer(bufferId);
            if (index < 0 || deviceBuffers[index] == null || words < 0 || words != deviceBufferLogicalBytes[index] / 4)
                return StatusInvalidArgument;
            return index;
        }

        private Color32 PackWord(int word)
        {
            return new Color32((byte)(word & 255), (byte)((word >> 8) & 255), (byte)((word >> 16) & 255), (byte)((word >> 24) & 255));
        }

        private int UnpackWord(Color32 pixel)
        {
            return pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
        }
)";
  if (hasPublicInput) {
    methods.insert("TryWriteFloatBuffer");
    methods.insert("TryWriteIntBuffer");
    methods.insert("TryWriteLongBuffer");
    output << R"(
        public int TryWriteFloatBuffer(int bufferId, float[] source)
        {
            if (source == null)
                return StatusInvalidArgument;
            int index = ValidateTransfer(bufferId, source.Length);
            if (index < 0)
                return index;
            Color32[] pixels = new Color32[deviceBuffers[index].width * deviceBuffers[index].height];
            for (int word = 0; word < source.Length; ++word)
                pixels[word] = PackWord(System.BitConverter.SingleToInt32Bits(source[word]));
            return UploadWords(index, pixels);
        }

        public int TryWriteIntBuffer(int bufferId, int[] source)
        {
            if (source == null)
                return StatusInvalidArgument;
            int index = ValidateTransfer(bufferId, source.Length);
            if (index < 0)
                return index;
            Color32[] pixels = new Color32[deviceBuffers[index].width * deviceBuffers[index].height];
            for (int word = 0; word < source.Length; ++word)
                pixels[word] = PackWord(source[word]);
            return UploadWords(index, pixels);
        }

        public int TryWriteLongBuffer(int bufferId, long[] source)
        {
            if (source == null)
                return StatusInvalidArgument;
            int index = ValidateTransfer(bufferId, source.Length * 2);
            if (index < 0)
                return index;
            Color32[] pixels = new Color32[deviceBuffers[index].width * deviceBuffers[index].height];
            for (int element = 0; element < source.Length; ++element)
            {
                long value = source[element];
                pixels[element * 2] = PackWord((int)value);
                pixels[element * 2 + 1] = PackWord((int)(value >> 32));
            }
            return UploadWords(index, pixels);
        }
)";
  }
  if (hasPublicOutput) {
    methods.insert("StartReadFloatBuffer");
    methods.insert("StartReadIntBuffer");
    methods.insert("StartReadLongBuffer");
    methods.insert("GetLastReadbackStatus");
    output << R"(
        private int StartReadback(int bufferId, int words, UdonSharpBehaviour receiver, string completionEvent)
        {
            if (receiver == null || string.IsNullOrEmpty(completionEvent))
                return StatusInvalidArgument;
            int index = ValidateTransfer(bufferId, words);
            if (index < 0)
                return index;
            readbackBusy = true;
            pendingReadbackIndex = index;
            pendingReadbackReceiver = receiver;
            pendingReadbackEvent = completionEvent;
            pendingReadbackGeneration = generation;
            lastReadbackStatus = StatusBusy;
            VRCAsyncGPUReadback.Request(deviceBuffers[index], 0, TextureFormat.RGBA32, this);
            return StatusSuccess;
        }

        private void CompletePublicReadback(int status)
        {
            lastReadbackStatus = status;
            readbackBusy = false;
            pendingFloatReadback = null;
            pendingIntReadback = null;
            pendingLongReadback = null;
            UdonSharpBehaviour receiver = pendingReadbackReceiver;
            string completionEvent = pendingReadbackEvent;
            pendingReadbackReceiver = null;
            pendingReadbackEvent = null;
            pendingReadbackGeneration = 0;
            if (receiver != null)
                receiver.SendCustomEvent(completionEvent);
        }

        public int StartReadFloatBuffer(int bufferId, float[] destination, UdonSharpBehaviour receiver, string completionEvent)
        {
            if (destination == null)
                return StatusInvalidArgument;
            if (readbackBusy)
                return StatusBusy;
            pendingFloatReadback = destination;
            pendingIntReadback = null;
            pendingLongReadback = null;
            int status = StartReadback(bufferId, destination.Length, receiver, completionEvent);
            if (status != StatusSuccess)
                pendingFloatReadback = null;
            return status;
        }

        public int StartReadIntBuffer(int bufferId, int[] destination, UdonSharpBehaviour receiver, string completionEvent)
        {
            if (destination == null)
                return StatusInvalidArgument;
            if (readbackBusy)
                return StatusBusy;
            pendingIntReadback = destination;
            pendingFloatReadback = null;
            pendingLongReadback = null;
            int status = StartReadback(bufferId, destination.Length, receiver, completionEvent);
            if (status != StatusSuccess)
                pendingIntReadback = null;
            return status;
        }

        public int StartReadLongBuffer(int bufferId, long[] destination, UdonSharpBehaviour receiver, string completionEvent)
        {
            if (destination == null)
                return StatusInvalidArgument;
            if (readbackBusy)
                return StatusBusy;
            pendingLongReadback = destination;
            pendingFloatReadback = null;
            pendingIntReadback = null;
            int status = StartReadback(bufferId, destination.Length * 2, receiver, completionEvent);
            if (status != StatusSuccess)
                pendingLongReadback = null;
            return status;
        }

        public int GetLastReadbackStatus()
        {
            return lastReadbackStatus;
        }
)";
  }
  for (const PublicBufferBinding &buffer : publicBuffers) {
    std::string method =
        (Twine(buffer.access == "input" ? "TryWrite" : "StartRead") +
         buffer.suffix)
            .str();
    if (!methods.insert(method).second)
      return invalid(Twine("generated U# method collision: ") + method);
    if (buffer.access == "input")
      output << "\n        public int " << method << '('
             << (buffer.elementType == "float32" ? "float[]"
                 : buffer.elementType == "int64" ? "long[]"
                                                 : "int[]")
             << " source)\n"
                "        {\n"
                "            return "
             << (buffer.elementType == "float32" ? "TryWriteFloatBuffer("
                 : buffer.elementType == "int64" ? "TryWriteLongBuffer("
                                                 : "TryWriteIntBuffer(")
             << buffer.id
             << ", source);\n"
                "        }\n";
    else
      output << "\n        public int " << method << '('
             << (buffer.elementType == "float32" ? "float[]"
                 : buffer.elementType == "int64" ? "long[]"
                                                 : "int[]")
             << " destination, UdonSharpBehaviour receiver, string "
                "completionEvent)\n"
                "        {\n"
                "            return "
             << (buffer.elementType == "float32" ? "StartReadFloatBuffer("
                 : buffer.elementType == "int64" ? "StartReadLongBuffer("
                                                 : "StartReadIntBuffer(")
             << buffer.id
             << ", destination, receiver, completionEvent);\n"
                "        }\n";
  }
  SmallVector<AsyncExportBinding> asyncExports;
  for (const json::Value &value : *exports) {
    const auto *entry = value.getAsObject();
    auto publicName = entry ? entry->getString("public_name") : std::nullopt;
    auto eventName = entry ? entry->getString("udon_event") : std::nullopt;
    auto statusSlot = entry ? entry->getString("status_slot") : std::nullopt;
    auto isAsync = entry ? entry->getBoolean("async") : std::nullopt;
    auto exportId = entry ? entry->getInteger("id") : std::nullopt;
    auto *parameters = entry ? entry->getArray("parameters") : nullptr;
    if (!publicName || !eventName || !statusSlot || !isAsync || !exportId ||
        *exportId <= 0 || !parameters || !isUdonSharpIdentifier(*publicName))
      return invalid("manifest contains an invalid export binding");
    std::string methodName =
        (Twine(*isAsync ? "Start" : "Try") + *publicName).str();
    if (!methods.insert(methodName).second)
      return invalid(Twine("generated U# method collision: ") + methodName);

    struct Parameter {
      std::string name;
      std::string type;
      std::string slot;
    };
    SmallVector<Parameter> parsedParameters;
    std::set<std::string> parameterNames;
    for (const json::Value &parameterValue : *parameters) {
      const auto *parameter = parameterValue.getAsObject();
      auto name = parameter ? parameter->getString("name") : std::nullopt;
      auto kind = parameter ? parameter->getString("kind") : std::nullopt;
      auto slot = parameter ? parameter->getString("heap_slot") : std::nullopt;
      if (!name || !kind || !slot || !isUdonSharpIdentifier(*name) ||
          !parameterNames.insert(name->str()).second)
        return invalid(Twine("invalid parameter in export: ") + *publicName);
      auto type = csharpType(*kind);
      if (!type)
        return type.takeError();
      parsedParameters.push_back({name->str(), type->str(), slot->str()});
    }

    const json::Object *result = entry->getObject("result");
    std::optional<std::string> resultType;
    std::optional<std::string> resultSlot;
    if (result) {
      auto kind = result->getString("kind");
      auto slot = result->getString("heap_slot");
      if (!kind || !slot)
        return invalid(Twine("invalid result in export: ") + *publicName);
      auto type = csharpType(*kind);
      if (!type)
        return type.takeError();
      resultType = type->str();
      resultSlot = slot->str();
      if (!parameterNames.insert("result").second)
        return invalid(Twine("result parameter collides in export: ") +
                       *publicName);
    }

    if (*isAsync) {
      std::string deviceBufferStatusLocal = "__udonclDeviceBufferStatus";
      while (parameterNames.count(deviceBufferStatusLocal))
        deviceBufferStatusLocal += '_';
      std::string statusMethod = (Twine("Get") + *publicName + "Status").str();
      if (!methods.insert(statusMethod).second)
        return invalid(Twine("generated U# method collision: ") + statusMethod);
      if (resultType) {
        std::string resultMethod =
            (Twine("TryGet") + *publicName + "Result").str();
        if (!methods.insert(resultMethod).second)
          return invalid(Twine("generated U# method collision: ") +
                         resultMethod);
        output << "\n        private " << *resultType << " last" << *publicName
               << "Result;\n";
      }
      output << "\n        public int " << methodName << '(';
      for (auto indexed : llvm::enumerate(parsedParameters)) {
        if (indexed.index() != 0)
          output << ", ";
        output << indexed.value().type << ' ' << indexed.value().name;
      }
      if (!parsedParameters.empty())
        output << ", ";
      output << "UdonSharpBehaviour receiver, string completionEvent)\n"
                "        {\n"
                "            if (disposed)\n"
                "                return StatusDisposed;\n"
                "            if (program == null)\n"
                "                return StatusNotInitialized;\n"
             << (hasPublicBuffers ? "            if (busy || readbackBusy)\n"
                                  : "            if (busy)\n")
             << "                return StatusBusy;\n"
                "            if (receiver == null || "
                "string.IsNullOrEmpty(completionEvent))\n"
                "                return StatusInvalidArgument;\n";
      if (hasBuffers)
        output << "            int " << deviceBufferStatusLocal
               << " = EnsureDeviceBuffersCreated();\n"
                  "            if ("
               << deviceBufferStatusLocal
               << " != StatusSuccess)\n"
                  "                return "
               << deviceBufferStatusLocal << ";\n";
      output << "            busy = true;\n"
                "            pendingAsyncReceiver = receiver;\n"
                "            pendingAsyncCompletionEvent = completionEvent;\n"
             << "            pendingAsyncExportId = " << *exportId
             << ";\n"
                "            pendingAsyncGeneration = generation;\n"
             << "            program.SetProgramVariable(\""
             << escapeString(asyncCommands->pendingObject) << "\", 0);\n"
             << "            program.SetProgramVariable(\""
             << escapeString(asyncCommands->completionObject) << "\", 0);\n"
             << "            program.SetProgramVariable(\""
             << escapeString(asyncCommands->commandKind) << "\", 0);\n";
      for (const Parameter &parameter : parsedParameters)
        output << "            program.SetProgramVariable(\""
               << escapeString(parameter.slot) << "\", " << parameter.name
               << ");\n";
      output << "            program.SendCustomEvent(\""
             << escapeString(*eventName)
             << "\");\n"
                "            PumpAsyncCommand();\n"
                "            return StatusSuccess;\n"
                "        }\n\n"
             << "        public int " << statusMethod
             << "()\n"
                "        {\n"
             << "            if (busy && pendingAsyncExportId == " << *exportId
             << ")\n"
                "                return StatusBusy;\n"
             << "            return lastAsyncExportId == " << *exportId
             << " ? lastAsyncStatus : StatusNotInitialized;\n"
                "        }\n";
      if (resultType)
        output << "\n        public int TryGet" << *publicName << "Result(out "
               << *resultType
               << " result)\n"
                  "        {\n"
               << "            result = last" << *publicName << "Result;\n"
               << "            if (busy && pendingAsyncExportId == "
               << *exportId
               << ")\n"
                  "                return StatusBusy;\n"
               << "            return lastAsyncExportId == " << *exportId
               << " ? lastAsyncStatus : StatusNotInitialized;\n"
                  "        }\n";
      asyncExports.push_back({*exportId, publicName->str(), statusSlot->str(),
                              resultType, resultSlot});
      continue;
    }

    output << "\n        public int " << methodName << '(';
    for (auto indexed : llvm::enumerate(parsedParameters)) {
      if (indexed.index() != 0)
        output << ", ";
      output << indexed.value().type << ' ' << indexed.value().name;
    }
    if (resultType) {
      if (!parsedParameters.empty())
        output << ", ";
      output << "out " << *resultType << " result";
    }
    output << ")\n        {\n";
    if (resultType)
      output << "            result = default;\n";
    output << "            if (disposed)\n"
              "                return StatusDisposed;\n"
              "            if (program == null)\n"
              "                return StatusNotInitialized;\n"
           << (hasPublicBuffers ? "            if (busy || readbackBusy)\n"
                                : "            if (busy)\n")
           << "                return StatusBusy;\n"
              "            busy = true;\n";
    for (const Parameter &parameter : parsedParameters)
      output << "            program.SetProgramVariable(\""
             << escapeString(parameter.slot) << "\", " << parameter.name
             << ");\n";
    output << "            program.SendCustomEvent(\""
           << escapeString(*eventName) << "\");\n"
           << "            object rawStatus = program.GetProgramVariable(\""
           << escapeString(*statusSlot)
           << "\");\n"
              "            if (rawStatus == null)\n"
              "            {\n"
              "                busy = false;\n"
              "                return StatusAbiMismatch;\n"
              "            }\n"
              "            int status = (int)rawStatus;\n";
    if (resultType)
      output
          << "            if (status == StatusSuccess)\n"
             "            {\n"
          << "                object rawResult = program.GetProgramVariable(\""
          << escapeString(*resultSlot)
          << "\");\n"
             "                if (rawResult == null)\n"
             "                {\n"
             "                    busy = false;\n"
             "                    return StatusAbiMismatch;\n"
             "                }\n"
          << "                result = (" << *resultType
          << ")rawResult;\n"
             "            }\n";
    output << "            busy = false;\n"
              "            return status;\n"
              "        }\n";
  }
  if (hasAsync) {
    const AsyncCommands &commands = *asyncCommands;
    output << R"(
        private void PumpAsyncCommand()
        {
            if (!busy || program == null || pendingAsyncGeneration != generation)
                return;
)";
    output
        << "            object rawKind = program.GetProgramVariable(\""
        << escapeString(commands.commandKind)
        << "\");\n"
           "            object rawPendingObject = program.GetProgramVariable(\""
        << escapeString(commands.pendingObject)
        << "\");\n"
           "            if (rawKind == null || rawPendingObject == null)\n"
           "            {\n"
           "                CompleteAsyncExport(StatusAbiMismatch);\n"
           "                return;\n"
           "            }\n"
           "            int kind = (int)rawKind;\n"
           "            int pendingObject = (int)rawPendingObject;\n"
           "            if (pendingObject == 0)\n"
           "            {\n"
           "                if (kind != 0)\n"
           "                    CompleteAsyncExport(StatusAbiMismatch);\n"
           "                else\n"
           "                    CompleteAsyncExportFromProgram();\n"
           "                return;\n"
           "            }\n"
           "            pendingAsyncObject = pendingObject;\n"
        << "            program.SetProgramVariable(\""
        << escapeString(commands.pendingObject)
        << "\", 0);\n"
           "            if (kind == 0)\n"
           "            {\n"
           "                pendingAsyncCompletionStatus = StatusSuccess;\n"
           "                "
           "SendCustomEventDelayedFrames(\"_ContinueAsyncCommand\", 1);\n"
           "                return;\n"
           "            }\n"
        << "            object rawBuffer = program.GetProgramVariable(\""
        << escapeString(commands.commandBuffer) << "\");\n"
        << "            object rawOffset = program.GetProgramVariable(\""
        << escapeString(commands.commandOffset) << "\");\n"
        << "            object rawSize = program.GetProgramVariable(\""
        << escapeString(commands.commandSize) << "\");\n"
        << "            object rawPointer = program.GetProgramVariable(\""
        << escapeString(commands.commandPointer) << "\");\n"
        << "            object rawHeap = program.GetProgramVariable(\""
        << escapeString(commands.rawHeap)
        << "\");\n"
           "            if (rawBuffer == null || rawOffset == null || rawSize "
           "== null || rawPointer == null || rawHeap == null)\n"
           "            {\n"
           "                FinishAsyncCommand(StatusAbiMismatch);\n"
           "                return;\n"
           "            }\n"
           "            int bufferId = (int)rawBuffer;\n"
           "            int offset = (int)rawOffset;\n"
           "            int size = (int)rawSize;\n"
           "            int pointer = (int)rawPointer;\n"
           "            int[] heap = (int[])rawHeap;\n"
           "            int index = FindDeviceBuffer(bufferId);\n"
           "            if (index < 0 || deviceBuffers[index] == null || "
           "offset < 0 || size <= 0 || pointer < 0 || (offset & 3) != 0 || "
           "(size & 3) != 0 || (pointer & 3) != 0 || offset > "
           "deviceBufferLogicalBytes[index] - size || pointer > heap.Length * "
           "4 - size)\n"
           "            {\n"
           "                FinishAsyncCommand(StatusInvalidArgument);\n"
           "                return;\n"
           "            }\n"
           "            asyncCommandBufferIndex = index;\n"
           "            asyncCommandOffsetWords = offset / 4;\n"
           "            asyncCommandSizeWords = size / 4;\n"
           "            asyncCommandPointerBytes = pointer;\n"
           "            asyncCommandHeap = heap;\n"
        << "            if (kind == " << commands.uploadKind
        << ")\n"
           "            {\n"
           "                if (offset == 0 && size == "
           "deviceBufferLogicalBytes[index])\n"
           "                {\n"
           "                    Color32[] pixels = new "
           "Color32[deviceBuffers[index].width * "
           "deviceBuffers[index].height];\n"
           "                    CopyHeapToPixels(pixels);\n"
           "                    FinishAsyncCommand(UploadWords(index, "
           "pixels));\n"
           "                    return;\n"
           "                }\n"
           "                asyncCommandReadbackMode = 2;\n"
           "                VRCAsyncGPUReadback.Request(deviceBuffers[index], "
           "0, TextureFormat.RGBA32, this);\n"
           "                return;\n"
           "            }\n"
        << "            if (kind == " << commands.readbackKind
        << ")\n"
           "            {\n"
           "                asyncCommandReadbackMode = 1;\n"
           "                VRCAsyncGPUReadback.Request(deviceBuffers[index], "
           "0, TextureFormat.RGBA32, this);\n"
           "                return;\n"
           "            }\n"
           "            FinishAsyncCommand(StatusAbiMismatch);\n"
           "        }\n";
    output << R"(

        private void CopyHeapToPixels(Color32[] pixels)
        {
            for (int word = 0; word < asyncCommandSizeWords; ++word)
            {
                int heapWord = asyncCommandPointerBytes / 4 + word;
                int pixelIndex = asyncCommandOffsetWords + word;
                int value = asyncCommandHeap[heapWord];
                pixels[pixelIndex] = new Color32((byte)(value & 255), (byte)((value >> 8) & 255), (byte)((value >> 16) & 255), (byte)((value >> 24) & 255));
            }
        }

        private void CompleteAsyncCommandReadback(VRCAsyncGPUReadbackRequest request)
        {
            int mode = asyncCommandReadbackMode;
            asyncCommandReadbackMode = 0;
            RenderTexture texture = deviceBuffers[asyncCommandBufferIndex];
            Color32[] pixels = new Color32[texture.width * texture.height];
            if (!request.TryGetData(pixels, 0))
            {
                FinishAsyncCommand(StatusExecutionFailure);
                return;
            }
            if (mode == 1)
            {
                for (int word = 0; word < asyncCommandSizeWords; ++word)
                {
                    int heapWord = asyncCommandPointerBytes / 4 + word;
                    Color32 pixel = pixels[asyncCommandOffsetWords + word];
                    asyncCommandHeap[heapWord] = pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
                }
                FinishAsyncCommand(StatusSuccess);
                return;
            }
            CopyHeapToPixels(pixels);
            FinishAsyncCommand(UploadWords(asyncCommandBufferIndex, pixels));
        }

        private void FinishAsyncCommand(int status)
        {
)";
    output << "            program.SetProgramVariable(\""
           << escapeString(commands.commandKind)
           << "\", 0);\n"
              "            asyncCommandHeap = null;\n"
              "            pendingAsyncCompletionStatus = status;\n"
              "            "
              "SendCustomEventDelayedFrames(\"_ContinueAsyncCommand\", 1);\n"
              "        }\n\n"
              "        public void _ContinueAsyncCommand()\n"
              "        {\n"
              "            if (!busy || program == null || "
              "pendingAsyncGeneration != generation)\n"
              "                return;\n"
           << "            program.SetProgramVariable(\""
           << escapeString(commands.completionObject)
           << "\", pendingAsyncObject);\n"
           << "            program.SetProgramVariable(\""
           << escapeString(commands.completionStatus)
           << "\", pendingAsyncCompletionStatus);\n"
           << "            program.SendCustomEvent(\""
           << escapeString(commands.completionEvent) << "\");\n"
           << "            program.SetProgramVariable(\""
           << escapeString(commands.completionObject)
           << "\", 0);\n"
              "            pendingAsyncObject = 0;\n"
              "            pendingAsyncCompletionStatus = 0;\n"
              "            PumpAsyncCommand();\n"
              "        }\n\n"
              "        private void CompleteAsyncExportFromProgram()\n"
              "        {\n";
    for (const AsyncExportBinding &binding : asyncExports) {
      output
          << "            if (pendingAsyncExportId == " << binding.id
          << ")\n"
             "            {\n"
          << "                object rawStatus = program.GetProgramVariable(\""
          << escapeString(binding.statusSlot)
          << "\");\n"
             "                if (rawStatus == null)\n"
             "                {\n"
             "                    CompleteAsyncExport(StatusAbiMismatch);\n"
             "                    return;\n"
             "                }\n"
             "                int status = (int)rawStatus;\n";
      if (binding.resultType)
        output << "                if (status == StatusSuccess)\n"
                  "                {\n"
               << "                    object rawResult = "
                  "program.GetProgramVariable(\""
               << escapeString(*binding.resultSlot)
               << "\");\n"
                  "                    if (rawResult == null)\n"
                  "                    {\n"
                  "                        "
                  "CompleteAsyncExport(StatusAbiMismatch);\n"
                  "                        return;\n"
                  "                    }\n"
               << "                    last" << binding.publicName
               << "Result = (" << *binding.resultType
               << ")rawResult;\n"
                  "                }\n";
      output << "                CompleteAsyncExport(status);\n"
                "                return;\n"
                "            }\n";
    }
    output << R"(            CompleteAsyncExport(StatusAbiMismatch);
        }

        private void CompleteAsyncExport(int status)
        {
            lastAsyncExportId = pendingAsyncExportId;
            lastAsyncStatus = status;
            busy = false;
            UdonSharpBehaviour receiver = pendingAsyncReceiver;
            string completionEvent = pendingAsyncCompletionEvent;
            pendingAsyncReceiver = null;
            pendingAsyncCompletionEvent = null;
            pendingAsyncExportId = 0;
            pendingAsyncGeneration = 0;
            if (receiver != null)
                receiver.SendCustomEvent(completionEvent);
        }
)";
  }
  if (hasAsync || hasPublicOutput) {
    output << R"(
        public override void OnAsyncGpuReadbackComplete(VRCAsyncGPUReadbackRequest request)
        {
)";
    if (hasAsync)
      output << R"(            if (asyncCommandReadbackMode != 0)
            {
                CompleteAsyncCommandReadback(request);
                return;
            }
)";
    if (hasPublicOutput)
      output << R"(            if (!readbackBusy)
                return;
            if (pendingReadbackGeneration != generation)
            {
                readbackBusy = false;
                pendingFloatReadback = null;
                pendingIntReadback = null;
                pendingLongReadback = null;
                pendingReadbackReceiver = null;
                pendingReadbackEvent = null;
                pendingReadbackGeneration = 0;
                return;
            }
            RenderTexture source = deviceBuffers[pendingReadbackIndex];
            Color32[] pixels = new Color32[source.width * source.height];
            bool success = request.TryGetData(pixels, 0);
            if (success)
            {
                int words = pendingFloatReadback != null ? pendingFloatReadback.Length : pendingIntReadback != null ? pendingIntReadback.Length : pendingLongReadback.Length * 2;
                for (int word = 0; word < words; ++word)
                {
                    int bits = UnpackWord(pixels[word]);
                    if (pendingFloatReadback != null)
                        pendingFloatReadback[word] = System.BitConverter.Int32BitsToSingle(bits);
                    else if (pendingIntReadback != null)
                        pendingIntReadback[word] = bits;
                }
                if (pendingLongReadback != null)
                    for (int element = 0; element < pendingLongReadback.Length; ++element)
                    {
                        long low = (long)UnpackWord(pixels[element * 2]) & 4294967295L;
                        long high = (long)UnpackWord(pixels[element * 2 + 1]) << 32;
                        pendingLongReadback[element] = low | high;
                    }
            }
            CompletePublicReadback(success ? StatusSuccess : StatusExecutionFailure);
)";
    else
      output << "            return;\n";
    output << "        }\n";
  }
  output << "    }\n}\n";
  output.flush();
  return source;
}

} // namespace vrc
