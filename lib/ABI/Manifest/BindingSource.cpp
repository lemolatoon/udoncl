#include "VRC/ABI/Manifest/BindingSource.h"
#include "VRC/ABI/Manifest/UdonSharpABI.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
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
  std::string pendingYield;
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
  /// Wall-clock time and number of continuations one frame may resume in
  /// the same Udon event. Zero resumes one continuation per frame.
  int64_t frameBudgetMicroseconds = 0;
  int64_t frameContinuationLimit = 0;
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
  auto pendingYield = commands->getString("pending_yield_slot");
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
  if (!pendingObject || !pendingYield || !completionObject ||
      !completionStatus || !completionEvent || !commandKind || !commandBuffer ||
      !commandOffset || !commandSize || !commandPointer || !rawHeap ||
      !uploadKind || !readbackKind || *uploadKind <= 0 || *readbackKind <= 0 ||
      *uploadKind == *readbackKind)
    return invalid("async binding command ABI is incomplete");
  // A manifest without a frame budget keeps one continuation per frame.
  int64_t frameBudget =
      commands->getInteger("frame_budget_microseconds").value_or(0);
  int64_t frameContinuations =
      commands->getInteger("frame_continuation_limit").value_or(0);
  if (frameBudget < 0 || frameBudget > 1000000 || frameContinuations < 0 ||
      frameContinuations > 1000000)
    return invalid("async binding frame budget is out of range");
  return AsyncCommands{pendingObject->str(),    pendingYield->str(),
                       completionObject->str(), completionStatus->str(),
                       completionEvent->str(),  commandKind->str(),
                       commandBuffer->str(),    commandOffset->str(),
                       commandSize->str(),      commandPointer->str(),
                       rawHeap->str(),          *uploadKind,
                       *readbackKind,           frameBudget,
                       frameContinuations};
}

/// Adapts the generated transfers to Device buffers of four words per RGBA32
/// float texel (bits, not values). Host words still travel as one word per
/// RGBA8 texel: an upload stages them in an RGBA8 texture four times as wide as
/// the buffer, which the transfer material's pass 0 packs into word texels;
/// a partial upload keeps the destination's other words from a snapshot in
/// `transferScratch` (pass 1, mode 0); a readback reads an RGBA8 copy made
/// by pass 2.
Expected<std::string> adaptWordTexelTransfers(std::string source) {
  struct Replacement {
    StringRef from, to;
    bool all;
  };
  const Replacement replacements[] = {
      {"        private Texture2D[] uploadTextures;\n",
       "        private Texture2D[] uploadTextures;\n"
       "        [SerializeField] private RenderTexture transferScratch;\n"
       "        private RenderTexture[] readbackStaging;\n"
       "\n"
       "        // An RGBA8 copy of a word-texel Device buffer, one word per "
       "texel.\n"
       "        private RenderTexture ReadbackSource(int index)\n"
       "        {\n"
       "            RenderTexture source = deviceBuffers[index];\n"
       "            if (readbackStaging == null)\n"
       "                readbackStaging = new "
       "RenderTexture[deviceBuffers.Length];\n"
       "            RenderTexture staging = readbackStaging[index];\n"
       "            if (staging == null)\n"
       "            {\n"
       "                staging = VRCRenderTexture.GetTemporary(source.width * "
       "4, source.height, 0, RenderTextureFormat.ARGB32, "
       "RenderTextureReadWrite.Linear, 1, RenderTextureMemoryless.None, "
       "VRTextureUsage.None, false);\n"
       "                readbackStaging[index] = staging;\n"
       "            }\n"
       "            VRCGraphics.Blit(source, staging, bufferCopyMaterial, 2);\n"
       "            return staging;\n"
       "        }\n",
       false},
      {"new Texture2D(destination.width, destination.height, "
       "TextureFormat.RGBA32, false, true)",
       "new Texture2D(destination.width * 4, destination.height, "
       "TextureFormat.RGBA32, false, true)",
       true},
      {"            VRCGraphics.Blit(staging, deviceBuffers[index], "
       "bufferCopyMaterial);\n",
       "            VRCGraphics.Blit(staging, deviceBuffers[index], "
       "bufferCopyMaterial, 0);\n",
       false},
      {"            bufferCopyMaterial.SetInteger(\"_VRC_TransferMode\", 1);\n",
       "            if (transferScratch == null)\n"
       "                return StatusNotInitialized;\n"
       "            VRCGraphics.Blit(destination, transferScratch, "
       "bufferCopyMaterial, 1);\n"
       "            bufferCopyMaterial.SetTexture(\"_VRC_TransferPrevious\", "
       "transferScratch);\n"
       "            bufferCopyMaterial.SetInteger(\"_VRC_TransferMode\", 1);\n",
       false},
      {"            VRCGraphics.Blit(staging, destination, "
       "bufferCopyMaterial);\n",
       "            VRCGraphics.Blit(staging, destination, bufferCopyMaterial, "
       "0);\n"
       "            bufferCopyMaterial.SetTexture(\"_VRC_TransferPrevious\", "
       "null);\n",
       false},
      {"Color32[deviceBuffers[index].width * deviceBuffers[index].height]",
       "Color32[deviceBuffers[index].width * 4 * deviceBuffers[index].height]",
       true},
      {"VRCAsyncGPUReadback.Request(deviceBuffers[index], 0, "
       "TextureFormat.RGBA32, this);",
       "VRCAsyncGPUReadback.Request(ReadbackSource(index), 0, "
       "TextureFormat.RGBA32, this);",
       true},
      {"Color32[texture.width * texture.height]",
       "Color32[texture.width * 4 * texture.height]", true},
      {"Color32[source.width * source.height]",
       "Color32[source.width * 4 * source.height]", true},
  };
  for (const Replacement &replacement : replacements) {
    size_t position = source.find(replacement.from.str());
    if (position == std::string::npos) {
      // Pieces of optional features (public buffers, async commands).
      if (replacement.all)
        continue;
      if (replacement.from.contains("uploadTextures"))
        return invalid("word-texel binding requires Device buffers");
      continue;
    }
    while (position != std::string::npos) {
      source.replace(position, replacement.from.size(), replacement.to.str());
      if (!replacement.all)
        break;
      position =
          source.find(replacement.from.str(), position + replacement.to.size());
    }
  }
  return source;
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
  output
      << "// <auto-generated />\n"
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
         "        public const int AsyncDiagnosticNone = 0;\n"
         "        public const int AsyncDiagnosticMissingCommandState = 1;\n"
         "        public const int AsyncDiagnosticInconsistentIdleState = 2;\n"
         "        public const int AsyncDiagnosticMissingCommandPayload = 3;\n"
         "        public const int AsyncDiagnosticUnknownCommandKind = 4;\n"
         "        public const int AsyncDiagnosticMissingExportStatus = 5;\n"
         "        public const int AsyncDiagnosticMissingExportResult = 6;\n"
         "        public const int AsyncDiagnosticUnknownExport = 7;\n\n"
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
              "        private int lastAsyncDiagnostic;\n"
              "        private int pendingAsyncObject;\n"
              "        private int pendingAsyncCompletionStatus;\n"
              "        private int asyncCommandReadbackMode;\n"
              "        private bool asyncCommandAwaited;\n"
              "        private int asyncCommandUnreportedStatus;\n"
              "        private int asyncCommandBufferIndex;\n"
              "        private int asyncCommandOffsetWords;\n"
              "        private int asyncCommandSizeWords;\n"
              "        private int asyncCommandPointerBytes;\n"
              "        private int[] asyncCommandHeap;\n";
  if (hasAsync)
    output << "        public const int AsyncFrameBudgetMicroseconds = "
           << asyncCommands->frameBudgetMicroseconds
           << ";\n"
              "        public const int AsyncFrameContinuationLimit = "
           << asyncCommands->frameContinuationLimit
           << ";\n"
              "        public const int AsyncFrameSettingMaximum = 1000000;\n"
              "        private bool asyncDriving;\n"
              "        private bool asyncPumpRequested;\n"
              "        private bool asyncContinuationReady;\n"
              // Run-time frame budget, initialized from the manifest
              // defaults above and changed through TrySetAsyncFrameBudget.
              "        private int asyncFrameBudgetMicroseconds = "
              "AsyncFrameBudgetMicroseconds;\n"
              "        private int asyncFrameContinuationLimit = "
              "AsyncFrameContinuationLimit;\n"
              "        private bool asyncFrameAdaptive;\n"
              "        private double asyncSliceEstimate;\n"
              "        private int asyncFrame = -1;\n"
              "        private int asyncFrameContinuations;\n"
              "        private double asyncFrameStart;\n"
              "        private double asyncFrameDeadline;\n"
              // Opt-in pump trace. Inert while the arrays are null: a
              // frame-pacing fixture assigns them through
              // UdonBehaviour.SetProgramVariable and reads them back.
              "        private int[] asyncTraceContinuations;\n"
              "        private int asyncTraceContinuationWords;\n"
              "        private int[] asyncTraceWindows;\n"
              "        private int asyncTraceWindowWords;\n";
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
  if (hasAsync)
    methods.insert({"GetLastAsyncDiagnostic", "TrySetAsyncFrameBudget",
                    "GetAsyncFrameBudgetMicroseconds",
                    "GetAsyncFrameContinuationLimit", "SetAsyncFrameAdaptive",
                    "IsAsyncFrameAdaptive"});
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
  if (hasAsync)
    output << R"(
        // Writes only words [firstWord, firstWord + wordCount) of the
        // staging surface: the transfer material's range-copy mode discards
        // every other fragment, so the rest of the Device buffer keeps its
        // contents without a readback.
        private int UploadWordRange(int index, Color32[] pixels, int firstWord, int wordCount)
        {
            if (bufferCopyMaterial == null || deviceBuffers[index] == null)
                return StatusNotInitialized;
            if (uploadTextures == null)
                uploadTextures = new Texture2D[deviceBuffers.Length];
            RenderTexture destination = deviceBuffers[index];
            Texture2D staging = uploadTextures[index];
            if (staging == null)
            {
                staging = new Texture2D(destination.width, destination.height, TextureFormat.RGBA32, false, true);
                uploadTextures[index] = staging;
            }
            staging.SetPixels32(pixels);
            staging.Apply(false, false);
            bufferCopyMaterial.SetInteger("_VRC_TransferMode", 1);
            bufferCopyMaterial.SetInteger("_VRC_TransferWidth", destination.width);
            bufferCopyMaterial.SetInteger("_VRC_TransferFirst", firstWord);
            bufferCopyMaterial.SetInteger("_VRC_TransferCount", wordCount);
            bufferCopyMaterial.SetInteger("_VRC_TransferDelta", 0);
            VRCGraphics.Blit(staging, destination, bufferCopyMaterial);
            bufferCopyMaterial.SetInteger("_VRC_TransferMode", 0);
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
    // A synchronous export declared safe to call while an async export is
    // suspended: it neither suspends nor touches the state the suspended
    // export owns, so it is not rejected with StatusBusy.
    bool allowedWhileBusy =
        entry->getBoolean("allowed_while_busy").value_or(false);
    if (allowedWhileBusy && *isAsync)
      return invalid(Twine("an async export cannot be allowed while busy: ") +
                     *publicName);
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
             << escapeString(asyncCommands->pendingYield) << "\", 0);\n"
             << "            program.SetProgramVariable(\""
             << escapeString(asyncCommands->commandKind) << "\", 0);\n"
             << "            asyncCommandAwaited = false;\n"
                "            asyncCommandUnreportedStatus = StatusSuccess;\n";
      for (const Parameter &parameter : parsedParameters)
        output << "            program.SetProgramVariable(\""
               << escapeString(parameter.slot) << "\", " << parameter.name
               << ");\n";
      // The export entry is the first Host slice of the export: it is
      // charged to the frame's window like a continuation.
      output << "            OpenAsyncFrameWindow();\n"
                "            asyncFrameContinuations = "
                "asyncFrameContinuations + 1;\n"
                "            program.SendCustomEvent(\""
             << escapeString(*eventName)
             << "\");\n"
                "            asyncPumpRequested = true;\n"
                "            DriveAsyncCommands();\n"
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
              "                return StatusNotInitialized;\n";
    StringRef restoreBusy =
        allowedWhileBusy ? "busy = wasBusy;\n" : "busy = false;\n";
    if (!allowedWhileBusy)
      output << (hasPublicBuffers ? "            if (busy || readbackBusy)\n"
                                  : "            if (busy)\n")
             << "                return StatusBusy;\n"
                "            busy = true;\n";
    else
      output << (hasPublicBuffers ? "            if (readbackBusy)\n"
                                    "                return StatusBusy;\n"
                                  : "")
             << "            bool wasBusy = busy;\n"
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
              "                "
           << restoreBusy
           << "                return StatusAbiMismatch;\n"
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
             "                    "
          << restoreBusy
          << "                    return StatusAbiMismatch;\n"
             "                }\n"
          << "                result = (" << *resultType
          << ")rawResult;\n"
             "            }\n";
    output << "            " << restoreBusy
           << "            return status;\n"
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
           "                lastAsyncDiagnostic = "
           "AsyncDiagnosticMissingCommandState;\n"
           "                CompleteAsyncExport(StatusAbiMismatch);\n"
           "                return;\n"
           "            }\n"
           "            int kind = (int)rawKind;\n"
           "            int pendingObject = (int)rawPendingObject;\n"
           "            if (pendingObject == 0 && kind == 0)\n"
           "            {\n"
           "                CompleteAsyncExportFromProgram();\n"
           "                return;\n"
           "            }\n"
           "            pendingAsyncObject = pendingObject;\n"
           "            // A cooperative checkpoint does not depend on the\n"
           "            // published transfer: the Host resumes while a\n"
           "            // readback is in flight. Every other suspension\n"
           "            // waits for the transfer to finish.\n"
           "            bool yields = false;\n"
           "            if (pendingObject != 0)\n"
           "            {\n"
        << "                object rawYield = program.GetProgramVariable(\""
        << escapeString(commands.pendingYield)
        << "\");\n"
           "                yields = rawYield != null && (int)rawYield != 0;\n"
        << "                program.SetProgramVariable(\""
        << escapeString(commands.pendingObject) << "\", 0);\n"
        << "                program.SetProgramVariable(\""
        << escapeString(commands.pendingYield)
        << "\", 0);\n"
           "            }\n"
           "            if (kind == 0)\n"
           "            {\n"
           "                // A transfer that failed while nothing waited\n"
           "                // for it fails the next suspension that waits.\n"
           "                pendingAsyncCompletionStatus = StatusSuccess;\n"
           "                if (!yields)\n"
           "                {\n"
           "                    pendingAsyncCompletionStatus = "
           "asyncCommandUnreportedStatus;\n"
           "                    asyncCommandUnreportedStatus = "
           "StatusSuccess;\n"
           "                }\n"
           "                asyncContinuationReady = true;\n"
           "                return;\n"
           "            }\n"
           "            if (asyncCommandReadbackMode != 0)\n"
           "            {\n"
           "                // The published readback is already in flight.\n"
           "                asyncCommandAwaited = !yields;\n"
           "                if (yields)\n"
           "                {\n"
           "                    pendingAsyncCompletionStatus = "
           "StatusSuccess;\n"
           "                    asyncContinuationReady = true;\n"
           "                }\n"
           "                return;\n"
           "            }\n"
           "            asyncCommandAwaited = true;\n"
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
           "                lastAsyncDiagnostic = "
           "AsyncDiagnosticMissingCommandPayload;\n"
           "                FinishAsyncCommand(StatusAbiMismatch);\n"
           "                return;\n"
           "            }\n"
           "            int physicalBufferToken = (int)rawBuffer;\n"
           "            int offset = (int)rawOffset;\n"
           "            int size = (int)rawSize;\n"
           "            int pointer = (int)rawPointer;\n"
           "            int[] heap = (int[])rawHeap;\n"
           "            int index = physicalBufferToken - 1;\n"
           "            if (index < 0 || index >= deviceBuffers.Length || "
           "deviceBuffers[index] == null || "
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
           "                Color32[] pixels = new "
           "Color32[deviceBuffers[index].width * "
           "deviceBuffers[index].height];\n"
           "                CopyHeapToPixels(pixels);\n"
           "                if (offset == 0 && size == "
           "deviceBufferLogicalBytes[index])\n"
           "                    FinishAsyncCommand(UploadWords(index, "
           "pixels));\n"
           "                else\n"
           "                    FinishAsyncCommand(UploadWordRange(index, "
           "pixels, asyncCommandOffsetWords, asyncCommandSizeWords));\n"
           "                return;\n"
           "            }\n"
        << "            if (kind == " << commands.readbackKind
        << ")\n"
           "            {\n"
           "                asyncCommandReadbackMode = 1;\n"
           "                VRCAsyncGPUReadback.Request(deviceBuffers[index], "
           "0, TextureFormat.RGBA32, this);\n"
           "                asyncCommandAwaited = !yields;\n"
           "                if (yields)\n"
           "                {\n"
           "                    pendingAsyncCompletionStatus = "
           "StatusSuccess;\n"
           "                    asyncContinuationReady = true;\n"
           "                }\n"
           "                return;\n"
           "            }\n"
           "            lastAsyncDiagnostic = "
           "AsyncDiagnosticUnknownCommandKind;\n"
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
            if (mode != 1)
            {
                FinishAsyncCommand(StatusAbiMismatch);
                return;
            }
            for (int word = 0; word < asyncCommandSizeWords; ++word)
            {
                int heapWord = asyncCommandPointerBytes / 4 + word;
                Color32 pixel = pixels[asyncCommandOffsetWords + word];
                asyncCommandHeap[heapWord] = pixel.r | (pixel.g << 8) | (pixel.b << 16) | (pixel.a << 24);
            }
            FinishAsyncCommand(StatusSuccess);
        }

        private void FinishAsyncCommand(int status)
        {
)";
    output << "            program.SetProgramVariable(\""
           << escapeString(commands.commandKind)
           << "\", 0);\n"
              "            asyncCommandHeap = null;\n"
              "            if (!asyncCommandAwaited)\n"
              "            {\n"
              "                // The Host already resumed from a checkpoint;\n"
              "                // it observes the idle command slot later.\n"
              "                if (status != StatusSuccess)\n"
              "                    asyncCommandUnreportedStatus = status;\n"
              "                return;\n"
              "            }\n"
              "            asyncCommandAwaited = false;\n"
              "            pendingAsyncCompletionStatus = status;\n"
              "            asyncContinuationReady = true;\n"
              "        }\n\n"
              "        private void DeliverAsyncCompletion()\n"
              "        {\n"
              "            if (pendingAsyncObject != 0)\n"
              "            {\n"
           << "                program.SetProgramVariable(\""
           << escapeString(commands.completionObject)
           << "\", pendingAsyncObject);\n"
           << "                program.SetProgramVariable(\""
           << escapeString(commands.completionStatus)
           << "\", pendingAsyncCompletionStatus);\n"
              "                double traceStart = asyncTraceContinuations "
              "!= null ? Time.realtimeSinceStartupAsDouble : 0.0;\n"
           << "                program.SendCustomEvent(\""
           << escapeString(commands.completionEvent) << "\");\n"
           << "                if (asyncTraceContinuations != null)\n"
              "                    TraceAsyncContinuation(traceStart);\n"
           << "                program.SetProgramVariable(\""
           << escapeString(commands.completionObject)
           << "\", 0);\n"
              "            }\n"
              "            pendingAsyncObject = 0;\n"
              "            pendingAsyncCompletionStatus = 0;\n"
              "        }\n\n";
    // Trace records: per delivered continuation (frame, 0 when it was the
    // first Host slice charged to the frame's window or 1 when it followed
    // another one, microseconds of the Udon event), and per exit of the
    // driving loop (frame, Host slices charged to the window, microseconds
    // since the window opened, reason: 1 continuation limit, 2 time budget,
    // 3 nothing ready, 4 adaptive estimate of the next slice exceeds the
    // budget).
    output << R"(        private void TraceAsyncContinuation(double start)
        {
            int word = asyncTraceContinuationWords;
            if (word + 3 > asyncTraceContinuations.Length)
                return;
            asyncTraceContinuations[word] = Time.frameCount;
            asyncTraceContinuations[word + 1] = asyncFrameContinuations > 1 ? 1 : 0;
            asyncTraceContinuations[word + 2] = (int)((Time.realtimeSinceStartupAsDouble - start) * 1000000.0);
            asyncTraceContinuationWords = word + 3;
        }

        private void TraceAsyncWindow(int reason)
        {
            int word = asyncTraceWindowWords;
            if (word + 4 > asyncTraceWindows.Length)
                return;
            asyncTraceWindows[word] = asyncFrame;
            asyncTraceWindows[word + 1] = asyncFrameContinuations;
            asyncTraceWindows[word + 2] = (int)((Time.realtimeSinceStartupAsDouble - asyncFrameStart) * 1000000.0);
            asyncTraceWindows[word + 3] = reason;
            asyncTraceWindowWords = word + 4;
        }

        // The first entry in a frame opens the frame's window. Every Host
        // slice that runs in the frame (an export entry or a continuation)
        // is charged to it, the first one included.
        private void OpenAsyncFrameWindow()
        {
            if (Time.frameCount == asyncFrame)
                return;
            asyncFrame = Time.frameCount;
            asyncFrameContinuations = 0;
            asyncFrameStart = Time.realtimeSinceStartupAsDouble;
            asyncFrameDeadline = asyncFrameStart + asyncFrameBudgetMicroseconds * 0.000001;
        }

        // The only loop that pumps commands. A continuation that is ready
        // in the same Udon event (a Device work slice or a synchronously
        // completed command) resumes here while the frame's budget lasts;
        // otherwise it waits for the next frame. The first Host slice of a
        // frame always runs, so every frame makes progress. An export
        // started from a completion event only requests a pump from the
        // running loop, so the pump never nests.
        private void DriveAsyncCommands()
        {
            if (asyncDriving)
                return;
            asyncDriving = true;
            OpenAsyncFrameWindow();
            while (true)
            {
                double sliceStart = 0.0;
                if (asyncFrameAdaptive)
                    sliceStart = Time.realtimeSinceStartupAsDouble;
                if (asyncContinuationReady)
                {
                    asyncContinuationReady = false;
                    if (asyncFrameContinuations > 0)
                    {
                        int stop = 0;
                        double now = Time.realtimeSinceStartupAsDouble;
                        if (asyncFrameContinuations >= asyncFrameContinuationLimit)
                            stop = 1;
                        else if (now >= asyncFrameDeadline)
                            stop = 2;
                        else if (asyncFrameAdaptive && now + asyncSliceEstimate > asyncFrameDeadline)
                            stop = 4;
                        if (stop != 0)
                        {
                            SendCustomEventDelayedFrames("_ContinueAsyncCommand", 1);
                            if (asyncTraceWindows != null)
                                TraceAsyncWindow(stop);
                            break;
                        }
                    }
                    asyncFrameContinuations = asyncFrameContinuations + 1;
                    DeliverAsyncCompletion();
                    asyncPumpRequested = true;
                }
                if (!asyncPumpRequested)
                {
                    if (asyncTraceWindows != null)
                        TraceAsyncWindow(3);
                    break;
                }
                asyncPumpRequested = false;
                PumpAsyncCommand();
                if (asyncFrameAdaptive)
                {
                    // Running average of one Host slice and its command.
                    double slice = Time.realtimeSinceStartupAsDouble - sliceStart;
                    if (asyncSliceEstimate <= 0.0)
                        asyncSliceEstimate = slice;
                    else
                        asyncSliceEstimate = asyncSliceEstimate + (slice - asyncSliceEstimate) * 0.25;
                }
            }
            asyncDriving = false;
        }

        // Resumes a continuation the previous frame deferred. It is charged
        // to this frame's window like any other continuation.
        public void _ContinueAsyncCommand()
        {
            if (!busy || program == null || pendingAsyncGeneration != generation)
                return;
            asyncContinuationReady = true;
            DriveAsyncCommands();
        }

        // Changes the frame budget at run time. Both limits take the
        // manifest's range (0 to AsyncFrameSettingMaximum); zero resumes one
        // Host slice per frame. The time budget applies from the next
        // frame's window, the continuation limit at once.
        public int TrySetAsyncFrameBudget(int microseconds, int continuations)
        {
            if (microseconds < 0 || microseconds > AsyncFrameSettingMaximum || continuations < 0 || continuations > AsyncFrameSettingMaximum)
                return StatusInvalidArgument;
            asyncFrameBudgetMicroseconds = microseconds;
            asyncFrameContinuationLimit = continuations;
            return StatusSuccess;
        }

        public int GetAsyncFrameBudgetMicroseconds()
        {
            return asyncFrameBudgetMicroseconds;
        }

        public int GetAsyncFrameContinuationLimit()
        {
            return asyncFrameContinuationLimit;
        }

        // Adaptive mode also stops before a slice that the running average
        // of the measured slices predicts would end past the budget.
        public void SetAsyncFrameAdaptive(bool enabled)
        {
            asyncFrameAdaptive = enabled;
            asyncSliceEstimate = 0.0;
        }

        public bool IsAsyncFrameAdaptive()
        {
            return asyncFrameAdaptive;
        }

)";
    output << "        private void CompleteAsyncExportFromProgram()\n"
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
             "                    lastAsyncDiagnostic = "
             "AsyncDiagnosticMissingExportStatus;\n"
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
                  "                        lastAsyncDiagnostic = "
                  "AsyncDiagnosticMissingExportResult;\n"
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
    output << R"(            lastAsyncDiagnostic = AsyncDiagnosticUnknownExport;
            CompleteAsyncExport(StatusAbiMismatch);
        }

        public int GetLastAsyncDiagnostic()
        {
            return lastAsyncDiagnostic;
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
                DriveAsyncCommands();
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
  const json::Object *program = manifest.getObject("program");
  auto texelFormat =
      program ? program->getString("texel_format") : std::nullopt;
  if (!texelFormat || *texelFormat == "rgba8_unorm")
    return source;
  if (*texelFormat != "rgba32_float")
    return invalid("binding source requires a supported Device texel format");
  // Without a Device buffer the binding moves no texels.
  if (!hasBuffers)
    return source;
  return adaptWordTexelTransfers(std::move(source));
}

} // namespace vrc
