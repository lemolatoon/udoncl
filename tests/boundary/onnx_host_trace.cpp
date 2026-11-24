// Executes the Device work an ONNX Host run records
// (tests/boundary/onnx_host_interpreter.py): synthesized kernels draw on the
// native fragment renderer from the HLSL udonclc emits for them, and library
// kernels run their OpenCL C sources compiled for the CPU, one call per
// output word, exactly as the Device kernels index their buffers. The trace
// is a text file of one command per line:
//
//   fused <name> <hlsl> <words> <width> <height> <inputs> <scalars>
//   vertex <hlsl>
//   buffer <handle> <words>
//   words <handle> <count> <word>...         (hexadecimal words)
//   fill <handle> <word>
//   dispatch <kernel> <count> <argument>...  (hexadecimal 32-bit words)
//   copy <source> <destination> <source-byte> <destination-byte> <bytes>
//   save <handle> <words> <path>
//
// Buffer handles and scalars of a dispatch are in kernel argument order;
// a synthesized kernel takes its inputs, then its result, then its scalars.
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

size_t udononnx_reindex_cpu_gid;
size_t udonblas_arena_sgemm_cpu_gid_x;
size_t udonblas_arena_sgemm_cpu_gid_y;
size_t udondnn_large_pointwise_cpu_gid_x;
size_t udondnn_large_pointwise_cpu_gid_y;
size_t udononnx_unary_cpu_gid;

#include "../../libraries/udonblas/device/arena_sgemm.cl"
#include "../../libraries/udononnxoperators/device/reindex.cl"
#include "../../libraries/udononnxoperators/device/unary.cl"
// Last: it maps exp to expf for the CPU build.
#include "../../libraries/udondnn/device/large_pointwise.cl"

namespace {
constexpr uint32_t surfaceWidth = 8192;

struct Fused {
  std::string name, hlsl;
  uint32_t id, words, width, height, inputs, scalars;
};

using Words = std::vector<uint32_t>;

float asFloat(uint32_t word) {
  float value;
  std::memcpy(&value, &word, sizeof value);
  return value;
}

int32_t asInt(uint32_t word) { return static_cast<int32_t>(word); }

class Executor {
public:
  std::map<uint32_t, Words> buffers;
  uint32_t minimumWords = 16384;

  Words &buffer(uint32_t handle) {
    auto found = buffers.find(handle);
    if (found == buffers.end())
      throw std::runtime_error("unknown buffer " + std::to_string(handle));
    return found->second;
  }
  float *floats(uint32_t handle) {
    return reinterpret_cast<float *>(buffer(handle).data());
  }
  const unsigned *unsignedWords(uint32_t handle) {
    return reinterpret_cast<const unsigned *>(buffer(handle).data());
  }

  /// Runs `body` once per word of the output buffer, as the Device kernel
  /// runs once per fragment of its surface.
  template <typename Body>
  void forEachWord(uint32_t output, size_t &gid, Body body) {
    size_t words = buffer(output).size();
    for (gid = 0; gid < words; ++gid)
      body();
  }
  template <typename Body>
  void forEachWord2D(uint32_t output, size_t &x, size_t &y, Body body) {
    size_t words = buffer(output).size();
    for (size_t gid = 0; gid < words; ++gid) {
      x = gid % surfaceWidth;
      y = gid / surfaceWidth;
      body();
    }
  }

  void library(const std::string &name, const std::vector<uint32_t> &a) {
    auto expect = [&](size_t count) {
      if (a.size() != count)
        throw std::runtime_error(name + ": " + std::to_string(a.size()) +
                                 " arguments");
    };
    if (name == "udonblas_arena_sgemm_strided_batched" ||
        name == "udonblas_arena_sgemm_strided_batched_large") {
      expect(13);
      auto kernel = name == "udonblas_arena_sgemm_strided_batched"
                        ? udonblas_arena_sgemm_strided_batched
                        : udonblas_arena_sgemm_strided_batched_large;
      forEachWord2D(a[2], udonblas_arena_sgemm_cpu_gid_x,
                    udonblas_arena_sgemm_cpu_gid_y, [&] {
                      kernel(floats(a[0]), floats(a[1]), floats(a[2]),
                             asInt(a[3]), asInt(a[4]), asInt(a[5]), asInt(a[6]),
                             asInt(a[7]), asInt(a[8]), asInt(a[9]),
                             asInt(a[10]), asInt(a[11]), asFloat(a[12]));
                    });
    } else if (name == "udononnx_cast_i64_f32") {
      expect(3);
      forEachWord(a[1], udononnx_reindex_cpu_gid, [&] {
        udononnx_cast_i64_f32(unsignedWords(a[0]), floats(a[1]), asInt(a[2]));
      });
    } else if (name == "udononnx_equal_f32") {
      expect(11);
      forEachWord(a[2], udononnx_reindex_cpu_gid, [&] {
        udononnx_equal_f32(floats(a[0]), floats(a[1]), floats(a[2]),
                           asInt(a[3]), asInt(a[4]), asInt(a[5]), asInt(a[6]),
                           asInt(a[7]), asInt(a[8]), asInt(a[9]), asInt(a[10]));
      });
    } else if (name == "udononnx_where_f32") {
      expect(14);
      forEachWord(a[3], udononnx_reindex_cpu_gid, [&] {
        udononnx_where_f32(floats(a[0]), floats(a[1]), floats(a[2]),
                           floats(a[3]), asInt(a[4]), asInt(a[5]), asInt(a[6]),
                           asInt(a[7]), asInt(a[8]), asInt(a[9]), asInt(a[10]),
                           asInt(a[11]), asInt(a[12]), asInt(a[13]));
      });
    } else if (name == "udononnx_concat2_f32") {
      expect(7);
      forEachWord(a[2], udononnx_reindex_cpu_gid, [&] {
        udononnx_concat2_f32(floats(a[0]), floats(a[1]), floats(a[2]),
                             asInt(a[3]), asInt(a[4]), asInt(a[5]),
                             asInt(a[6]));
      });
    } else if (name == "udondnn_large_op_tensor") {
      expect(15);
      forEachWord2D(a[3], udondnn_large_pointwise_cpu_gid_x,
                    udondnn_large_pointwise_cpu_gid_y, [&] {
                      udondnn_large_op_tensor(
                          floats(a[0]), floats(a[1]), floats(a[2]),
                          floats(a[3]), asInt(a[4]), asInt(a[5]), asInt(a[6]),
                          asInt(a[7]), asInt(a[8]), asInt(a[9]), asInt(a[10]),
                          asInt(a[11]), asFloat(a[12]), asFloat(a[13]),
                          asFloat(a[14]));
                    });
    } else if (name == "udononnx_unary_f32") {
      expect(4);
      forEachWord(a[1], udononnx_unary_cpu_gid, [&] {
        udononnx_unary_f32(floats(a[0]), floats(a[1]), asInt(a[2]),
                           asInt(a[3]));
      });
    } else {
      throw std::runtime_error("no CPU build of library kernel " + name);
    }
  }
};

std::vector<std::string> split(const std::string &line) {
  std::istringstream stream(line);
  std::vector<std::string> words;
  for (std::string word; stream >> word;)
    words.push_back(word);
  return words;
}

uint32_t hex(const std::string &text) {
  return static_cast<uint32_t>(std::stoul(text, nullptr, 16));
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: onnx-host-trace <trace>\n";
    return 2;
  }
  try {
    std::ifstream input(argv[1]);
    if (!input)
      throw std::runtime_error(std::string("cannot read ") + argv[1]);
    std::vector<std::vector<std::string>> commands;
    std::map<std::string, Fused> fused;
    std::string vertex;
    for (std::string line; std::getline(input, line);) {
      auto words = split(line);
      if (words.empty())
        continue;
      if (words[0] == "fused") {
        Fused kernel{words[1],
                     words[2],
                     static_cast<uint32_t>(1000 + fused.size()),
                     static_cast<uint32_t>(std::stoul(words[3])),
                     static_cast<uint32_t>(std::stoul(words[4])),
                     static_cast<uint32_t>(std::stoul(words[5])),
                     static_cast<uint32_t>(std::stoul(words[6])),
                     static_cast<uint32_t>(std::stoul(words[7]))};
        fused[kernel.name] = kernel;
      } else if (words[0] == "vertex") {
        vertex = words[1];
      } else {
        commands.push_back(words);
      }
    }
    std::vector<native_opencl::GatherBinding> bindings;
    for (const auto &[name, kernel] : fused)
      bindings.push_back({kernel.id, kernel.words, kernel.scalars, name,
                          kernel.hlsl, kernel.width, kernel.height,
                          kernel.inputs});
    // Constructing the session compiles every fragment; a rejected shader
    // throws with the compiler diagnostics.
    native_opencl::Session session("", bindings, vertex);

    Executor executor;
    for (const auto &[name, kernel] : fused)
      executor.minimumWords = std::max(executor.minimumWords, kernel.words);
    unsigned dispatches = 0, draws = 0;
    for (const auto &command : commands) {
      const std::string &verb = command[0];
      if (verb == "buffer") {
        size_t words = std::stoul(command[2]);
        // The renderer binds every input as a whole draw surface, and the
        // library kernels mask their addresses to a power of two.
        executor.buffers[hex(command[1])] =
            Words(std::max<size_t>(words, executor.minimumWords), 0u);
      } else if (verb == "words") {
        Words &buffer = executor.buffer(hex(command[1]));
        size_t count = std::stoul(command[2]);
        for (size_t index = 0; index < count; ++index)
          buffer.at(index) = hex(command[3 + index]);
      } else if (verb == "fill") {
        Words &buffer = executor.buffer(hex(command[1]));
        std::fill(buffer.begin(), buffer.end(), hex(command[2]));
      } else if (verb == "dispatch") {
        std::vector<uint32_t> arguments;
        size_t count = std::stoul(command[2]);
        for (size_t index = 0; index < count; ++index)
          arguments.push_back(hex(command[3 + index]));
        ++dispatches;
        auto found = fused.find(command[1]);
        if (found == fused.end()) {
          executor.library(command[1], arguments);
          continue;
        }
        const Fused &kernel = found->second;
        if (arguments.size() != kernel.inputs + 1 + kernel.scalars)
          throw std::runtime_error(kernel.name + ": wrong argument count");
        std::vector<const unsigned char *> inputs;
        for (uint32_t index = 0; index < kernel.inputs; ++index)
          inputs.push_back(reinterpret_cast<const unsigned char *>(
              executor.buffer(arguments[index]).data()));
        Words &output = executor.buffer(arguments[kernel.inputs]);
        std::vector<uint32_t> scalars(arguments.begin() + kernel.inputs + 1,
                                      arguments.end());
        Words result(kernel.words);
        // The draw starts from the target's words, which a partial-output
        // kernel keeps where it writes nothing.
        if (!session.drawGather(
                kernel.id, inputs.data(), inputs.size(), kernel.words * 4,
                reinterpret_cast<const unsigned char *>(scalars.data()),
                scalars.size() * 4,
                reinterpret_cast<unsigned char *>(result.data()),
                reinterpret_cast<const unsigned char *>(output.data())))
          throw std::runtime_error("draw failed for " + kernel.name);
        ++draws;
        std::copy(result.begin(),
                  result.begin() + std::min(result.size(), output.size()),
                  output.begin());
      } else if (verb == "copy") {
        Words &source = executor.buffer(hex(command[1]));
        Words &destination = executor.buffer(hex(command[2]));
        size_t sourceByte = std::stoul(command[3]),
               destinationByte = std::stoul(command[4]),
               bytes = std::stoul(command[5]);
        if (sourceByte % 4 || destinationByte % 4 || bytes % 4)
          throw std::runtime_error("unaligned copy");
        for (size_t word = 0; word < bytes / 4; ++word)
          destination.at(destinationByte / 4 + word) =
              source.at(sourceByte / 4 + word);
      } else if (verb == "save") {
        Words &buffer = executor.buffer(hex(command[1]));
        size_t words = std::stoul(command[2]);
        std::ofstream output(command[3], std::ios::binary);
        output.write(reinterpret_cast<const char *>(buffer.data()),
                     static_cast<std::streamsize>(words * 4));
        if (!output)
          throw std::runtime_error("cannot write " + command[3]);
      } else {
        throw std::runtime_error("unknown trace command " + verb);
      }
    }
    std::cout << "dispatches=" << dispatches << " draws=" << draws
              << " device=\"" << session.deviceName() << "\"\n";
    return 0;
  } catch (const std::exception &failure) {
    std::cerr << failure.what() << "\n";
    return 1;
  }
}
