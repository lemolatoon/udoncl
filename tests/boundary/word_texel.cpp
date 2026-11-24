// Runs each gather kernel lowered twice, once with one word per RGBA8 texel
// and once with four words per RGBA32 float texel, on the native
// fragment renderer over the same 8192-word-wide surfaces, and compares the
// two outputs word for word. The storage layout is the only difference, so
// every word must be bit-identical.
//
// usage: word-texel-kernel <vertex> <cases>
// Each line of <cases> is tab-separated:
//   name, RGBA8 fragment, word-texel fragment, input count,
//   scalar payload (hex bytes, "-" for none), input words file, surface words
//   [, destination words file]
// The input file holds the input surfaces one after another. With a
// destination, both draws start from its words instead of a cleared target
// (a partial-output kernel keeps the words it discards), and the RGBA8
// output is written to "<destination file>.out".
#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr uint32_t surfaceWidth = 8192;

std::vector<std::string> split(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, '\t'))
    fields.push_back(field);
  return fields;
}

std::vector<unsigned char> hexBytes(const std::string &text) {
  std::vector<unsigned char> bytes;
  if (text == "-")
    return bytes;
  for (size_t index = 0; index + 1 < text.size(); index += 2)
    bytes.push_back(
        static_cast<unsigned char>(std::stoul(text.substr(index, 2), 0, 16)));
  return bytes;
}

std::vector<uint32_t> draw(native_opencl::Session &session, uint32_t id,
                           const std::vector<uint32_t> &inputs,
                           uint32_t inputCount, uint32_t surfaceWords,
                           const std::vector<unsigned char> &scalars,
                           const std::vector<uint32_t> &destination) {
  std::vector<const unsigned char *> pointers;
  for (uint32_t input = 0; input < inputCount; ++input)
    pointers.push_back(reinterpret_cast<const unsigned char *>(
        inputs.data() + size_t(input) * surfaceWords));
  std::vector<uint32_t> output(surfaceWords, 0xdeadbeefu);
  if (!session.drawGather(
          id, pointers.data(), inputCount, surfaceWords * 4,
          scalars.empty() ? nullptr : scalars.data(), scalars.size(),
          reinterpret_cast<unsigned char *>(output.data()),
          destination.empty()
              ? nullptr
              : reinterpret_cast<const unsigned char *>(destination.data())))
    throw std::runtime_error("gather draw failed");
  return output;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: word-texel-kernel <vertex> <cases>\n";
    return 2;
  }
  std::ifstream cases(argv[2]);
  if (!cases) {
    std::cerr << "cannot read " << argv[2] << "\n";
    return 2;
  }
  bool ok = true;
  std::string line;
  // Consecutive cases of one kernel share its compiled shaders.
  std::unique_ptr<native_opencl::Session> session;
  std::string sessionKey;
  while (std::getline(cases, line)) {
    if (line.empty())
      continue;
    auto fields = split(line);
    if (fields.size() != 7 && fields.size() != 8) {
      std::cerr << "malformed case: " << line << std::endl;
      return 2;
    }
    try {
      const std::string &name = fields[0];
      uint32_t inputCount = static_cast<uint32_t>(std::stoul(fields[3]));
      auto scalars = hexBytes(fields[4]);
      uint32_t surfaceWords = static_cast<uint32_t>(std::stoul(fields[6]));
      uint32_t surfaceHeight = surfaceWords / surfaceWidth;
      std::ifstream file(fields[5], std::ios::binary);
      std::vector<uint32_t> inputs(size_t(inputCount) * surfaceWords);
      file.read(reinterpret_cast<char *>(inputs.data()),
                std::streamsize(inputs.size() * 4));
      if (!file)
        throw std::runtime_error("short input file " + fields[5]);
      std::vector<uint32_t> destination;
      if (fields.size() == 8) {
        std::ifstream target(fields[7], std::ios::binary);
        destination.resize(surfaceWords);
        target.read(reinterpret_cast<char *>(destination.data()),
                    std::streamsize(destination.size() * 4));
        if (!target)
          throw std::runtime_error("short destination file " + fields[7]);
      }
      uint32_t scalarCount = static_cast<uint32_t>(scalars.size() / 4);
      std::string key = fields[1] + "|" + fields[2] + "|" + fields[3] + "|" +
                        fields[6] + "|" + std::to_string(scalarCount);
      if (!session || key != sessionKey) {
        session.reset();
        native_opencl::GatherBinding narrow{
            1,         surfaceWords, scalarCount,   "narrow",
            fields[1], surfaceWidth, surfaceHeight, inputCount};
        native_opencl::GatherBinding wide = narrow;
        wide.id = 2;
        wide.name = "wide";
        wide.fragment = fields[2];
        wide.texelWords = 4;
        session = std::make_unique<native_opencl::Session>(
            "", std::vector<native_opencl::GatherBinding>{narrow, wide},
            argv[1]);
        sessionKey = key;
      }
      auto expected = draw(*session, 1, inputs, inputCount, surfaceWords,
                           scalars, destination);
      auto actual = draw(*session, 2, inputs, inputCount, surfaceWords, scalars,
                         destination);
      if (!destination.empty()) {
        std::ofstream saved(fields[7] + ".out", std::ios::binary);
        saved.write(reinterpret_cast<const char *>(expected.data()),
                    std::streamsize(expected.size() * 4));
      }
      uint32_t exact = 0, nonzero = 0;
      int64_t first = -1;
      for (uint32_t index = 0; index < surfaceWords; ++index) {
        exact += expected[index] == actual[index];
        nonzero += expected[index] != 0;
        if (expected[index] != actual[index] && first < 0)
          first = index;
      }
      std::cout << name << " exact=" << exact << "/" << surfaceWords
                << " nonzero=" << nonzero;
      if (first >= 0)
        std::cout << " first=" << first << ":" << std::hex
                  << expected[size_t(first)] << ":" << actual[size_t(first)]
                  << std::dec;
      std::cout << std::endl;
      ok &= exact == surfaceWords;
    } catch (const std::exception &failure) {
      std::cout << fields[0] << " error=" << failure.what() << std::endl;
      session.reset();
      ok = false;
    }
  }
  return ok ? 0 : 1;
}
