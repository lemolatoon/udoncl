#include "Renderer.h"
#include <cstring>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
namespace {
void check(HRESULT result) {
  if (FAILED(result))
    throw std::runtime_error("D3D11 native fragment operation failed: " +
                             std::to_string(result));
}
ComPtr<ID3DBlob> compile(const std::string &source, const char *entry,
                         const char *target) {
  ComPtr<ID3DBlob> code, errors;
  HRESULT result = D3DCompile(
      source.data(), source.size(), nullptr, nullptr, nullptr, entry, target,
      D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS, 0, &code,
      &errors);
  if (FAILED(result))
    throw std::runtime_error(
        errors
            ? std::string(static_cast<const char *>(errors->GetBufferPointer()),
                          errors->GetBufferSize())
            : "HLSL compilation failed");
  return code;
}
class Renderer final : public FragmentRenderer {
  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  ComPtr<ID3D11VertexShader> vertex;
  ComPtr<ID3D11PixelShader> fragment;
  ComPtr<ID3D11Texture2D> target, staging, input;
  ComPtr<ID3D11ShaderResourceView> inputView;
  uint32_t width = 1, height = 1;
  bool gather = false;
  ComPtr<ID3D11RenderTargetView> view;
  ComPtr<ID3D11Buffer> uniform;
  ComPtr<ID3D11RasterizerState> raster;

public:
  explicit Renderer(const std::string &path, uint32_t x = 0, uint32_t y = 1)
      : width(x ? x : 1), height(y), gather(x != 0) {
    if (!height || uint64_t(width) * height > 16384)
      throw std::runtime_error("gather extent exceeds supported profile");
    D3D_FEATURE_LEVEL requested = D3D_FEATURE_LEVEL_11_0, actual;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0,
                            &requested, 1, D3D11_SDK_VERSION, &device, &actual,
                            &context));
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("cannot read HLSL");
    auto pixel = compile(std::string(std::istreambuf_iterator<char>(file), {}),
                         "frag", "ps_5_0");
    auto triangle = compile(
        "float4 vert(uint id : SV_VertexID) : SV_Position { return float4(id "
        "== 2 ? 3.0 : -1.0, id == 1 ? 3.0 : -1.0, 0.0, 1.0); }",
        "vert", "vs_5_0");
    check(device->CreatePixelShader(
        pixel->GetBufferPointer(), pixel->GetBufferSize(), nullptr, &fragment));
    check(device->CreateVertexShader(triangle->GetBufferPointer(),
                                     triangle->GetBufferSize(), nullptr,
                                     &vertex));
    D3D11_TEXTURE2D_DESC texture{};
    texture.Width = width;
    texture.Height = height;
    texture.MipLevels = texture.ArraySize = 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.SampleDesc.Count = 1;
    texture.BindFlags = D3D11_BIND_RENDER_TARGET;
    check(device->CreateTexture2D(&texture, nullptr, &target));
    check(device->CreateRenderTargetView(target.Get(), nullptr, &view));
    if (gather) {
      texture.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      check(device->CreateTexture2D(&texture, nullptr, &input));
      check(device->CreateShaderResourceView(input.Get(), nullptr, &inputView));
    }
    texture.BindFlags = 0;
    texture.Usage = D3D11_USAGE_STAGING;
    texture.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    check(device->CreateTexture2D(&texture, nullptr, &staging));
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = 64;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    check(device->CreateBuffer(&buffer, nullptr, &uniform));
    D3D11_RASTERIZER_DESC rasterDesc{};
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.CullMode = D3D11_CULL_NONE;
    rasterDesc.DepthClipEnable = TRUE;
    check(device->CreateRasterizerState(&rasterDesc, &raster));
  }
  std::string deviceName() const override {
    return "Direct3D11 WARP (software)";
  }
  std::array<uint8_t, 4> draw(float value) override {
    auto bytes = drawWords(
        {},
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(&value), 4));
    std::array<uint8_t, 4> result;
    std::memcpy(result.data(), bytes.data(), 4);
    return result;
  }
  std::vector<uint8_t> drawWords(std::span<const uint8_t> state,
                                 std::span<const uint8_t> scalars) override {
    if (scalars.size() > 64 ||
        (gather ? state.size() != width * height * 4 : !state.empty()))
      throw std::runtime_error("invalid fragment input/payload size");
    uint8_t data[64]{};
    if (!scalars.empty())
      std::memcpy(data, scalars.data(), scalars.size());
    context->UpdateSubresource(uniform.Get(), 0, nullptr, data, 0, 0);
    if (gather) {
      ID3D11ShaderResourceView *srv = nullptr;
      context->PSSetShaderResources(0, 1, &srv);
      context->UpdateSubresource(input.Get(), 0, nullptr, state.data(),
                                 width * 4, 0);
      srv = inputView.Get();
      context->PSSetShaderResources(0, 1, &srv);
    }
    float clear[4]{1, 0, 1, 1};
    context->ClearRenderTargetView(view.Get(), clear);
    ID3D11RenderTargetView *rt = view.Get();
    context->OMSetRenderTargets(1, &rt, nullptr);
    context->OMSetBlendState(nullptr, nullptr, 0xffffffff);
    D3D11_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    context->RSSetViewports(1, &viewport);
    context->RSSetState(raster.Get());
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(vertex.Get(), nullptr, 0);
    context->PSSetShader(fragment.Get(), nullptr, 0);
    ID3D11Buffer *cb = uniform.Get();
    context->PSSetConstantBuffers(gather ? 1 : 0, 1, &cb);
    context->Draw(3, 0);
    context->CopyResource(staging.Get(), target.Get());
    D3D11_MAPPED_SUBRESOURCE mapped{};
    check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped));
    std::vector<uint8_t> result(width * height * 4);
    for (uint32_t row = 0; row < height; ++row)
      std::memcpy(result.data() + size_t(row) * width * 4,
                  static_cast<const uint8_t *>(mapped.pData) +
                      size_t(row) * mapped.RowPitch,
                  size_t(width) * 4);
    context->Unmap(staging.Get(), 0);
    return result;
  }
};
} // namespace
std::unique_ptr<FragmentRenderer>
createFragmentRenderer(const std::string &shader, const std::string &) {
  return std::make_unique<Renderer>(shader);
}

std::unique_ptr<FragmentRenderer>
createGatherRenderer(const std::string &shader, const std::string &,
                     uint32_t width, uint32_t height) {
  if (!width || !height || uint64_t(width) * height < 4)
    throw std::runtime_error("invalid gather extent");
  return std::make_unique<Renderer>(shader, width, height);
}
