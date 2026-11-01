Shader "UdonCL/UdonATen/WeightInt4packMmCompose" {
Properties {
    _WeightPage ("External weight page", 2D) = "black" {}
    _CompactPage ("Compact Q4/BF16 page", Int) = 0
    _PageRow ("Compact atlas row", Int) = 0
    _Activation ("Activation state", 2D) = "black" {}
    _Accumulator ("Prior output state", 2D) = "black" {}
    _ActivationOffset ("Activation offset", Int) = 0
    _ActivationWidth ("Activation texture width", Int) = 8192
    _AccumulatorOffset ("Accumulator offset", Int) = 7168
    _KOffset ("K tile offset", Int) = 0
    _N ("Output features", Int) = 0
    _K ("Input tile features", Int) = 0
    _Accumulate ("Accumulate prior output", Int) = 0
}
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _WeightPage;
Texture2D<float4> _Activation;
Texture2D<float4> _Accumulator;
int _CompactPage;
int _PageRow;
int _ActivationOffset;
int _ActivationWidth;
int _AccumulatorOffset;
int _KOffset;
int _N;
int _K;
int _Accumulate;
struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
struct v2f { float4 vertex : SV_POSITION; float2 uv : TEXCOORD0; };
v2f vert(appdata input) {
    v2f output;
    output.vertex = UnityObjectToClipPos(input.vertex);
    output.uv = input.uv;
    return output;
}
uint loadByte(uint address) {
    float4 packed = _WeightPage.Load(int3(address >> 2u, (uint)_PageRow, 0));
    uint lane = address & 3u;
    float value = lane == 0u ? packed.r : lane == 1u ? packed.g : lane == 2u ? packed.b : packed.a;
    return (uint)round(saturate(value) * 255.0);
}
float4 encodeFloat(float value) {
    uint bits = asuint(value);
    return float4(
        (float)(bits & 255u),
        (float)((bits >> 8u) & 255u),
        (float)((bits >> 16u) & 255u),
        (float)((bits >> 24u) & 255u)) / 255.0;
}
float compactWeight(uint word) {
    uint row = word / 72u;
    uint local = word - row * 72u;
    uint block = local / 18u;
    uint slot = local - block * 18u;
    if (slot < 16u)
        return (float)loadByte(row * 80u + block * 16u + slot);
    uint qparam = row * 80u + 64u + block * 4u + (slot - 16u) * 2u;
    uint bits = (loadByte(qparam) | (loadByte(qparam + 1u) << 8u)) << 16u;
    return asfloat(bits);
}
float4 frag(v2f input) : SV_Target {
    uint word = min(8191u, uint(input.uv.x * 8192.0));
    float4 value = _CompactPage != 0
        ? (word < 4608u ? encodeFloat(compactWeight(word)) : float4(0.0, 0.0, 0.0, 0.0))
        : _WeightPage.Load(int3(word, 0, 0));
    if (word >= 6144u && word < 6144u + (uint)_K) {
        uint activationWidth = max(1u, (uint)_ActivationWidth);
        uint activation = (uint)_ActivationOffset + (uint)_KOffset + word - 6144u;
        value = _Activation.Load(
            int3(activation % activationWidth, activation / activationWidth, 0));
    }
    if (_Accumulate != 0 && word >= 7168u && word < 7168u + (uint)_N)
        value = _Accumulator.Load(
            int3((uint)_AccumulatorOffset + word - 7168u, 0, 0));
    return value;
}
ENDCG
} }
}
