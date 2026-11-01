Shader "UdonCL/UdonATen/WeightInt4packMmCompose" {
Properties {
    _WeightPage ("External weight page", 2D) = "black" {}
    _Activation ("Activation state", 2D) = "black" {}
    _Accumulator ("Prior output state", 2D) = "black" {}
    _ActivationOffset ("Activation offset", Int) = 0
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
int _ActivationOffset;
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
float4 frag(v2f input) : SV_Target {
    uint word = min(8191u, uint(input.uv.x * 8192.0));
    float4 value = _WeightPage.Load(int3(word, 0, 0));
    if (word >= 6144u && word < 6144u + (uint)_K)
        value = _Activation.Load(
            int3((uint)_ActivationOffset + (uint)_KOffset + word - 6144u, 0, 0));
    if (_Accumulate != 0 && word >= 7168u && word < 7168u + (uint)_N)
        value = _Accumulator.Load(
            int3((uint)_AccumulatorOffset + word - 7168u, 0, 0));
    return value;
}
ENDCG
} }
}
