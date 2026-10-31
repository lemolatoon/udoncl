Shader "UdonCL/MNIST/Display" {
Properties {
    _MainTex ("Drawing", 2D) = "black" {}
    _Logits ("Packed logits", 2D) = "black" {}
}
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _MainTex;
Texture2D<float4> _Logits;
SamplerState sampler_MainTex;

struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
struct v2f { float4 vertex : SV_POSITION; float2 uv : TEXCOORD0; };
v2f vert(appdata input) {
    v2f output;
    output.vertex = UnityObjectToClipPos(input.vertex);
    output.uv = input.uv;
    return output;
}
float decodeFloat(float4 bytes) {
    uint4 b = uint4(round(bytes * 255.0));
    return asfloat(b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u));
}
fixed4 frag(v2f input) : SV_Target {
    uint prediction = 0u;
    float best = decodeFloat(_Logits.Load(int3(0, 0, 0)));
    [unroll] for (uint digit = 1u; digit < 10u; ++digit) {
        float score = decodeFloat(_Logits.Load(int3(digit, 0, 0)));
        if (score > best) { best = score; prediction = digit; }
    }
    float ink = _MainTex.Sample(sampler_MainTex, input.uv).r;
    float stripe = step(abs(input.uv.y - 0.08), 0.055);
    float selected = step(abs(input.uv.x - (float(prediction) + 0.5) / 10.0), 0.045);
    return float4(max(ink, stripe * selected), ink, ink, 1.0);
}
ENDCG
} }
}
