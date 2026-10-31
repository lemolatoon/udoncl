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
uint digitSegments(uint digit) {
    if (digit == 0u) return 0x3fu;
    if (digit == 1u) return 0x06u;
    if (digit == 2u) return 0x5bu;
    if (digit == 3u) return 0x4fu;
    if (digit == 4u) return 0x66u;
    if (digit == 5u) return 0x6du;
    if (digit == 6u) return 0x7du;
    if (digit == 7u) return 0x07u;
    if (digit == 8u) return 0x7fu;
    return 0x6fu;
}
float box(float2 position2d, float2 center, float2 halfSize) {
    float2 distance = abs(position2d - center) - halfSize;
    return 1.0 - step(0.0, max(distance.x, distance.y));
}
fixed4 frag(v2f input) : SV_Target {
    uint prediction = 0u;
    float best = decodeFloat(_Logits.Load(int3(0, 0, 0)));
    [unroll] for (uint digit = 1u; digit < 10u; ++digit) {
        float score = decodeFloat(_Logits.Load(int3(digit, 0, 0)));
        if (score > best) { best = score; prediction = digit; }
    }
    float ink = _MainTex.Sample(sampler_MainTex, input.uv).r;
    float2 glyphPosition = (input.uv - float2(0.82, 0.74)) / float2(0.13, 0.22);
    uint segments = digitSegments(prediction);
    float glyph = 0.0;
    if (segments & 0x01u) glyph = max(glyph, box(glyphPosition, float2(0, 0.90), float2(0.56, 0.10)));
    if (segments & 0x02u) glyph = max(glyph, box(glyphPosition, float2(0.66, 0.46), float2(0.10, 0.36)));
    if (segments & 0x04u) glyph = max(glyph, box(glyphPosition, float2(0.66, -0.46), float2(0.10, 0.36)));
    if (segments & 0x08u) glyph = max(glyph, box(glyphPosition, float2(0, -0.90), float2(0.56, 0.10)));
    if (segments & 0x10u) glyph = max(glyph, box(glyphPosition, float2(-0.66, -0.46), float2(0.10, 0.36)));
    if (segments & 0x20u) glyph = max(glyph, box(glyphPosition, float2(-0.66, 0.46), float2(0.10, 0.36)));
    if (segments & 0x40u) glyph = max(glyph, box(glyphPosition, float2(0, 0), float2(0.56, 0.10)));
    return float4(max(ink, glyph), ink * (1.0 - glyph), ink * (1.0 - glyph), 1.0);
}
ENDCG
} }
}
