Shader "Hidden/UdonCL/TinyLLaMA/ArgmaxReadback" {
Properties { _Logits ("Packed logits", 2D) = "black" {} }
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _Logits;
struct appdata { float4 vertex : POSITION; };
struct v2f { float4 vertex : SV_POSITION; };
v2f vert(appdata input) {
    v2f output;
    output.vertex = UnityObjectToClipPos(input.vertex);
    return output;
}
float decodeFloat(float4 bytes) {
    uint4 b = uint4(round(bytes * 255.0));
    return asfloat(b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u));
}
fixed4 frag(v2f input) : SV_Target {
    uint prediction = 0u;
    float best = decodeFloat(_Logits.Load(int3(352, 0, 0)));
    [unroll] for (uint token = 1u; token < 16u; ++token) {
        float score = decodeFloat(_Logits.Load(int3(352u + token, 0, 0)));
        if (score > best) {
            best = score;
            prediction = token;
        }
    }
    return fixed4(float(prediction) / 255.0, 0.0, 0.0, 1.0);
}
ENDCG
} }
}
