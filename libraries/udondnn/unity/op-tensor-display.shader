Shader "UdonCL/UdonDNN/OpTensorDisplay" {
Properties { _Result ("Packed OpTensor result", 2D) = "black" {} }
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _Result;
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
    uint x = min(3u, uint(input.uv.x * 4.0));
    uint y = min(2u, uint(input.uv.y * 3.0));
    float value = decodeFloat(_Result.Load(int3(1536u + y * 4u + x, 0, 0)));
    float magnitude = saturate(abs(value) * 0.25);
    return value >= 0.0 ? fixed4(magnitude, 0.6, 0.15, 1.0)
                        : fixed4(0.15, 0.6, magnitude, 1.0);
}
ENDCG
} }
}
