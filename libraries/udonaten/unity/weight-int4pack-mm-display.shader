Shader "UdonCL/UdonATen/WeightInt4packMmDisplay" {
Properties { _Result ("ATen INT4 matrix result", 2D) = "black" {} }
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
    uint row = min(7u, uint(input.uv.x * 8.0));
    float value = decodeFloat(_Result.Load(int3(row, 0, 0)));
    float magnitude = saturate(abs(value) * 0.25);
    return value >= 0.0 ? fixed4(0.1, magnitude, 0.8, 1.0)
                        : fixed4(magnitude, 0.1, 0.8, 1.0);
}
ENDCG
} }
}
