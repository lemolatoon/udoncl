Shader "UdonCL/UdonDNN/ConvolutionDisplay" {
Properties { _Result ("Packed convolution result", 2D) = "black" {} }
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
    uint row = min(11u, uint(input.uv.y * 12.0));
    float value = decodeFloat(_Result.Load(int3(1536u + row * 4u + x, 0, 0)));
    float magnitude = saturate(abs(value) * 0.2);
    return value >= 0.0 ? fixed4(0.08, magnitude, 0.35, 1.0)
                        : fixed4(magnitude, 0.08, 0.35, 1.0);
}
ENDCG
} }
}
