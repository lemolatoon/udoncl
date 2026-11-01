Shader "UdonCL/UdonBLAS/SdotDisplay" {
Properties { _Result ("Packed SDOT result", 2D) = "black" {} }
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _Result;
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
    float value = decodeFloat(_Result.Load(int3(768, 0, 0)));
    float magnitude = saturate(abs(value) * 0.125);
    return value >= 0.0 ? fixed4(0.1, magnitude, 0.2, 1.0)
                        : fixed4(magnitude, 0.1, 0.2, 1.0);
}
ENDCG
} }
}
