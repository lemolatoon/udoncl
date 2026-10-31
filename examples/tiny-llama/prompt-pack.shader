Shader "Hidden/UdonCL/TinyLLaMA/PromptPack" {
Properties { _VRC_Input ("Packed model state", 2D) = "black" {} }
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _VRC_Input;
int _Token0;
int _Token1;
int _Token2;
int _Token3;
struct appdata { float4 vertex : POSITION; };
struct v2f { float4 vertex : SV_POSITION; };
v2f vert(appdata input) {
    v2f output;
    output.vertex = UnityObjectToClipPos(input.vertex);
    return output;
}
float4 encodeFloat(float value) {
    uint bits = asuint(value);
    return float4(bits & 255u, (bits >> 8u) & 255u,
                  (bits >> 16u) & 255u, (bits >> 24u) & 255u) / 255.0;
}
float4 frag(v2f input) : SV_Target {
    uint word = min(2047u, uint(input.vertex.x));
    if (word >= 64u)
        return _VRC_Input.Load(int3(word, 0, 0));
    uint position = word / 16u;
    uint token = word % 16u;
    int selected = position == 0u ? _Token0 :
                   position == 1u ? _Token1 :
                   position == 2u ? _Token2 : _Token3;
    return encodeFloat(token == uint(selected & 15) ? 1.0 : 0.0);
}
ENDCG
} }
}
