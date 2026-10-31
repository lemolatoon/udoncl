Shader "UdonCL/MNIST/Brush" {
Properties {
    _MainTex ("Drawing", 2D) = "black" {}
    _BrushX ("Brush X", Float) = -10
    _BrushY ("Brush Y", Float) = -10
    _Clear ("Clear", Float) = 0
}
SubShader { Pass { Cull Off ZWrite Off ZTest Always Blend Off
CGPROGRAM
#pragma target 5.0
#pragma vertex vert_img
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _MainTex;
float _BrushX;
float _BrushY;
float _Clear;

float4 frag(float4 position : SV_POSITION) : SV_Target {
    if (_Clear > 0.5)
        return 0.0;
    uint2 pixel = uint2(position.xy);
    float2 uv = (float2(pixel) + 0.5) / 28.0;
    float oldInk = _MainTex.Load(int3(pixel, 0)).r;
    float distanceInPixels = length((uv - float2(_BrushX, _BrushY)) * 28.0);
    float ink = saturate(1.75 - distanceInPixels);
    return max(oldInk, ink).xxxx;
}
ENDCG
} }
}
