Shader "UdonCL/MNIST/PackInput" {
Properties {
    _MainTex ("Drawing", 2D) = "black" {}
    _Weights ("Packed weights", 2D) = "black" {}
}
SubShader { Pass { Cull Off ZWrite Off ZTest Always Blend Off
CGPROGRAM
#pragma target 5.0
#pragma vertex vert_img
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _MainTex;
Texture2D<float4> _Weights;

float4 encodeFloat(float value) {
    uint bits = asuint(value);
    return float4(bits & 255u, (bits >> 8u) & 255u,
                  (bits >> 16u) & 255u, bits >> 24u) / 255.0;
}

float4 frag(float4 position : SV_POSITION) : SV_Target {
    uint width;
    uint height;
    _Weights.GetDimensions(width, height);
    uint index = uint(position.y) * width + uint(position.x);
    if (index >= 784u)
        return _Weights.Load(int3(index % width, index / width, 0));
    uint x = index % 28u;
    uint y = index / 28u;
    float ink = _MainTex.Load(int3(x, 27u - y, 0)).r;
    return encodeFloat(saturate(ink));
}
ENDCG
} }
}
