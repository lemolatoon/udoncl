Shader "UdonCL/UdonATen/WeightInt4packMmScatter" {
Properties {
    _TileState ("Completed tile state", 2D) = "black" {}
    _PriorResult ("Prior output features", 2D) = "black" {}
    _OutputOffset ("Output feature offset", Integer) = 0
    _N ("Tile output features", Integer) = 0
    _OutputWidth ("Output texture width", Integer) = 8192
    _OutputFeatures ("Complete output features", Integer) = 0
    _HasPrior ("Copy prior output", Integer) = 0
}
SubShader { Pass {
CGPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma fragment frag
#include "UnityCG.cginc"
Texture2D<float4> _TileState;
Texture2D<float4> _PriorResult;
int _OutputOffset;
int _N;
int _OutputWidth;
int _OutputFeatures;
int _HasPrior;
struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
struct v2f { float4 vertex : SV_POSITION; float2 uv : TEXCOORD0; };
v2f vert(appdata input) {
    v2f output;
    output.vertex = UnityObjectToClipPos(input.vertex);
    output.uv = input.uv;
    return output;
}
float4 frag(v2f input) : SV_Target {
    uint width = max(1u, (uint)_OutputWidth);
    uint height = max(1u, ((uint)_OutputFeatures + width - 1u) / width);
    uint x = min(width - 1u, uint(input.uv.x * (float)width));
    uint y = min(height - 1u, uint(input.uv.y * (float)height));
    uint feature = min((uint)_OutputFeatures - 1u, y * width + x);
    if (feature >= (uint)_OutputOffset && feature < (uint)(_OutputOffset + _N))
        return _TileState.Load(int3(7168u + feature - (uint)_OutputOffset, 0, 0));
    if (_HasPrior != 0)
        return _PriorResult.Load(int3(feature % width, feature / width, 0));
    return float4(0.0, 0.0, 0.0, 0.0);
}
ENDCG
} }
}
