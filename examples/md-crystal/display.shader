Shader "UdonCL/LennardJonesDisplay"
{
    Properties
    {
        _StateTex ("Packed particle state", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Cull Off
            CGPROGRAM
            #pragma target 5.0
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            Texture2D<float4> _StateTex;

            struct Attributes { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct Varyings { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };

            Varyings vert(Attributes input)
            {
                Varyings output;
                output.position = UnityObjectToClipPos(input.vertex);
                output.uv = input.uv;
                return output;
            }

            float decode(uint word)
            {
                uint4 bytes = uint4(round(_StateTex.Load(int3(word, 0, 0)) * 255.0));
                return asfloat(bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u));
            }

            float4 frag(Varyings input) : SV_Target
            {
                const float2 box = float2(8.979696, 7.776645);
                float3 color = lerp(float3(0.015, 0.045, 0.10), float3(0.02, 0.12, 0.20), input.uv.y);
                [unroll] for (uint particle = 0u; particle < 64u; ++particle)
                {
                    uint baseWord = particle * 4u;
                    float2 position = float2(decode(baseWord), decode(baseWord + 1u)) / box;
                    float2 velocity = float2(decode(baseWord + 2u), decode(baseWord + 3u));
                    float distanceToParticle = length((input.uv - position) * float2(box.x / box.y, 1.0));
                    float heat = saturate(dot(velocity, velocity) * 6.0);
                    float3 particleColor = lerp(float3(0.20, 0.65, 1.0), float3(1.0, 0.18, 0.04), heat);
                    color = lerp(color, particleColor, smoothstep(0.035, 0.018, distanceToParticle));
                }
                return float4(color, 1.0);
            }
            ENDCG
        }
    }
}
