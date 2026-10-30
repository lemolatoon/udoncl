Shader "UdonCL/LennardJonesDisplay3D"
{
    Properties { _StateTex ("Packed 3D particle state", 2D) = "black" {} }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            Cull Front
            ZWrite On
            CGPROGRAM
            #pragma target 5.0
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            Texture2D<float4> _StateTex;
            struct Attributes { float4 vertex : POSITION; };
            struct Varyings { float4 position : SV_POSITION; float3 localPosition : TEXCOORD0; };
            Varyings vert(Attributes v) { Varyings o; o.position=UnityObjectToClipPos(v.vertex); o.localPosition=v.vertex.xyz; return o; }
            float decode(uint word) {
                uint4 b=uint4(round(_StateTex.Load(int3(word,0,0))*255.0));
                return asfloat(b.x|(b.y<<8u)|(b.z<<16u)|(b.w<<24u));
            }
            float4 frag(Varyings input) : SV_Target
            {
                const float box=3.174802104, radius=.065;
                float3 origin=mul(unity_WorldToObject,float4(_WorldSpaceCameraPos,1)).xyz;
                float3 ray=normalize(input.localPosition-origin);
                float rayEnd=length(input.localPosition-origin), closest=rayEnd+1;
                float3 hitColor=float3(.012,.025,.055);
                [unroll] for(uint particle=0u;particle<32u;++particle)
                {
                    uint baseWord=particle*8u;
                    float3 center=float3(decode(baseWord),decode(baseWord+1u),decode(baseWord+2u))/box-.5;
                    float3 velocity=float3(decode(baseWord+4u),decode(baseWord+5u),decode(baseWord+6u));
                    float along=dot(center-origin,ray);
                    float3 perpendicular=(origin+ray*along)-center;
                    float radial2=dot(perpendicular,perpendicular);
                    if(along>0&&along<closest&&along<rayEnd&&radial2<radius*radius)
                    {
                        closest=along;
                        float heat=saturate(dot(velocity,velocity)*2.5);
                        float3 normal=normalize(perpendicular-ray*sqrt(max(0,radius*radius-radial2)));
                        float light=.32+.68*saturate(dot(normal,normalize(float3(-.4,.7,-.55))));
                        hitColor=lerp(float3(.16,.65,1),float3(1,.14,.02),heat)*light;
                    }
                }
                float fog=saturate(rayEnd*.12);
                return float4(lerp(hitColor,float3(.018,.055,.09),closest>rayEnd?.35:fog),1);
            }
            ENDCG
        }
    }
}
