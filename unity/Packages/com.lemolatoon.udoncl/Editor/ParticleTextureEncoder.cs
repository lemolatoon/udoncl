using System;
using System.Collections.Generic;

namespace Lemolatoon.Udoncl.Editor
{
    public struct ParticleState
    {
        public float X;
        public float Y;
        public float Vx;
        public float Vy;

        public ParticleState(float x, float y, float vx, float vy)
        {
            X = x;
            Y = y;
            Vx = vx;
            Vy = vy;
        }
    }

    public static class ParticleTextureEncoder
    {
        public const int ParticleCount = 64;
        public const int TextureWidth = ParticleCount * 4;
        public const int TextureHeight = 1;

        public static byte[] Encode(IList<ParticleState> particles)
        {
            if (particles == null || particles.Count != ParticleCount)
                throw new ArgumentException("Exactly 64 particle states are required.", "particles");

            byte[] bytes = new byte[TextureWidth * TextureHeight * 4];
            int offset = 0;
            for (int i = 0; i < particles.Count; i++)
            {
                ParticleState particle = particles[i];
                WriteFloat32LittleEndian(bytes, ref offset, particle.X);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Y);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Vx);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Vy);
            }
            return bytes;
        }

        public static ParticleState[] CreateLennardJonesInitialState()
        {
            ParticleState[] particles = new ParticleState[ParticleCount];
            const float lattice = 1.122462f;
            float momentumX = 0.0f;
            float momentumY = 0.0f;
            for (int i = 0; i < ParticleCount; i++)
            {
                int row = i / 8;
                float x = ((i % 8) + 0.5f * (row % 2)) * lattice;
                float y = row * lattice * 0.8660254f;
                float vx = 0.02f * (float)Math.Sin(i);
                float vy = 0.02f * (float)Math.Cos(i);
                particles[i] = new ParticleState(x, y, vx, vy);
                momentumX += vx;
                momentumY += vy;
            }
            momentumX /= ParticleCount;
            momentumY /= ParticleCount;
            for (int i = 0; i < ParticleCount; i++)
            {
                ParticleState particle = particles[i];
                particle.Vx -= momentumX;
                particle.Vy -= momentumY;
                particles[i] = particle;
            }
            return particles;
        }

        private static void WriteFloat32LittleEndian(byte[] destination, ref int offset, float value)
        {
            if (float.IsNaN(value) || float.IsInfinity(value))
                throw new ArgumentException("Particle state values must be finite.");

            byte[] encoded = BitConverter.GetBytes(value);
            if (!BitConverter.IsLittleEndian)
            {
                Array.Reverse(encoded);
            }
            Buffer.BlockCopy(encoded, 0, destination, offset, 4);
            offset += 4;
        }
    }
}
