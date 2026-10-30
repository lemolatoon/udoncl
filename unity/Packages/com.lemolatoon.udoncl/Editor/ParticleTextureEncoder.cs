using System;
using System.Collections.Generic;

namespace Lemolatoon.Udoncl.Editor
{
    public struct ParticleState
    {
        public float X;
        public float Y;
        public float Z;
        public float Vx;
        public float Vy;
        public float Vz;

        public ParticleState(float x, float y, float z, float vx, float vy, float vz)
        {
            X = x;
            Y = y;
            Z = z;
            Vx = vx;
            Vy = vy;
            Vz = vz;
        }
    }

    public static class ParticleTextureEncoder
    {
        public const int ParticleCount = 32;
        public const int WordsPerParticle = 8;
        public const int TextureWidth = ParticleCount * WordsPerParticle;
        public const int TextureHeight = 1;

        public static byte[] Encode(IList<ParticleState> particles)
        {
            if (particles == null || particles.Count != ParticleCount)
                throw new ArgumentException("Exactly 32 particle states are required.", "particles");

            byte[] bytes = new byte[TextureWidth * TextureHeight * 4];
            int offset = 0;
            for (int i = 0; i < particles.Count; i++)
            {
                ParticleState particle = particles[i];
                WriteFloat32LittleEndian(bytes, ref offset, particle.X);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Y);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Z);
                WriteFloat32LittleEndian(bytes, ref offset, 0.0f);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Vx);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Vy);
                WriteFloat32LittleEndian(bytes, ref offset, particle.Vz);
                WriteFloat32LittleEndian(bytes, ref offset, 0.0f);
            }
            return bytes;
        }

        public static ParticleState[] CreateLennardJonesInitialState()
        {
            ParticleState[] particles = new ParticleState[ParticleCount];
            const float lattice = 1.587401052f;
            float momentumX = 0.0f;
            float momentumY = 0.0f;
            float momentumZ = 0.0f;
            float[,] basis = { { 0, 0, 0 }, { 0, 0.5f, 0.5f }, { 0.5f, 0, 0.5f }, { 0.5f, 0.5f, 0 } };
            for (int i = 0; i < ParticleCount; i++)
            {
                int cell = i / 4, b = i % 4;
                int xCell = cell % 2, yCell = (cell / 2) % 2, zCell = cell / 4;
                float x = (xCell + basis[b, 0]) * lattice;
                float y = (yCell + basis[b, 1]) * lattice;
                float z = (zCell + basis[b, 2]) * lattice;
                float vx = 0.012f * (float)Math.Sin(i);
                float vy = 0.012f * (float)Math.Cos(i);
                float vz = 0.012f * (float)Math.Sin(i * 1.618f);
                particles[i] = new ParticleState(x, y, z, vx, vy, vz);
                momentumX += vx;
                momentumY += vy;
                momentumZ += vz;
            }
            momentumX /= ParticleCount;
            momentumY /= ParticleCount;
            momentumZ /= ParticleCount;
            for (int i = 0; i < ParticleCount; i++)
            {
                ParticleState particle = particles[i];
                particle.Vx -= momentumX;
                particle.Vy -= momentumY;
                particle.Vz -= momentumZ;
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
