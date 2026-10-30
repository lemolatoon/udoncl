using NUnit.Framework;

namespace Lemolatoon.Udoncl.Editor.Tests
{
    public sealed class ParticleTextureEncoderTests
    {
        [Test]
        public void EncodesAlignedThreeDimensionalParticleState()
        {
            ParticleState[] particles = ParticleTextureEncoder.CreateLennardJonesInitialState();
            particles[0] = new ParticleState(1.0f, -2.0f, 3.0f, 0.5f, -0.25f, 0.125f);

            byte[] data = ParticleTextureEncoder.Encode(particles);

            Assert.AreEqual(256 * 4, data.Length);
            Assert.AreEqual(32, ParticleTextureEncoder.ParticleCount);
            Assert.AreEqual(8, ParticleTextureEncoder.WordsPerParticle);
            CollectionAssert.AreEqual(new byte[] { 0x00, 0x00, 0x80, 0x3f }, new byte[] { data[0], data[1], data[2], data[3] });
            CollectionAssert.AreEqual(new byte[] { 0x00, 0x00, 0x00, 0xc0 }, new byte[] { data[4], data[5], data[6], data[7] });
            CollectionAssert.AreEqual(new byte[] { 0x00, 0x00, 0x40, 0x40 }, new byte[] { data[8], data[9], data[10], data[11] });
            CollectionAssert.AreEqual(new byte[] { 0x00, 0x00, 0x00, 0x3f }, new byte[] { data[16], data[17], data[18], data[19] });
        }
    }
}
