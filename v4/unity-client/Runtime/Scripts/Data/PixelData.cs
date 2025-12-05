using UnityEngine;

namespace SGAPS.Runtime.Data
{
    /// <summary>
    /// Represents a single sampled pixel with UV coordinates and grayscale value.
    /// </summary>
    [System.Serializable]
    public struct PixelData
    {
        /// <summary>
        /// Horizontal UV coordinate in range [0, 1].
        /// </summary>
        public float U;

        /// <summary>
        /// Vertical UV coordinate in range [0, 1].
        /// </summary>
        public float V;

        /// <summary>
        /// Grayscale pixel value in range [0, 255].
        /// </summary>
        public byte Value;

        /// <summary>
        /// Creates a new PixelData instance.
        /// </summary>
        /// <param name="u">U coordinate [0, 1]</param>
        /// <param name="v">V coordinate [0, 1]</param>
        /// <param name="value">Grayscale value [0, 255]</param>
        public PixelData(float u, float v, byte value)
        {
            U = u;
            V = v;
            Value = value;
        }

        /// <summary>
        /// Returns UV as a Vector2.
        /// </summary>
        public Vector2 UV => new Vector2(U, V);

        /// <summary>
        /// Returns normalized value in range [0, 1].
        /// </summary>
        public float NormalizedValue => Value / 255f;

        public override string ToString()
        {
            return $"Pixel(U:{U:F3}, V:{V:F3}, Value:{Value})";
        }
    }
}
