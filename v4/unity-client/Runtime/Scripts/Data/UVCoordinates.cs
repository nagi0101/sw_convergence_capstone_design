using UnityEngine;

namespace SGAPS.Runtime.Data
{
    /// <summary>
    /// Represents UV coordinates received from the server for pixel sampling.
    /// </summary>
    [System.Serializable]
    public class UVCoordinates
    {
        /// <summary>
        /// Target frame ID these coordinates should be used for.
        /// </summary>
        public ulong TargetFrameId;

        /// <summary>
        /// Array of UV coordinate pairs.
        /// </summary>
        public Vector2[] Coordinates;

        /// <summary>
        /// Creates empty UV coordinates.
        /// </summary>
        public UVCoordinates()
        {
            TargetFrameId = 0;
            Coordinates = System.Array.Empty<Vector2>();
        }

        /// <summary>
        /// Creates UV coordinates with specified values.
        /// </summary>
        /// <param name="targetFrameId">Target frame ID</param>
        /// <param name="coordinates">UV coordinate array</param>
        public UVCoordinates(ulong targetFrameId, Vector2[] coordinates)
        {
            TargetFrameId = targetFrameId;
            Coordinates = coordinates ?? System.Array.Empty<Vector2>();
        }

        /// <summary>
        /// Number of coordinates.
        /// </summary>
        public int Count => Coordinates?.Length ?? 0;

        /// <summary>
        /// Gets a coordinate by index.
        /// </summary>
        public Vector2 this[int index] => Coordinates[index];
    }

    /// <summary>
    /// Defines the sampling pattern for initial UV coordinate generation.
    /// </summary>
    public enum SamplingPattern
    {
        /// <summary>
        /// Uniform grid pattern covering the image evenly.
        /// </summary>
        UniformGrid,

        /// <summary>
        /// Random sampling with uniform distribution.
        /// </summary>
        Random,

        /// <summary>
        /// Stratified random sampling for better coverage.
        /// </summary>
        Stratified,

        /// <summary>
        /// Checkerboard pattern.
        /// </summary>
        Checkerboard
    }
}
