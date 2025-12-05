using System;
using UnityEngine;
using SGAPS.Runtime.Data;

namespace SGAPS.Runtime.Core
{
    /// <summary>
    /// Samples pixels from a RenderTexture at specified UV coordinates.
    /// Generates initial sampling patterns and performs efficient pixel reads.
    /// </summary>
    public class PixelSampler
    {
        private readonly int sampleCount;
        private Texture2D readbackTexture;
        private Vector2Int lastResolution;

        /// <summary>
        /// Number of samples to generate/collect.
        /// </summary>
        public int SampleCount => sampleCount;

        /// <summary>
        /// Creates a new PixelSampler.
        /// </summary>
        /// <param name="sampleCount">Number of pixels to sample</param>
        public PixelSampler(int sampleCount = 500)
        {
            if (sampleCount <= 0)
            {
                throw new ArgumentException("sampleCount must be positive", nameof(sampleCount));
            }

            this.sampleCount = sampleCount;
        }

        /// <summary>
        /// Generates initial UV coordinates using the specified pattern.
        /// </summary>
        /// <param name="pattern">Sampling pattern to use</param>
        /// <param name="seed">Random seed (for random patterns)</param>
        /// <returns>UVCoordinates with generated positions</returns>
        public UVCoordinates GenerateInitialPattern(SamplingPattern pattern, int seed = 0)
        {
            Vector2[] coords;

            switch (pattern)
            {
                case SamplingPattern.UniformGrid:
                    coords = GenerateUniformGrid();
                    break;
                case SamplingPattern.Random:
                    coords = GenerateRandom(seed);
                    break;
                case SamplingPattern.Stratified:
                    coords = GenerateStratified(seed);
                    break;
                case SamplingPattern.Checkerboard:
                    coords = GenerateCheckerboard();
                    break;
                default:
                    coords = GenerateUniformGrid();
                    break;
            }

            return new UVCoordinates(0, coords);
        }

        /// <summary>
        /// Generates uniform grid UV coordinates.
        /// </summary>
        private Vector2[] GenerateUniformGrid()
        {
            // Calculate grid dimensions (assuming 4:3 aspect ratio)
            float aspectRatio = 4f / 3f;
            int rows = Mathf.CeilToInt(Mathf.Sqrt(sampleCount / aspectRatio));
            int cols = Mathf.CeilToInt(rows * aspectRatio);

            // Adjust to match sample count
            while (rows * cols < sampleCount)
            {
                if ((float)cols / rows < aspectRatio)
                    cols++;
                else
                    rows++;
            }

            Vector2[] coords = new Vector2[sampleCount];
            int index = 0;

            for (int i = 0; i < rows && index < sampleCount; i++)
            {
                for (int j = 0; j < cols && index < sampleCount; j++)
                {
                    float u = (j + 0.5f) / cols;
                    float v = (i + 0.5f) / rows;
                    coords[index++] = new Vector2(u, v);
                }
            }

            return coords;
        }

        /// <summary>
        /// Generates random UV coordinates.
        /// </summary>
        private Vector2[] GenerateRandom(int seed)
        {
            UnityEngine.Random.InitState(seed);
            Vector2[] coords = new Vector2[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                coords[i] = new Vector2(
                    UnityEngine.Random.value,
                    UnityEngine.Random.value
                );
            }

            return coords;
        }

        /// <summary>
        /// Generates stratified random UV coordinates.
        /// </summary>
        private Vector2[] GenerateStratified(int seed)
        {
            UnityEngine.Random.InitState(seed);

            float aspectRatio = 4f / 3f;
            int rows = Mathf.CeilToInt(Mathf.Sqrt(sampleCount / aspectRatio));
            int cols = Mathf.CeilToInt(rows * aspectRatio);

            Vector2[] coords = new Vector2[sampleCount];
            int index = 0;

            float cellWidth = 1f / cols;
            float cellHeight = 1f / rows;

            for (int i = 0; i < rows && index < sampleCount; i++)
            {
                for (int j = 0; j < cols && index < sampleCount; j++)
                {
                    float u = (j + UnityEngine.Random.value) * cellWidth;
                    float v = (i + UnityEngine.Random.value) * cellHeight;
                    coords[index++] = new Vector2(u, v);
                }
            }

            return coords;
        }

        /// <summary>
        /// Generates checkerboard pattern UV coordinates.
        /// </summary>
        private Vector2[] GenerateCheckerboard()
        {
            // For checkerboard, we need twice the grid density and skip every other
            float aspectRatio = 4f / 3f;
            int rows = Mathf.CeilToInt(Mathf.Sqrt(sampleCount * 2 / aspectRatio));
            int cols = Mathf.CeilToInt(rows * aspectRatio);

            Vector2[] coords = new Vector2[sampleCount];
            int index = 0;

            for (int i = 0; i < rows && index < sampleCount; i++)
            {
                for (int j = 0; j < cols && index < sampleCount; j++)
                {
                    // Checkerboard: skip if (i + j) is odd
                    if ((i + j) % 2 == 0)
                    {
                        float u = (j + 0.5f) / cols;
                        float v = (i + 0.5f) / rows;
                        coords[index++] = new Vector2(u, v);
                    }
                }
            }

            // Fill remaining if needed
            int fillRow = 0, fillCol = 1;
            while (index < sampleCount)
            {
                if ((fillRow + fillCol) % 2 == 1)
                {
                    float u = (fillCol + 0.5f) / cols;
                    float v = (fillRow + 0.5f) / rows;
                    coords[index++] = new Vector2(u, v);
                }
                fillCol++;
                if (fillCol >= cols)
                {
                    fillCol = 0;
                    fillRow++;
                }
            }

            return coords;
        }

        /// <summary>
        /// Samples pixels from a RenderTexture at the specified UV coordinates.
        /// </summary>
        /// <param name="source">Source RenderTexture (grayscale R8)</param>
        /// <param name="uvCoords">UV coordinates to sample</param>
        /// <returns>Array of PixelData with sampled values</returns>
        public PixelData[] SamplePixels(RenderTexture source, UVCoordinates uvCoords)
        {
            if (source == null || uvCoords == null || uvCoords.Count == 0)
            {
                return Array.Empty<PixelData>();
            }

            // Ensure readback texture matches source resolution
            EnsureReadbackTexture(source.width, source.height);

            // Read back RenderTexture to CPU
            RenderTexture currentActive = RenderTexture.active;
            RenderTexture.active = source;
            readbackTexture.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            readbackTexture.Apply();
            RenderTexture.active = currentActive;

            // Sample pixels at UV coordinates
            PixelData[] result = new PixelData[uvCoords.Count];

            for (int i = 0; i < uvCoords.Count; i++)
            {
                Vector2 uv = uvCoords[i];

                int x = Mathf.Clamp(Mathf.FloorToInt(uv.x * source.width), 0, source.width - 1);
                int y = Mathf.Clamp(Mathf.FloorToInt(uv.y * source.height), 0, source.height - 1);

                Color pixel = readbackTexture.GetPixel(x, y);
                byte value = (byte)(pixel.r * 255f);

                result[i] = new PixelData(uv.x, uv.y, value);
            }

            return result;
        }

        /// <summary>
        /// Samples pixels using raw texture data for better performance.
        /// </summary>
        public PixelData[] SamplePixelsFast(RenderTexture source, UVCoordinates uvCoords)
        {
            if (source == null || uvCoords == null || uvCoords.Count == 0)
            {
                return Array.Empty<PixelData>();
            }

            EnsureReadbackTexture(source.width, source.height);

            // Read back to CPU
            RenderTexture currentActive = RenderTexture.active;
            RenderTexture.active = source;
            readbackTexture.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            readbackTexture.Apply();
            RenderTexture.active = currentActive;

            // Get raw data
            byte[] rawData = readbackTexture.GetRawTextureData();
            int width = source.width;
            int height = source.height;

            PixelData[] result = new PixelData[uvCoords.Count];

            for (int i = 0; i < uvCoords.Count; i++)
            {
                Vector2 uv = uvCoords[i];

                int x = Mathf.Clamp(Mathf.FloorToInt(uv.x * width), 0, width - 1);
                int y = Mathf.Clamp(Mathf.FloorToInt(uv.y * height), 0, height - 1);

                int index = y * width + x;
                byte value = (index < rawData.Length) ? rawData[index] : (byte)0;

                result[i] = new PixelData(uv.x, uv.y, value);
            }

            return result;
        }

        private void EnsureReadbackTexture(int width, int height)
        {
            if (readbackTexture == null ||
                lastResolution.x != width ||
                lastResolution.y != height)
            {
                if (readbackTexture != null)
                {
                    UnityEngine.Object.Destroy(readbackTexture);
                }

                readbackTexture = new Texture2D(width, height, TextureFormat.R8, false)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };

                lastResolution = new Vector2Int(width, height);
            }
        }

        /// <summary>
        /// Releases resources.
        /// </summary>
        public void Dispose()
        {
            if (readbackTexture != null)
            {
                UnityEngine.Object.Destroy(readbackTexture);
                readbackTexture = null;
            }
        }
    }
}
