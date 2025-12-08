using System;
using System.Collections.Generic;
using UnityEngine;

namespace SGAPSMAEClient
{
    /// <summary>
    /// Lightweight pixel extractor for game frames.
    /// Optimized for minimal CPU overhead.
    /// </summary>
    public class PixelExtractor
    {
        private readonly int _frameWidth;
        private readonly int _frameHeight;
        private readonly int _targetWidth;
        private readonly int _targetHeight;
        
        // Reusable buffers to avoid allocation
        private readonly List<Vector2Int> _validCoords;
        private readonly List<Color32> _pixelValues;
        
        /// <summary>
        /// Create a new pixel extractor.
        /// </summary>
        /// <param name="config">Client configuration</param>
        public PixelExtractor(ClientConfig config)
        {
            _frameWidth = config.frameWidth;
            _frameHeight = config.frameHeight;
            _targetWidth = config.targetWidth;
            _targetHeight = config.targetHeight;
            
            _validCoords = new List<Vector2Int>(1000);
            _pixelValues = new List<Color32>(1000);
        }
        
        /// <summary>
        /// Extract pixels from texture at specified coordinates.
        /// </summary>
        /// <param name="texture">Source texture</param>
        /// <param name="coordinates">Sampling coordinates in target resolution</param>
        /// <param name="validCoords">Output valid coordinates</param>
        /// <param name="pixelValues">Output pixel values</param>
        public void Extract(
            Texture2D texture, 
            Vector2Int[] coordinates,
            out Vector2Int[] validCoords,
            out Color32[] pixelValues)
        {
            _validCoords.Clear();
            _pixelValues.Clear();
            
            // Get raw pixel data
            var pixels = texture.GetPixels32();
            int texWidth = texture.width;
            int texHeight = texture.height;
            
            // Scale factors
            float scaleU = (float)texHeight / _targetHeight;
            float scaleV = (float)texWidth / _targetWidth;
            
            foreach (var coord in coordinates)
            {
                // Scale coordinate to texture space
                int u = Mathf.FloorToInt(coord.x * scaleU);
                int v = Mathf.FloorToInt(coord.y * scaleV);
                
                // Bounds check
                if (u >= 0 && u < texHeight && v >= 0 && v < texWidth)
                {
                    // Unity texture coordinates: bottom-left origin
                    int pixelIndex = (texHeight - 1 - u) * texWidth + v;
                    
                    if (pixelIndex >= 0 && pixelIndex < pixels.Length)
                    {
                        _validCoords.Add(coord);
                        _pixelValues.Add(pixels[pixelIndex]);
                    }
                }
            }
            
            validCoords = _validCoords.ToArray();
            pixelValues = _pixelValues.ToArray();
        }
        
        /// <summary>
        /// Fast extraction using NativeArray (for burst compilation).
        /// </summary>
        /// <param name="pixelData">Raw pixel data</param>
        /// <param name="textureWidth">Texture width</param>
        /// <param name="textureHeight">Texture height</param>
        /// <param name="coordinates">Sampling coordinates</param>
        /// <param name="validCoords">Output valid coordinates</param>
        /// <param name="pixelValues">Output pixel values</param>
        public void ExtractFast(
            Color32[] pixelData,
            int textureWidth,
            int textureHeight,
            Vector2Int[] coordinates,
            out Vector2Int[] validCoords,
            out Color32[] pixelValues)
        {
            _validCoords.Clear();
            _pixelValues.Clear();
            
            float scaleU = (float)textureHeight / _targetHeight;
            float scaleV = (float)textureWidth / _targetWidth;
            
            for (int i = 0; i < coordinates.Length; i++)
            {
                int u = (int)(coordinates[i].x * scaleU);
                int v = (int)(coordinates[i].y * scaleV);
                
                if (u >= 0 && u < textureHeight && v >= 0 && v < textureWidth)
                {
                    int pixelIndex = (textureHeight - 1 - u) * textureWidth + v;
                    
                    if (pixelIndex >= 0 && pixelIndex < pixelData.Length)
                    {
                        _validCoords.Add(coordinates[i]);
                        _pixelValues.Add(pixelData[pixelIndex]);
                    }
                }
            }
            
            validCoords = _validCoords.ToArray();
            pixelValues = _pixelValues.ToArray();
        }
    }
}
