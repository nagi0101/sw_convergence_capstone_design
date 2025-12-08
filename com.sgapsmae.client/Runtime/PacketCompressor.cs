using System;
using System.IO;
using System.IO.Compression;
using UnityEngine;

namespace SGAPSMAEClient
{
    /// <summary>
    /// Compresses pixel data for network transmission.
    /// </summary>
    public class PacketCompressor
    {
        private readonly int _compressionLevel;
        private byte[] _buffer;
        private MemoryStream _memoryStream;
        
        /// <summary>
        /// Create a new packet compressor.
        /// </summary>
        /// <param name="compressionLevel">Compression level (1-9)</param>
        public PacketCompressor(int compressionLevel = 1)
        {
            _compressionLevel = Mathf.Clamp(compressionLevel, 1, 9);
            _buffer = new byte[65536];
            _memoryStream = new MemoryStream();
        }
        
        /// <summary>
        /// Compress pixel data into a network packet.
        /// </summary>
        /// <param name="frameIdx">Frame index</param>
        /// <param name="coordinates">Pixel coordinates</param>
        /// <param name="pixelValues">RGB pixel values</param>
        /// <returns>Compressed packet bytes</returns>
        public byte[] Compress(int frameIdx, Vector2Int[] coordinates, Color32[] pixelValues)
        {
            _memoryStream.SetLength(0);
            _memoryStream.Position = 0;
            
            // Write header
            WriteInt32(_memoryStream, frameIdx);
            WriteInt32(_memoryStream, coordinates.Length);
            
            // Write pixel data
            for (int i = 0; i < coordinates.Length; i++)
            {
                // Coordinates as uint16
                WriteUInt16(_memoryStream, (ushort)coordinates[i].x);
                WriteUInt16(_memoryStream, (ushort)coordinates[i].y);
                
                // RGB as bytes
                _memoryStream.WriteByte(pixelValues[i].r);
                _memoryStream.WriteByte(pixelValues[i].g);
                _memoryStream.WriteByte(pixelValues[i].b);
            }
            
            // Compress
            byte[] uncompressed = _memoryStream.ToArray();
            return CompressBytes(uncompressed);
        }
        
        /// <summary>
        /// Decompress coordinate packet from server.
        /// </summary>
        /// <param name="data">Compressed packet bytes</param>
        /// <returns>Sampling coordinates</returns>
        public Vector2Int[] DecompressCoordinates(byte[] data)
        {
            byte[] decompressed = DecompressBytes(data);
            
            using (var stream = new MemoryStream(decompressed))
            {
                int numCoords = ReadInt32(stream);
                var coordinates = new Vector2Int[numCoords];
                
                for (int i = 0; i < numCoords; i++)
                {
                    int u = ReadUInt16(stream);
                    int v = ReadUInt16(stream);
                    coordinates[i] = new Vector2Int(u, v);
                }
                
                return coordinates;
            }
        }
        
        private byte[] CompressBytes(byte[] data)
        {
            using (var outputStream = new MemoryStream())
            {
                using (var deflateStream = new DeflateStream(outputStream, CompressionLevel.Fastest))
                {
                    deflateStream.Write(data, 0, data.Length);
                }
                return outputStream.ToArray();
            }
        }
        
        private byte[] DecompressBytes(byte[] data)
        {
            using (var inputStream = new MemoryStream(data))
            using (var deflateStream = new DeflateStream(inputStream, CompressionMode.Decompress))
            using (var outputStream = new MemoryStream())
            {
                deflateStream.CopyTo(outputStream);
                return outputStream.ToArray();
            }
        }
        
        private static void WriteInt32(Stream stream, int value)
        {
            stream.WriteByte((byte)(value & 0xFF));
            stream.WriteByte((byte)((value >> 8) & 0xFF));
            stream.WriteByte((byte)((value >> 16) & 0xFF));
            stream.WriteByte((byte)((value >> 24) & 0xFF));
        }
        
        private static void WriteUInt16(Stream stream, ushort value)
        {
            stream.WriteByte((byte)(value & 0xFF));
            stream.WriteByte((byte)((value >> 8) & 0xFF));
        }
        
        private static int ReadInt32(Stream stream)
        {
            int b0 = stream.ReadByte();
            int b1 = stream.ReadByte();
            int b2 = stream.ReadByte();
            int b3 = stream.ReadByte();
            return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        
        private static ushort ReadUInt16(Stream stream)
        {
            int b0 = stream.ReadByte();
            int b1 = stream.ReadByte();
            return (ushort)(b0 | (b1 << 8));
        }
    }
}
