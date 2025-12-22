using System;
using UnityEngine;

namespace SGAPS.Runtime.Core
{
    /// <summary>
    /// Handles frame capture of the final rendered screen including UI and post-processing.
    /// Uses ScreenCapture API to capture exactly what the user sees.
    /// Must be called after WaitForEndOfFrame in a coroutine.
    /// </summary>
    public class FrameCaptureHandler : IDisposable
    {
        private RenderTexture screenRT;
        private RenderTexture grayscaleRT;
        private RenderTexture debugRT; // For downsampled debug frame
        private Material grayscaleMaterial;

        private bool isDisposed = false;
        private Vector2Int lastScreenSize;
        private readonly Vector2Int debugTextureSize = new Vector2Int(112, 112);

        /// <summary>
        /// Gets the current grayscale RenderTexture after capture.
        /// </summary>
        public RenderTexture GrayscaleTexture => grayscaleRT;

        /// <summary>
        /// Current screen resolution.
        /// </summary>
        public Vector2Int ScreenResolution => new Vector2Int(Screen.width, Screen.height);

        /// <summary>
        /// The resolution of the downsampled debug texture.
        /// </summary>
        public Vector2Int DebugTextureResolution => debugTextureSize;

        /// <summary>
        /// Creates a new FrameCaptureHandler for final screen capture.
        /// </summary>
        public FrameCaptureHandler()
        {
            InitializeMaterial();
            EnsureRenderTextures();
        }

        private void EnsureRenderTextures()
        {
            int width = Screen.width;
            int height = Screen.height;

            // Recreate if screen size changed
            if (screenRT != null && (lastScreenSize.x != width || lastScreenSize.y != height))
            {
                screenRT.Release();
                UnityEngine.Object.Destroy(screenRT);
                screenRT = null;

                if (grayscaleRT != null)
                {
                    grayscaleRT.Release();
                    UnityEngine.Object.Destroy(grayscaleRT);
                    grayscaleRT = null;
                }

                // Debug RT does not depend on screen size, but we can recreate it here for cleanliness
                if (debugRT != null)
                {
                    debugRT.Release();
                    UnityEngine.Object.Destroy(debugRT);
                    debugRT = null;
                }
            }

            if (screenRT == null)
            {
                // Explicitly use Linear read/write to prevent implicit sRGB conversions by Unity
                screenRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
                {
                    name = "SGAPS_ScreenRT",
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                screenRT.Create();

                grayscaleRT = new RenderTexture(width, height, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear)
                {
                    name = "SGAPS_GrayscaleRT",
                    filterMode = FilterMode.Bilinear, // Bilinear is better for downsampling
                    wrapMode = TextureWrapMode.Clamp
                };
                grayscaleRT.Create();

                debugRT = new RenderTexture(debugTextureSize.x, debugTextureSize.y, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear)
                {
                    name = "SGAPS_DebugRT",
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };
                debugRT.Create();

                lastScreenSize = new Vector2Int(width, height);
                Debug.Log($"[SGAPS.FrameCaptureHandler] Created RenderTextures (Linear). Screen: {width}x{height}, Debug: {debugTextureSize.x}x{debugTextureSize.y}");
            }
        }

        private void InitializeMaterial()
        {
            Shader shader = Shader.Find("Hidden/SGAPS/GrayscaleConvert");

            if (shader == null)
            {
                Debug.LogError("[SGAPS.FrameCaptureHandler] GrayscaleConvert shader not found!");
                shader = Shader.Find("Unlit/Texture");
            }

            grayscaleMaterial = new Material(shader)
            {
                hideFlags = HideFlags.HideAndDontSave
            };
        }

        /// <summary>
        /// Captures the final rendered screen to a full-resolution grayscale RenderTexture.
        /// </summary>
        public RenderTexture CaptureScreen()
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(FrameCaptureHandler));
            }

            EnsureRenderTextures();

            ScreenCapture.CaptureScreenshotIntoRenderTexture(screenRT);

            Graphics.Blit(screenRT, grayscaleRT, grayscaleMaterial);

            return grayscaleRT;
        }

        /// <summary>
        /// Downsamples the current grayscale texture for debug frame transmission.
        /// Assumes CaptureScreen() has already been called for this frame.
        /// </summary>
        /// <returns>A low-resolution grayscale RenderTexture of the captured screen.</returns>
        public RenderTexture CaptureDebugFrameTexture()
        {
            // Do NOT call CaptureScreen() here. It causes artifacts (Red Channel Bug) and is redundant.
            // SGAPSManager calls CaptureScreen() before calling this.

            // Downsample the existing grayscaleRT to the debug texture
            Graphics.Blit(grayscaleRT, debugRT);

            return debugRT;
        }

        /// <summary>
        /// Releases all resources.
        /// </summary>
        public void Dispose()
        {
            if (isDisposed) return;
            isDisposed = true;

            if (screenRT != null)
            {
                screenRT.Release();
                UnityEngine.Object.Destroy(screenRT);
                screenRT = null;
            }

            if (grayscaleRT != null)
            {
                grayscaleRT.Release();
                UnityEngine.Object.Destroy(grayscaleRT);
                grayscaleRT = null;
            }

            if (debugRT != null)
            {
                debugRT.Release();
                UnityEngine.Object.Destroy(debugRT);
                debugRT = null;
            }

            if (grayscaleMaterial != null)
            {
                UnityEngine.Object.Destroy(grayscaleMaterial);
                grayscaleMaterial = null;
            }

            Debug.Log("[SGAPS.FrameCaptureHandler] Disposed");
        }
    }
}
