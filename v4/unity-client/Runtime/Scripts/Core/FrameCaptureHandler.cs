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
        private Material grayscaleMaterial;

        private bool isDisposed = false;
        private Vector2Int lastScreenSize;

        /// <summary>
        /// Gets the current grayscale RenderTexture after capture.
        /// </summary>
        public RenderTexture GrayscaleTexture => grayscaleRT;

        /// <summary>
        /// Current screen resolution.
        /// </summary>
        public Vector2Int ScreenResolution => new Vector2Int(Screen.width, Screen.height);

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
            }

            if (screenRT == null)
            {
                screenRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32)
                {
                    name = "SGAPS_ScreenRT",
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                screenRT.Create();

                grayscaleRT = new RenderTexture(width, height, 0, RenderTextureFormat.R8)
                {
                    name = "SGAPS_GrayscaleRT",
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp
                };
                grayscaleRT.Create();

                lastScreenSize = new Vector2Int(width, height);
                Debug.Log($"[SGAPS.FrameCaptureHandler] Created RenderTextures at {width}x{height}");
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
        /// Captures the final rendered screen to a grayscale RenderTexture.
        /// MUST be called after WaitForEndOfFrame in a coroutine.
        /// Includes UI, post-processing, and all camera compositing.
        /// </summary>
        /// <returns>The grayscale RenderTexture of the captured screen</returns>
        public RenderTexture CaptureScreen()
        {
            if (isDisposed)
            {
                throw new ObjectDisposedException(nameof(FrameCaptureHandler));
            }

            EnsureRenderTextures();

            // Capture final screen (UI, post-processing, everything)
            ScreenCapture.CaptureScreenshotIntoRenderTexture(screenRT);

            // Convert to grayscale
            Graphics.Blit(screenRT, grayscaleRT, grayscaleMaterial);

            return grayscaleRT;
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

            if (grayscaleMaterial != null)
            {
                UnityEngine.Object.Destroy(grayscaleMaterial);
                grayscaleMaterial = null;
            }

            Debug.Log("[SGAPS.FrameCaptureHandler] Disposed");
        }
    }
}
