using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace SGAPS.Runtime.Utilities
{
    /// <summary>
    /// Utility class for performance monitoring.
    /// Tracks FPS, capture times, network latency with rolling averages.
    /// </summary>
    public class PerformanceMonitor
    {
        #region Singleton

        private static PerformanceMonitor instance;

        public static PerformanceMonitor Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new PerformanceMonitor();
                }
                return instance;
            }
        }

        #endregion

        public int RollingWindowSize { get; set; } = 60;

        private readonly Dictionary<string, Queue<double>> timingSamples = new Dictionary<string, Queue<double>>();
        private readonly Dictionary<string, Stopwatch> activeStopwatches = new Dictionary<string, Stopwatch>();

        private readonly Queue<float> fpsHistory = new Queue<float>();
        private float lastFpsUpdateTime;
        private int frameCountSinceLastUpdate;

        private readonly Queue<double> networkLatencySamples = new Queue<double>();
        private readonly Dictionary<long, DateTime> pendingRequests = new Dictionary<long, DateTime>();
        private long requestIdCounter = 0;

        public PerformanceMonitor()
        {
            lastFpsUpdateTime = Time.realtimeSinceStartup;
        }

        /// <summary>Begins measuring time for a named sample.</summary>
        public void BeginSample(string sampleName)
        {
            if (!activeStopwatches.ContainsKey(sampleName))
            {
                activeStopwatches[sampleName] = new Stopwatch();
            }
            activeStopwatches[sampleName].Restart();
        }

        /// <summary>Ends measurement and records elapsed time.</summary>
        public double EndSample(string sampleName)
        {
            if (!activeStopwatches.TryGetValue(sampleName, out var stopwatch))
            {
                Debug.LogWarning($"[PerformanceMonitor] No BeginSample for: {sampleName}");
                return -1;
            }

            stopwatch.Stop();
            double elapsedMs = stopwatch.Elapsed.TotalMilliseconds;
            RecordSample(sampleName, elapsedMs);
            return elapsedMs;
        }

        /// <summary>Records a timing sample directly.</summary>
        public void RecordSample(string sampleName, double timeMs)
        {
            if (!timingSamples.ContainsKey(sampleName))
            {
                timingSamples[sampleName] = new Queue<double>();
            }

            var samples = timingSamples[sampleName];
            samples.Enqueue(timeMs);

            while (samples.Count > RollingWindowSize)
            {
                samples.Dequeue();
            }
        }

        /// <summary>Updates FPS calculation. Call once per frame.</summary>
        public void UpdateFPS()
        {
            frameCountSinceLastUpdate++;

            float currentTime = Time.realtimeSinceStartup;
            float elapsed = currentTime - lastFpsUpdateTime;

            if (elapsed >= 0.5f)
            {
                float fps = frameCountSinceLastUpdate / elapsed;

                fpsHistory.Enqueue(fps);
                while (fpsHistory.Count > RollingWindowSize)
                {
                    fpsHistory.Dequeue();
                }

                frameCountSinceLastUpdate = 0;
                lastFpsUpdateTime = currentTime;
            }
        }

        public float GetCurrentFPS() => 1.0f / Time.deltaTime;

        public float GetAverageFPS()
        {
            if (fpsHistory.Count == 0) return 0;

            float sum = 0;
            foreach (var fps in fpsHistory)
            {
                sum += fps;
            }
            return sum / fpsHistory.Count;
        }

        /// <summary>Marks a network request sent for latency tracking.</summary>
        public long MarkRequestSent()
        {
            long requestId = requestIdCounter++;
            pendingRequests[requestId] = DateTime.UtcNow;
            return requestId;
        }

        /// <summary>Marks request complete and records latency.</summary>
        public double MarkRequestComplete(long requestId)
        {
            if (!pendingRequests.TryGetValue(requestId, out var startTime))
            {
                return -1;
            }

            pendingRequests.Remove(requestId);
            double latencyMs = (DateTime.UtcNow - startTime).TotalMilliseconds;

            networkLatencySamples.Enqueue(latencyMs);
            while (networkLatencySamples.Count > RollingWindowSize)
            {
                networkLatencySamples.Dequeue();
            }

            return latencyMs;
        }

        public double GetAverageNetworkLatency()
        {
            if (networkLatencySamples.Count == 0) return 0;

            double sum = 0;
            foreach (var latency in networkLatencySamples)
            {
                sum += latency;
            }
            return sum / networkLatencySamples.Count;
        }

        /// <summary>Gets the rolling average for a named sample.</summary>
        public double GetAverage(string sampleName)
        {
            if (!timingSamples.TryGetValue(sampleName, out var samples) || samples.Count == 0)
            {
                return 0;
            }

            double sum = 0;
            foreach (var sample in samples)
            {
                sum += sample;
            }
            return sum / samples.Count;
        }

        public void Clear()
        {
            timingSamples.Clear();
            activeStopwatches.Clear();
            fpsHistory.Clear();
            networkLatencySamples.Clear();
            pendingRequests.Clear();
        }

        public void PrintReport()
        {
            Debug.Log("=== SGAPS Performance Report ===");
            Debug.Log($"Average FPS: {GetAverageFPS():F1}");
            Debug.Log($"Network Latency: {GetAverageNetworkLatency():F2}ms");

            foreach (var sampleName in timingSamples.Keys)
            {
                Debug.Log($"{sampleName}: Avg={GetAverage(sampleName):F2}ms");
            }
        }
    }
}
