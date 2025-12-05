using System;
using UnityEngine;

namespace SGAPS.Runtime.Data
{
    /// <summary>
    /// Collects and manages game state vector data.
    /// 
    /// The state vector provides game context to the server's reconstruction model.
    /// Values are reset after each frame is sent.
    /// 
    /// Uses a sentinel value (-999.0) to indicate unused dimensions,
    /// allowing variable-length state vectors while maintaining a fixed buffer size.
    /// </summary>
    public class StateVectorCollector
    {
        /// <summary>
        /// Default sentinel value for unused state dimensions.
        /// </summary>
        public const float DEFAULT_SENTINEL = -999.0f;

        private readonly float[] states;
        private readonly float sentinelValue;
        private readonly int maxDim;
        private int highestUsedIndex = -1;

        /// <summary>
        /// Maximum dimension of the state vector.
        /// </summary>
        public int MaxDimension => maxDim;

        /// <summary>
        /// Number of dimensions currently in use.
        /// </summary>
        public int UsedDimensions => highestUsedIndex + 1;

        /// <summary>
        /// Creates a new StateVectorCollector.
        /// </summary>
        /// <param name="maxStateDim">Maximum number of state dimensions</param>
        /// <param name="sentinelValue">Value to use for unused dimensions</param>
        public StateVectorCollector(int maxStateDim = 64, float sentinelValue = DEFAULT_SENTINEL)
        {
            if (maxStateDim <= 0)
            {
                throw new ArgumentException("maxStateDim must be positive", nameof(maxStateDim));
            }

            this.maxDim = maxStateDim;
            this.sentinelValue = sentinelValue;
            this.states = new float[maxStateDim];

            Reset();
        }

        /// <summary>
        /// Sets a value in the state vector at the specified index.
        /// </summary>
        /// <param name="index">State vector index (0 to maxDim-1)</param>
        /// <param name="value">Value to set</param>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public void SetState(int index, float value)
        {
            if (index < 0 || index >= maxDim)
            {
                Debug.LogWarning($"[StateVectorCollector] Index {index} out of range [0, {maxDim})");
                return;
            }

            states[index] = value;

            if (index > highestUsedIndex)
            {
                highestUsedIndex = index;
            }
        }

        /// <summary>
        /// Gets the value at the specified index.
        /// </summary>
        /// <param name="index">State vector index</param>
        /// <returns>State value, or sentinel if out of range</returns>
        public float GetState(int index)
        {
            if (index < 0 || index >= maxDim)
            {
                return sentinelValue;
            }
            return states[index];
        }

        /// <summary>
        /// Gets the used portion of the state vector (up to highestUsedIndex).
        /// </summary>
        /// <returns>Array containing only the used state values</returns>
        public float[] GetUsedStates()
        {
            if (highestUsedIndex < 0)
            {
                return Array.Empty<float>();
            }

            float[] result = new float[highestUsedIndex + 1];
            Array.Copy(states, result, highestUsedIndex + 1);
            return result;
        }

        /// <summary>
        /// Gets the full state vector including sentinel values.
        /// </summary>
        /// <returns>Copy of the full state array</returns>
        public float[] GetFullStates()
        {
            float[] result = new float[maxDim];
            Array.Copy(states, result, maxDim);
            return result;
        }

        /// <summary>
        /// Resets all state values to sentinel.
        /// Should be called after each frame is sent.
        /// </summary>
        public void Reset()
        {
            for (int i = 0; i < maxDim; i++)
            {
                states[i] = sentinelValue;
            }
            highestUsedIndex = -1;
        }

        /// <summary>
        /// Checks if a value at the given index is set (not sentinel).
        /// </summary>
        /// <param name="index">State vector index</param>
        /// <returns>True if the value is set</returns>
        public bool IsSet(int index)
        {
            if (index < 0 || index >= maxDim)
            {
                return false;
            }
            return !Mathf.Approximately(states[index], sentinelValue);
        }
    }
}
