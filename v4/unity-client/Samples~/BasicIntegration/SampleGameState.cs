using UnityEngine;
using SGAPS.Runtime.Core;
using SGAPS.Runtime.Data;

namespace SGAPS.Samples
{
    /// <summary>
    /// Sample script showing how to integrate SGAPS with a game.
    /// 
    /// USAGE:
    /// 1. Attach this script to your player or game manager
    /// 2. Assign the SGAPSManager reference
    /// 3. Customize state indices for your game
    /// 
    /// STATE VECTOR INDICES (example):
    /// 0-2: Player position (X, Y, Z)
    /// 3-5: Player velocity (X, Y, Z)
    /// 6: Player health (normalized 0-1)
    /// 7: Current level
    /// 8: Time in level
    /// 9: Is grounded (0 or 1)
    /// </summary>
    public class SampleGameState : MonoBehaviour
    {
        [Header("SGAPS Reference")]
        [Tooltip("Reference to SGAPSManager in scene")]
        public SGAPSManager sgapsManager;

        [Header("Player References")]
        public Transform playerTransform;
        public Rigidbody playerRigidbody;

        [Header("Game State")]
        [Range(0, 100)]
        public float playerHealth = 100f;
        public float maxHealth = 100f;
        public int currentLevel = 1;

        // State vector indices
        private const int STATE_POS_X = 0;
        private const int STATE_POS_Y = 1;
        private const int STATE_POS_Z = 2;
        private const int STATE_VEL_X = 3;
        private const int STATE_VEL_Y = 4;
        private const int STATE_VEL_Z = 5;
        private const int STATE_HEALTH = 6;
        private const int STATE_LEVEL = 7;
        private const int STATE_TIME = 8;
        private const int STATE_GROUNDED = 9;

        private float levelStartTime;
        private CharacterController characterController;

        private void Start()
        {
            if (sgapsManager == null)
            {
                sgapsManager = FindObjectOfType<SGAPSManager>();
            }

            characterController = playerTransform?.GetComponent<CharacterController>();
            levelStartTime = Time.time;
        }

        private void LateUpdate()
        {
            if (sgapsManager == null) return;

            // Player position
            if (playerTransform != null)
            {
                Vector3 pos = playerTransform.position;
                sgapsManager.SetState(STATE_POS_X, pos.x);
                sgapsManager.SetState(STATE_POS_Y, pos.y);
                sgapsManager.SetState(STATE_POS_Z, pos.z);
            }

            // Player velocity
            if (playerRigidbody != null)
            {
                Vector3 vel = playerRigidbody.velocity;
                sgapsManager.SetState(STATE_VEL_X, vel.x);
                sgapsManager.SetState(STATE_VEL_Y, vel.y);
                sgapsManager.SetState(STATE_VEL_Z, vel.z);
            }

            // Health (normalized)
            sgapsManager.SetState(STATE_HEALTH, playerHealth / maxHealth);

            // Level info
            sgapsManager.SetState(STATE_LEVEL, currentLevel);
            sgapsManager.SetState(STATE_TIME, Time.time - levelStartTime);

            // Grounded
            bool grounded = characterController != null && characterController.isGrounded;
            sgapsManager.SetState(STATE_GROUNDED, grounded ? 1f : 0f);
        }

        public void SetHealth(float health)
        {
            playerHealth = Mathf.Clamp(health, 0, maxHealth);
        }

        public void OnLevelStart(int level)
        {
            currentLevel = level;
            levelStartTime = Time.time;
        }
    }
}
