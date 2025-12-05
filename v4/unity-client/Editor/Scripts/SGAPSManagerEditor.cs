using UnityEngine;
using UnityEditor;
using SGAPS.Runtime.Core;

namespace SGAPS.Editor
{
    /// <summary>
    /// Custom inspector for SGAPSManager component.
    /// </summary>
    [CustomEditor(typeof(SGAPSManager))]
    public class SGAPSManagerEditor : UnityEditor.Editor
    {
        private SGAPSManager manager;
        private bool showRuntimeStats = true;

        private void OnEnable()
        {
            manager = (SGAPSManager)target;
        }

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space(10);
            DrawConnectionControls();

            if (Application.isPlaying)
            {
                EditorGUILayout.Space(10);
                DrawRuntimeStats();
            }

            if (Application.isPlaying)
            {
                Repaint();
            }
        }

        private void DrawConnectionControls()
        {
            EditorGUILayout.LabelField("Connection Controls", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();

            EditorGUI.BeginDisabledGroup(!Application.isPlaying || manager.IsConnected);
            if (GUILayout.Button("Connect", GUILayout.Height(30)))
            {
                manager.ConnectToServer();
            }
            EditorGUI.EndDisabledGroup();

            EditorGUI.BeginDisabledGroup(!Application.isPlaying || !manager.IsConnected);
            if (GUILayout.Button("Disconnect", GUILayout.Height(30)))
            {
                manager.DisconnectFromServer();
            }
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.EndHorizontal();

            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox("Enter Play Mode to connect to server.", MessageType.Info);
            }
        }

        private void DrawRuntimeStats()
        {
            showRuntimeStats = EditorGUILayout.Foldout(showRuntimeStats, "Runtime Statistics", true);

            if (!showRuntimeStats) return;

            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            // Connection status
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Status:", GUILayout.Width(120));

            GUIStyle statusStyle = new GUIStyle(EditorStyles.label);
            if (manager.IsConnected)
            {
                statusStyle.normal.textColor = new Color(0.2f, 0.8f, 0.2f);
                EditorGUILayout.LabelField("● Connected", statusStyle);
            }
            else
            {
                statusStyle.normal.textColor = new Color(0.8f, 0.2f, 0.2f);
                EditorGUILayout.LabelField("● Disconnected", statusStyle);
            }
            EditorGUILayout.EndHorizontal();

            // Frame count
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Frames:", GUILayout.Width(120));
            EditorGUILayout.LabelField(manager.FrameCount.ToString("N0"));
            EditorGUILayout.EndHorizontal();

            // FPS
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("FPS:", GUILayout.Width(120));
            EditorGUILayout.LabelField($"{(1.0f / Time.deltaTime):F1}");
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();
        }
    }
}
