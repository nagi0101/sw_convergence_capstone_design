using UnityEditor;
using UnityEngine;

namespace SGAPSMAEClient.Editor
{
    /// <summary>
    /// Custom editor for SGAPSMAEGameClient component.
    /// </summary>
    [CustomEditor(typeof(SGAPSMAEGameClient))]
    public class SGAPSMAEGameClientEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            
            var client = (SGAPSMAEGameClient)target;
            
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Actions", EditorStyles.boldLabel);
            
            EditorGUI.BeginDisabledGroup(!Application.isPlaying);
            
            EditorGUILayout.BeginHorizontal();
            
            if (!client.IsRecording)
            {
                if (GUILayout.Button("Start Recording"))
                {
                    // Generate initial coordinates
                    var coords = new Vector2Int[500];
                    for (int i = 0; i < coords.Length; i++)
                    {
                        coords[i] = new Vector2Int(
                            Random.Range(0, 224),
                            Random.Range(0, 224)
                        );
                    }
                    client.SetInitialCoordinates(coords);
                    client.StartRecording();
                }
            }
            else
            {
                if (GUILayout.Button("Stop Recording"))
                {
                    client.StopRecording();
                }
            }
            
            if (GUILayout.Button("Reset"))
            {
                client.Reset();
            }
            
            EditorGUILayout.EndHorizontal();
            
            EditorGUI.EndDisabledGroup();
            
            // Statistics
            if (Application.isPlaying)
            {
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("Statistics", EditorStyles.boldLabel);
                
                var stats = client.GetStatistics();
                EditorGUILayout.LabelField($"Frames: {stats.FrameCount}");
                EditorGUILayout.LabelField($"Avg Time: {stats.AverageProcessingTimeMs:F2} ms");
                EditorGUILayout.LabelField($"Buffer: {stats.CoordinateBufferCount}");
                
                // Repaint for live updates
                Repaint();
            }
        }
    }
}
