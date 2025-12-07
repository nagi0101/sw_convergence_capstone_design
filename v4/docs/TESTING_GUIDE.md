# SGAPS Phase 1 Finalization - Testing Guide

**Version:** 1.0  
**Date:** December 8, 2025

This document provides instructions on how to test the features and improvements implemented to finalize Phase 1 of the SGAPS project.

---

## 1. Server-Side Testing

### 1.1. Verifying `scikit-image` Dependency

The SSIM (Structural Similarity Index) metric calculation has been upgraded to use the industry-standard `scikit-image` library for better accuracy and performance.

**▶ How to Test:**

1.  Navigate to the server directory:
    ```bash
    cd sgaps-server
    ```

2.  Install/update the dependencies from the updated `requirements.txt` file. It's recommended to do this in a Python virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  Verify that `scikit-image` is installed correctly by running:
    ```bash
    pip show scikit-image
    ```
    This should display the package information, confirming it's part of the environment. The server will now automatically use this improved metric when calculating reconstruction quality.

### 1.2. Verifying `load_checkpoint` Warning

A stub function for `load_checkpoint` was added to `OpenCVReconstructor`, and the server logic has been updated to call this function when a new session starts. This makes the code's behavior clearer for future (Phase 2) integration.

**▶ How to Test:**

1.  Navigate to the server directory:
    ```bash
    cd sgaps-server
    ```

2.  Run the SGAPS server:
    ```bash
    python main.py
    ```

3.  In Unity, enter Play Mode and connect to the server using the `SGAPSManager` component.

4.  When the client connects and a session is created, you should see the following warning message in the server's console output. No code modification is needed.

    ```
    WARNING:sgaps.core.reconstructor:Attempted to load checkpoint 'your_checkpoint_key', but model loading is a Phase 2+ feature and is not yet implemented. Falling back to baseline reconstructor.
    ```
    ('your_checkpoint_key' will match what is set in the `SGAPSManager` component in Unity). This confirms the function is being called correctly.

---

## 2. Client-Side Testing (Unity)

### 2.1. Testing the Enhanced `SGAPSManager` Inspector

The custom editor for the `SGAPSManager` component has been significantly enhanced to provide more detailed runtime feedback directly in the Unity Inspector. This replaces the need for a separate Canvas-based debug UI.

**▶ How to Test:**

1.  Open your Unity project containing the SGAPS client package.
2.  In any scene, select a GameObject that has the `SGAPSManager` component attached.
3.  Observe the Inspector window for this GameObject. You should see the custom UI.

4.  **In Edit Mode:**
    *   The "Connection Controls" section will be visible but the buttons will be disabled.
    *   A help box will state "Enter Play Mode to connect to server."
    *   The "Runtime Statistics" foldout will show basic information but most values will be default/zero.

5.  **In Play Mode:**
    *   Start the SGAPS server if it's not already running.
    *   Enter Play Mode in Unity.
    *   The **Connect** button is now enabled. Click it to establish a connection.
    *   Expand the "Runtime Statistics" foldout if it's not already open.
    *   Observe the detailed runtime statistics:
        *   **Status:** Will change to `● Connected` (green) when connected, and `● Disconnected` (red) otherwise.
        *   **Endpoint:** Displays the configured server endpoint.
        *   **Resolution:** Shows the current screen resolution.
        *   **Target FPS:** Indicates the FPS configured by the server.
        *   **Current FPS:** Displays the actual frames per second the Unity application is rendering.
        *   **Frames Sent:** Shows the total number of frames successfully sent to the server.
        *   **Sampled Pixels:** The number of pixels currently being sampled per frame.
        *   **Last Capture:** The time taken for the last frame capture and processing.
        *   **Avg Capture:** The average time taken for frame capture and processing.
        *   **Error:** If any error occurs during the session, it will be displayed here in red.
    *   Click the **Disconnect** button. The status should change back to `● Disconnected`, and frame-related statistics will cease updating.
    *   Verify that all fields are displaying relevant and updating information as expected.
