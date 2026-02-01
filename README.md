# 🎭 Final Robust Video Face Swapper

**Advanced Automated Face Swap Pipeline**

## 🧠 Overview
A robust Python application utilizing **OpenCV** and Deep Learning models to perform seamless face swapping in video feeds. This project focuses on frame-by-frame stability and blending accuracy.

## ⚡ Key Capabilities
*   **Source Independence:** Takes a `source_face.jpg` and applies it to target video.
*   **Robustness:** Handles occlusion and lighting variations better than standard scripts.
*   **Pipeline:**
    1.  Face Detection (Landmark alignment).
    2.  Mask Generation.
    3.  Seamless Cloning/Blending.

## 🛠️ Requirements
*   Python 3.8+
*   OpenCV (`cv2`)
*   NumPy