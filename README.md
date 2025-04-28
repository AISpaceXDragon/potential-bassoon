# Real-Time Missing and New Object Detection from Video

## 🛠 Project Description

This project provides a **real-time video analytics pipeline** to:

- 🔎 **Detect missing objects**: Identify when an object present earlier in the scene disappears.
- 🆕 **Detect new objects**: Identify when a new object appears in the scene.

The solution is optimized for **high speed** and **high accuracy**, making it suitable for **real-world deployment scenarios** like surveillance, smart retail, or automated monitoring.

Inference is powered by a **YOLOv8 model** in half precision and accelerated on **GPU**.

---

## 🚀 Features

- **Real-Time Processing** on GPU
- **Model in half precision** Model for High FPS
- **Missing Object Detection** (compared to initial scene)
- **New Object Detection** (appearing objects)
- **Average FPS** Calculation and Display

---

## 📦 Installation

```bash
git clone https://github.com/your_username/real-time-missing-new-object-detection.git
cd real-time-missing-new-object-detection
pip install -r requirements.txt
```
---

## 📄 Usage

1. Place your input video file (e.g., `your_video.mp4`) in the project directory.
2. Run the detection script:

```bash
python detect_missing_new_objects.py --video your_video.mp4
```

By default, the script:

- Loads the video.
- Performs detection frame-by-frame.
- Displays frames only when a missing or new object is detected.
- Prints detailed logs in the console.
- Shows average FPS at the end.

---

## 🧠 Methodology

- **Initial Frame**: The objects detected in the first frame are saved as a **reference set**.
- **Each Frame**:
  - Detects objects using **YOLOv8n model**.
  - Compares current detections against the initial set:
    - **Missing Objects**: Classes that were in the initial frame but are absent now.
    - **New Objects**: Classes that appear now but were absent initially.
- **Performance Optimization**:
  - Uses **half precision** for lightweight inference.
  - **GPU-accelerated**.

---

## ⚙️ Configuration

You can adjust settings in `detect_missing_new_objects.py`:

| Parameter | Description | Default |
|:---|:---|:---|
| `--video` | Path to the input video file. If not provided, the script uses a default video file from this repository. | `'your_video.mp4'` |
| `--env` | Specifies whether the script is running in Google Colab or a local system. Choices: ["colab", "local"] | `'local'` |
| `--threshold` | The number of frames to wait before declaring an object as missing or newly detected. | `'30'` |
| `--device` | Specifies the device to run inference on. Choices: ["cuda", "cpu", "mps", "auto"] | `'auto'` |
| `--model` | Specifies the YOLO model to use. | `'yolov8n.pt'` |
| `--resolution` | The width of the frame for inference. | `'640'` |
| `--save_frames` | If enabled, saves frames when objects appear or disappear in a directory called detected_frames. | `'False'` |
| `--half_precision` | If enabled, uses half-precision (FP16) for model inference, which reduces inference time on CUDA-enabled devices. | `'True'` |

---

## 🤝 Acknowledgements

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
