# Perception-and-Situation-Understanding-for-AD-ADAS-systems

A real-time monocular perception pipeline for road-scene video analysis using **Depth Anything V2**, **YOLOv8**, **Kalman-based tracking**, **bird’s-eye-view lane detection**, and a transparent **HUD visualization**.

This project processes a driving video and generates multiple annotated outputs, including:

- an **annotated scene video**
- an **RGB + depth side-by-side video**
- a **bird’s-eye-view (BEV) lane visualization**
- tracked pedestrians, vehicles, traffic signs, and traffic lights
- depth-aware distance reasoning and motion intent estimation

The implementation is optimized for **Windows + local Python + VS Code**, with special care taken to ensure stable video writing and practical real-time performance.

---

## Overview

The goal of this project is to build a lightweight Advanced Driver Assistance System (ADAS)-style perception pipeline from a single RGB video source.

The system combines:

- **monocular depth estimation** for scene understanding
- **object detection** for road users and signs
- **tracking** for temporal consistency
- **lane detection in BEV space**
- **HUD-based scene summarization**
- **multi-video export for debugging and evaluation**

This makes the project useful as a prototype for:

- perception research
- ADAS concept demonstrations
- computer vision portfolios
- academic projects
- rapid testing of depth + detection fusion

---

## Key Features

### 1. Monocular Depth Estimation
Uses **Depth Anything V2** to infer dense depth from a single RGB frame.

- stable depth colormap generation
- percentile-based normalization
- automatic depth inversion detection
- smoothed and locked inversion after warmup
- road-region-only depth thresholding for improved consistency

### 2. Object Detection with YOLOv8
Uses **YOLOv8** for detecting:

- pedestrians
- cars
- trucks
- buses
- motorcycles
- traffic lights

A second custom YOLO model is used for:

- traffic sign detection

### 3. Tracking
Includes lightweight tracking using:

- **Kalman filter motion prediction**
- **IoU-based association**
- separate trackers for:
  - pedestrians
  - vehicles
  - traffic signs

This allows more stable labels and temporal reasoning across frames.

### 4. Pedestrian Reasoning
Pedestrians are analyzed using:

- side classification (`LEFT`, `FRONT`, `RIGHT`)
- depth class (`FAR`, `MID`, `CLOSE`)
- intent estimation:
  - `STRAIGHT`
  - `CROSS_L2R`
  - `CROSS_R2L`
  - `UNKNOWN`

### 5. Vehicle Motion Reasoning
Vehicles are tracked and assigned simple approach behavior using depth history:

- `APPROACHING`
- `MOVING_AWAY`
- `STABLE`
- `UNKNOWN`

### 6. Traffic Light State Estimation
Traffic lights are classified directly from the cropped image region using HSV color analysis:

- `RED`
- `YELLOW`
- `GREEN`
- `UNKNOWN`

### 7. Bird’s-Eye View Lane Detection
Lane detection is performed in warped top-view space using:

- perspective transform
- edge detection
- Hough line proposals
- sliding window search
- polynomial lane fitting

The detected lane is drawn both in:

- the **BEV output**
- the **main annotated frame**

### 8. Transparent HUD
A transparent black HUD overlays scene statistics without hiding the video, including:

- compared detections count
- grid vs depth mismatch statistics
- current depth inversion state
- lane-detection status
- tracked object summaries

### 9. Windows-Safe Video Output
To avoid common OpenCV/MP4 issues on Windows:

- AVI with **MJPG** is always written reliably
- MP4 export is attempted as optional best-effort output

---

## Pipeline Structure

The processing flow is:

1. Read input video
2. Resize frame to processing resolution
3. Segment road-like region
4. Estimate dense depth
5. Run YOLO object detection periodically
6. Run traffic sign detection periodically
7. Track pedestrians, vehicles, and signs
8. Estimate:
   - pedestrian intent
   - vehicle approach behavior
   - traffic light state
9. Generate BEV transform
10. Detect lanes in BEV space
11. Warp lane overlay back to original processed frame
12. Draw transparent HUD
13. Export output videos

---

## Important Improvements Included

This version contains several practical fixes and upgrades:

### Grid vs Depth Mismatch Fix
Two important improvements were introduced:

- depth thresholds (`p30`, `p60`) are computed only on the **road region**
- object depth is sampled from the lower **feet band** of the bounding box instead of the full box

This makes depth-distance reasoning more stable for pedestrians and vehicles.

### Transparent HUD
The HUD background is rendered using transparent black so that the original frame remains visible.

### Lane Detection Added
A full lane pipeline has been integrated using:

- BEV transform
- Hough proposals
- sliding window fitting
- overlay on both BEV and main output

### Windows Video Writing Reliability
AVI output is enforced as the main output format to avoid the common “top-left only” or incomplete MP4-writing issue on Windows systems.

---

## Example Outputs

The script generates three main outputs:

### 1. Annotated Video
Shows:

- detected pedestrians
- tracked vehicles
- traffic signs
- traffic light states
- lane overlay
- HUD summary

### 2. RGB + Depth Video
A side-by-side comparison of:

- original RGB frame
- depth colormap frame

Useful for debugging depth behavior.

### 3. BEV Video
Shows:

- bird’s-eye transformed road scene
- lane detection overlay

---

## Project Structure

A typical local project structure is expected as follows:

```text
C:\AI_Project
│
├── videos
│   └── Prank_Video.mp4
│
├── models
│   ├── depth_anything_v2_vits.pth
│   └── traffic_sign_yolov8.pt
│
├── Depth-Anything-V2
│   └── depth_anything_v2
│
└── output
