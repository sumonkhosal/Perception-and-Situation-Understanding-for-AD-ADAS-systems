# Perception and Situation Understanding for AD/ADAS Systems

A real-time monocular perception pipeline for road-scene video analysis using **Depth Anything V2**, **YOLOv8**, **Kalman-based tracking**, **bird’s-eye-view (BEV) lane detection**, and a transparent **HUD visualization**.

This project processes driving videos and generates multiple annotated outputs, including:

- an **annotated scene video**
- an **RGB + depth side-by-side video**
- a **bird’s-eye-view (BEV) lane visualization**
- tracked **pedestrians**, **vehicles**, **traffic signs**, and **traffic lights**
- depth-aware **distance reasoning** and **motion intent estimation**

The implementation is optimized for **Windows + local Python + VS Code**, with special care taken to ensure stable video writing and practical real-time performance.

---

## Project Overview

The goal of this project is to build a lightweight **Advanced Driver Assistance System (ADAS)** style perception pipeline from a single RGB video source.

The system combines:

- **monocular depth estimation** for scene understanding
- **object detection** for road users and signs
- **tracking** for temporal consistency
- **lane detection in BEV space**
- **HUD-based scene summarization**
- **multi-video export** for debugging and evaluation

### Main Features

- **Depth Anything V2** for dense monocular depth estimation
- **YOLOv8** for object detection
- **Kalman filter + IoU tracking** for stable object tracking
- **Traffic light state estimation** using HSV-based color analysis
- **Pedestrian intent estimation** based on position and motion history
- **Vehicle approach/motion reasoning** using depth history
- **Bird’s-eye-view lane detection** using:
  - perspective transform
  - edge detection
  - Hough line proposals
  - sliding window search
  - polynomial lane fitting
- **Transparent HUD overlay** for live scene summaries
- **Windows-safe AVI output** with optional MP4 export

---

## Repository Structure

A recommended repository structure is shown below:

```text
Perception-and-Situation-Understanding-for-AD-ADAS-systems/
│
├── AI_IN_PROPULSION.py
├── README.md
├── requirements.txt
├── LICENSE
│
├── videos/
│   └── Prank_Video.mp4
│
├── models/
│   ├── depth_anything_v2_vits.pth
│   └── traffic_sign_yolov8.pt
│
├── Depth-Anything-V2/
│   └── depth_anything_v2/
│
└── output/
