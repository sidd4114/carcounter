# Vehicle Detection and Counting System

This is a real-time vehicle detection and counting system using YOLOv8, OpenCV, and the SORT tracking algorithm. It detects and counts vehicles as they pass through a defined region of interest in a video or webcam feed.

## About

To explore my interest in computer vision, I built this project by referring to a YouTube tutorial. I wanted to understand how detection and tracking work together in real-time applications. While the code is not fully original, it helped me understand core concepts like object detection, region masking, and object tracking.

## Features

- Real-time vehicle detection using YOLOv8
- Vehicle tracking using the SORT algorithm
- Region-of-interest mask for focused detection
- Object ID persistence and counting logic
- Works with both webcam and video input

## Tech Stack

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- cvzone
- SORT (Simple Online Realtime Tracker)

## Installation

1. Clone the repository
```bash
git clone https://github.com/sidd4114/carcounter.git
cd carcounter
