# Walkable Path Detection and Audio Navigation System

A real-time semantic segmentation system that detects walkable areas from camera input and provides directional audio guidance for visually impaired users.

## Demo

![Real-Time Demo](assets/demo.gif)

## Overview

The system uses semantic segmentation to identify walkable regions in street scenes and converts them into simple navigation commands: **LEFT, RIGHT, FORWARD, or STOP**.

It combines a deep learning segmentation model with a custom walkability mapping and navigation algorithm.

## Pipeline

**Input Image → Semantic Segmentation → Walkability Map → Navigation → Audio Guidance**

![Pipeline](assets/pipeline.png)

## Dataset

The project uses the **Mapillary Vistas** dataset containing 25,000+ high-resolution street-level images with 150+ semantic classes.

The dataset annotations are converted into compact class IDs and used to train the segmentation model.

## Model

- **Architecture:** DeepLabv3 with ResNet-50 backbone
- **Framework:** PyTorch / TorchVision
- **Input:** 512 × 512 RGB images
- **Loss:** Cross Entropy
- **Optimizer:** Adam

## Walkability Mapping

Semantic classes are assigned a **walkability score between 0 and 1**.

For example:

- Sidewalk → 1.0
- Road → 0.7
- Grass → 0.3
- Obstacles such as vehicles, walls and poles → 0

This mapping can be changed without retraining the segmentation model.

## Navigation

The predicted walkable regions are converted into a binary mask. Perspective weighting gives greater importance to areas closer to the user.

A weighted centroid is then used to determine whether the user should move **LEFT, RIGHT, or FORWARD**.

A safety check on the immediate path can override the forward command and produce **STOP** when necessary.

## Results

![Walkability Overlay](assets/overlay.png)

The system produces a walkability overlay and real-time directional guidance from video input.

## Audio Guidance

The navigation commands are converted into speech using **pyttsx3**, allowing the system to provide offline audio instructions such as:

`LEFT` · `RIGHT` · `FORWARD` · `STOP`

## Tech Stack

Python · PyTorch · TorchVision · OpenCV · NumPy · Matplotlib · pyttsx3 · Mapillary Vistas

## Future Improvements

- Mobile deployment
- Depth information
- Haptic feedback
- Lightweight models for real-time deployment

## Authors

Shauvik Gogoi and team
