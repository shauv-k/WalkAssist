# WalkAssist

### Semantic Segmentation for Assistive Navigation

WalkAssist is a real-time computer vision system that detects walkable regions in outdoor environments and provides directional guidance for visually impaired users.

The system combines semantic segmentation, custom walkability scoring, perspective-aware processing, centroid-based navigation, safety checks, and offline text-to-speech to convert visual scene understanding into actionable navigation instructions.

---

## Demo

### Input → Segmentation → Walkability → Navigation

<p align="center">
  <img src="assets/input.png" width="32%" alt="Input street scene">
  <img src="assets/semantic-mask.png" width="32%" alt="Semantic segmentation mask">
  <img src="assets/binary-mask.png" width="32%" alt="Binary walkability mask">
</p>

<p align="center">
  <img src="assets/walkability-overlay.png" width="48%" alt="Walkability overlay highlighting detected walkable regions">
  <img src="assets/navigation-output.png" width="48%" alt="Final navigation output with directional guidance">
</p>

---

## Overview

Traditional navigation systems such as GPS provide global directions but do not directly understand the immediate environment around a pedestrian.

WalkAssist uses semantic segmentation to understand the scene at the pixel level and converts the detected semantic classes into walkability information.

The system then determines a suitable walking direction and provides audio commands such as:

- `LEFT`
- `RIGHT`
- `FORWARD`
- `STOP`

---

## How It Works

```text
Input Frame
     │
     ▼
Semantic Segmentation
     │
     ▼
Walkability Mapping
     │
     ▼
Binary Walkability Mask
     │
     ▼
Perspective Weighting
     │
     ▼
Centroid-Based Direction Estimation
     │
     ▼
Safety Strip Validation
     │
     ▼
Directional Audio Guidance
