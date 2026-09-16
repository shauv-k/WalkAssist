# Walkable Path Detection and Audio Navigation System

## 1. Project Overview

The Walkable Path Detection and Audio Navigation System is a real-time camera-based navigation system designed to identify walkable regions in outdoor environments and provide directional audio guidance.

The system combines semantic segmentation with a custom walkability mapping, perspective weighting, centroid-based direction estimation, safety validation, visual overlays, and offline text-to-speech feedback.

The overall pipeline is:

```text
Mapillary Vistas
       ↓
Annotation Preprocessing
       ↓
Compact Pixel-Level Masks
       ↓
DeepLabv3 + ResNet-50
       ↓
Semantic Segmentation
       ↓
Walkability Mapping
       ↓
Binary Walkability Mask
       ↓
Perspective Weighting
       ↓
Centroid + Safety Strip Analysis
       ↓
Direction
       ↓
Offline Audio Guidance
```

---

# 2. Problem Definition

Outdoor navigation is challenging for visually impaired users because environments contain obstacles, uneven terrain, vehicles, and changing road layouts.

Traditional navigation methods such as GPS provide location information but do not provide detailed real-time semantic understanding of the immediate walking area. Proximity sensors can detect nearby objects but do not identify the type of surface or determine whether a region is suitable for walking.

The objective of this project is to detect walkable regions from visual input and convert that information into simple directional commands:

```text
LEFT
RIGHT
FORWARD
STOP
```

The system therefore connects scene understanding with actionable navigation.

---

# 3. Dataset

The project uses the **Mapillary Vistas Dataset**.

Mapillary Vistas contains more than 25,000 high-resolution street-level images covering diverse countries, weather conditions, lighting conditions, and urban environments. The dataset contains more than 150 semantic classes, including sidewalks, roads, vegetation, buildings, terrain, poles, vehicles, and pedestrians.

Each image is associated with a JSON annotation file containing polygon-based object annotations. A project-level `meta.json` file defines the dataset's semantic classes, including their IDs, names, and colors.

The dataset is used to train the semantic segmentation model to classify every pixel in an image according to its semantic class.

---

# 4. Dataset Structure

The relevant dataset components are:

```text
Dataset/
├── Images/
│   └── street-scene images
│
├── Annotations/
│   └── annotation JSON files
│
└── meta.json
```

### Raw Images

The image directory contains the RGB street-scene images used as model input.

### Annotation JSON

Each annotation JSON describes the objects present in an image. Objects contain their semantic class and polygon coordinates describing their shape.

These polygon coordinates are converted into pixel-level segmentation masks.

### meta.json

`meta.json` contains the master class definitions. It provides the class IDs, class names, and colors used by the dataset.

It acts as the mapping between the dataset's semantic labels and their corresponding pixel representation.

---

# 5. Annotation Preprocessing

The original Mapillary annotations contain polygon descriptions rather than a ready-to-use pixel-level segmentation mask.

For every image, the polygons from its annotation JSON are converted into a pixel-wise mask. Each pixel in the resulting mask represents the semantic class assigned to that location.

The class information is obtained from `meta.json`.

The original class IDs are simplified into compact IDs so that the segmentation model can work with a consistent set of pixel labels.

The preprocessing therefore produces paired training data:

```text
RGB Image
    +
Pixel-Level Segmentation Mask
```

These image-mask pairs are loaded by the custom dataset class during training.

---

# 6. Compact Class Mapping

The original dataset contains a large number of semantic classes. The project maps the relevant classes into compact IDs used by the segmentation model.

The compact representation allows each pixel to contain a single class index while maintaining the semantic meaning of the original Mapillary class.

The preprocessing pipeline is:

```text
Mapillary Polygon Annotation
            ↓
Read Class Information from meta.json
            ↓
Convert Polygon Coordinates to Pixel Mask
            ↓
Simplify Original Class IDs
            ↓
Assign Compact Class IDs
            ↓
Image + Mask Pair
```

This produces the segmentation labels used to train the model.

---

# 7. Walkable Classes

The project defines classes that can represent walkable surfaces.

The walkable classes used in the project include:

- Road side
- Bike lane
- Crosswalk - plain
- Curb cut
- Driveway
- Parking
- Parking aisle
- Pedestrian area
- Road
- Road shoulder
- Service lane
- Sidewalk
- Traffic island

Additional surfaces shown in the project documentation include:

- Park
- Trail
- Footbridge
- Grass
- Stairs
- Gravel track
- Plaza
- Corridor

The project distinguishes between classes present in the dataset/model and additional classes that can be included as walkable surfaces in future extensions.

---

# 8. Walkability Mapping

Semantic segmentation identifies what each pixel represents, but semantic class identity alone does not determine how suitable that region is for walking.

To convert semantic information into navigation information, the project uses a custom:

```text
walkability_meta.json
```

Each semantic class is assigned a walkability score between `0` and `1`.

Examples used by the project are:

```text
Sidewalk       → 1.0
Road            → 0.7
Grass           → 0.3
Walls           → 0
Vehicles        → 0
Poles           → 0
```

The semantic segmentation output is therefore transformed into a pixel-level walkability representation.

A class can be treated as walkable or non-walkable through this mapping without changing the trained segmentation model.

---

# 9. Semantic Segmentation Model

The project uses a DeepLabv3 model with a ResNet-50 backbone for semantic segmentation.

The model performs pixel-wise classification of the input scene. An RGB frame is provided to the model, and the output is a semantic segmentation map in which each pixel corresponds to a semantic class.

The project documentation describes the segmentation architecture as:

```text
DeepLabv3
    +
ResNet-50 Backbone
```

The report describes the model as DeepLabv3+ with a ResNet50 backbone.

The model operates on `512 × 512` RGB input frames.

---

# 10. Training Framework

The model is implemented using:

```text
PyTorch
TorchVision
OpenCV
NumPy
Matplotlib
```

A custom dataset class reads the raw images and annotation JSON files. It uses `meta.json` to convert the dataset class information into pixel indices and produces image-mask pairs for training.

The training configuration uses:

```text
Loss      : Cross Entropy Loss
Optimizer : Adam
```

The training process learns the mapping between RGB street scenes and their pixel-level semantic segmentation masks.

After training, the model produces a semantic class prediction for every pixel of an input frame.

---

# 11. Model Training Pipeline

The training pipeline begins with the preprocessed Mapillary Vistas data.

```text
Raw RGB Image
      +
Annotation JSON
      +
meta.json
      ↓
Polygon-to-Pixel Mask Conversion
      ↓
Compact Class ID Mapping
      ↓
Image + Segmentation Mask
      ↓
Custom Dataset Class
      ↓
DeepLabv3 + ResNet-50
      ↓
Semantic Segmentation Model
```

Cross Entropy Loss is used to compare the predicted pixel classes against the segmentation masks.

Adam is used to update the model parameters during training.

The resulting trained model is used for frame-level semantic segmentation.

---

# 12. Model Output

For an RGB input frame, the trained model produces a semantic segmentation map.

The output identifies the semantic class associated with each pixel in the scene.

The output is then passed to the walkability-processing stage.

The visual output of the system includes:

```text
Input RGB Image
Semantic RGB Mask
Binary Walkability Mask
Walkable Area Overlay
Direction
Walkable Class Information
```

The binary walkability mask and overlay provide the visual representation used to inspect the detected walking region.

---

# 13. Semantic Mask to Walkability Mask

The semantic segmentation mask contains multiple classes.

The system applies the walkability mapping to these semantic classes to determine which pixels contribute to the walkable region.

The resulting representation is converted into a binary walkability mask:

```text
White → Walkable
Black → Non-walkable
```

The binary mask isolates the regions that can be used for navigation.

This mask is also used to generate the visual walkable-area overlay.

---

# 14. Walkable Area Overlay

The binary walkability mask is used to highlight the detected walking region on the original image.

The overlay combines:

```text
Original RGB Frame
        +
Detected Walkable Region
```

This produces a visual representation of where the system believes the walkable area is located.

The project results include output showing the input image, the predicted semantic mask, the binary walkability mask, and the final walkable-area overlay.

---

# 15. Perspective Weighting

Not every pixel in the camera frame represents an equally important area for immediate navigation.

Pixels near the bottom of the frame correspond to the area closer to the user. The system therefore gives greater importance to these pixels.

The weighting function is:

```text
weight = (i / H)²
```

where:

- `i` is the vertical pixel position.
- `H` is the frame height.

The weight increases toward the bottom of the image.

This causes nearby ground regions to have greater influence on navigation than distant regions.

---

# 16. Weighted Walkable Region

The binary walkability mask is combined with the perspective weighting.

The resulting weighted representation gives greater importance to walkable pixels located closer to the bottom of the frame.

This is used to determine the direction of the available walking region.

The system therefore does not treat every detected walkable pixel equally. Near-field walkable pixels contribute more strongly to the navigation decision.

---

# 17. Centroid-Based Navigation

The system computes the centroid of the weighted walkable region.

The centroid represents the center of the detected walkable area.

Its horizontal position is compared with the center of the frame.

```text
Walkable centroid
        ↓
Horizontal position
        ↓
Compare with frame center
        ↓
Direction
```

The navigation logic is:

```text
Centroid left of frame center
        → LEFT

Centroid right of frame center
        → RIGHT

Centroid near frame center
        → FORWARD
```

This converts the detected geometry of the walkable region into a simple directional instruction.

---

# 18. Safety Strip Check

Centroid-based navigation is combined with a separate safety check in the bottom-center portion of the frame.

A bottom-center region is analyzed to determine whether the forward path contains walkable pixels.

If the bottom-center safety strip contains no walkable pixels, the system treats the forward direction as unsafe.

The system then evaluates the available walkable region to the left and right.

The decision is:

```text
Forward strip contains walkable pixels
        → Forward can be selected

Forward strip has no walkable pixels
        ↓
Check available walkable region
        ↓
Left / Right
        ↓
If no usable region exists
        → STOP
```

This prevents the centroid calculation from producing a forward command when the immediate area directly ahead is not walkable.

---

# 19. Complete Navigation Logic

The complete navigation process combines perspective weighting, centroid estimation, and the safety-strip check.

```text
Semantic Segmentation
        ↓
Walkability Mapping
        ↓
Binary Walkability Mask
        ↓
Perspective Weighting
        ↓
Weighted Walkable Region
        ↓
Centroid Calculation
        ↓
Horizontal Deviation
        ↓
Safety Strip Validation
        ↓
Navigation Command
```

The possible commands are:

```text
LEFT
RIGHT
FORWARD
STOP
```

The centroid determines the direction of the walkable region, while the safety strip provides an additional validation of the immediate forward path.

---

# 20. Audio Guidance

The navigation command is converted into spoken feedback using `pyttsx3`.

`pyttsx3` provides offline text-to-speech, so the audio guidance does not require an internet connection.

The system provides four primary spoken commands:

```text
"forward"
"left"
"right"
"stop"
```

The navigation output is therefore directly converted into an instruction that can be communicated to the user.

---

# 21. Audio Rate Limiting

The system processes video frames continuously, so the same direction can be generated across multiple consecutive frames.

Speaking the same command on every frame would produce excessive and repetitive audio.

The system therefore rate-limits the speech output.

A command is spoken when:

```text
The direction changes
OR
The minimum time interval has passed
```

This allows the system to provide continuous navigation feedback without repeatedly speaking the same command on every processed frame.

---

# 22. Real-Time Processing

The system processes the visual input frame by frame.

For each frame:

```text
RGB Frame
    ↓
Semantic Segmentation
    ↓
Walkability Mapping
    ↓
Binary Walkability Mask
    ↓
Walkable Area Analysis
    ↓
Centroid Calculation
    ↓
Safety Strip Check
    ↓
Direction
    ↓
Audio Command
```

The output can simultaneously be visualized as an overlay containing the detected walkable area and the navigation direction.

The project results demonstrate the system operating on multiple video samples with real-time walkable-area visualization and directional guidance.

---

# 23. Output Visualization

The system produces several stages of visual output.

### Input

The original RGB street scene.

### Semantic RGB Mask

The semantic segmentation output showing the classes detected by the model.

### Binary Walkability Mask

The semantic output converted into a two-class representation:

```text
White → Walkable
Black → Non-walkable
```

### Walkable Area Overlay

The detected walkable region is overlaid onto the original image.

### Navigation Output

The final output combines the walkable-area visualization with the detected direction and walkable class information.

The project results show outputs containing:

```text
Input Image
Predicted RGB Mask
Binary Walkability Mask
Walkable Area Overlay
Direction
Walkable Class
```

---

# 24. Dynamic Walkability

Walkability is separated from semantic segmentation through the `walkability_meta.json` mapping.

The segmentation model learns semantic classes, while the walkability mapping determines how those classes are treated for navigation.

This means that the walkability definition can be changed by modifying the class-to-score mapping without retraining the segmentation model.

For example:

```text
Semantic Segmentation
        ↓
"sidewalk"
"road"
"grass"
"vehicle"
"pole"
        ↓
Walkability Mapping
        ↓
Walkability Scores
        ↓
Binary Walkability Mask
```

This separation allows the navigation rules to be adjusted independently of the semantic segmentation model.

---

# 25. End-to-End System Architecture

The complete system can be represented as five major stages.

## Stage 1: Data Preprocessing

Mapillary Vistas images and polygon annotations are converted into pixel-level segmentation masks.

```text
Images
+
annotation.json
+
meta.json
        ↓
Pixel-Level Masks
```

## Stage 2: Model Training

The image-mask pairs are loaded using the custom dataset class and used to train the DeepLabv3 segmentation model with a ResNet-50 backbone.

```text
Image + Mask
      ↓
DeepLabv3 + ResNet-50
      ↓
Trained Segmentation Model
```

## Stage 3: Semantic Segmentation

An RGB frame is passed through the trained model.

```text
RGB Frame
    ↓
Semantic Segmentation
    ↓
Class per Pixel
```

## Stage 4: Walkability and Navigation

The semantic classes are converted into walkability scores and then into a binary walkability mask.

```text
Semantic Mask
    ↓
Walkability Mapping
    ↓
Binary Walkability Mask
    ↓
Perspective Weighting
    ↓
Centroid
    +
Safety Strip
    ↓
Direction
```

## Stage 5: Audio Guidance

The direction is converted into offline spoken feedback.

```text
Direction
    ↓
pyttsx3
    ↓
LEFT / RIGHT / FORWARD / STOP
```

---

# 26. Detailed Data Flow

The complete data flow is:

```text
                         TRAINING

Mapillary Vistas
      │
      ├── RGB Images
      │
      ├── annotation.json
      │
      └── meta.json
              │
              ▼
      Polygon Processing
              │
              ▼
      Pixel-Level Masks
              │
              ▼
      Compact Class IDs
              │
              ▼
      Image + Mask Pairs
              │
              ▼
      Custom Dataset Class
              │
              ▼
      DeepLabv3 + ResNet-50
              │
              ▼
      Trained Model


                         INFERENCE

RGB Video Frame
      │
      ▼
Semantic Segmentation
      │
      ▼
Semantic Mask
      │
      ▼
walkability_meta.json
      │
      ▼
Walkability Representation
      │
      ▼
Binary Walkability Mask
      │
      ├───────────────► Overlay
      │
      ▼
Perspective Weighting
      │
      ▼
Weighted Walkable Region
      │
      ▼
Centroid Calculation
      │
      ▼
Safety Strip Check
      │
      ▼
Direction
      │
      ▼
pyttsx3
      │
      ▼
Audio Guidance
```

---

# 27. Example Navigation Scenario

Consider a frame where the detected walkable region is shifted toward the left side of the image.

The segmentation model first identifies the semantic regions in the scene.

The walkability mapping converts those semantic regions into walkability values. The resulting binary mask isolates the usable walking area.

Perspective weighting gives greater importance to the lower portion of the frame.

The centroid of the weighted walkable region is then calculated. If this centroid lies to the left of the frame center, the navigation output becomes:

```text
LEFT
```

If the centroid lies near the center and the bottom-center safety strip contains walkable pixels, the output becomes:

```text
FORWARD
```

If the forward safety strip is blocked, the system checks the available walkable region on either side and selects the corresponding direction. If no usable region is available, the system outputs:

```text
STOP
```

The selected command is then passed to the offline text-to-speech system.

---

# 28. Software Components

The project uses the following software components:

| Component | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| TorchVision | Vision models and utilities |
| OpenCV | Image/video processing |
| NumPy | Numerical and mask operations |
| Matplotlib | Visualization |
| pyttsx3 | Offline text-to-speech |
| Mapillary Vistas | Semantic segmentation dataset |

---

# 29. Important Project Files

The project uses the following important data/configuration components:

```text
meta.json
```

Defines the dataset's semantic classes, IDs, names, and colors.

```text
annotation.json
```

Contains polygon annotations for individual images.

```text
walkability_meta.json
```

Defines the walkability score associated with each semantic class.

The relationship between these files is:

```text
meta.json
      ↓
Semantic Class Definitions
      ↓
annotation.json
      ↓
Pixel-Level Semantic Masks
      ↓
Model Training
      ↓
Semantic Prediction
      ↓
walkability_meta.json
      ↓
Walkability Mask
```

---

# 30. Reproduction Workflow

The complete implementation follows this sequence.

### Step 1 — Obtain the Dataset

Use the Mapillary Vistas dataset containing the RGB street-level images, polygon annotation JSON files, and `meta.json`.

### Step 2 — Preprocess the Annotations

Read the class definitions from `meta.json` and the polygon annotations from each image's annotation JSON.

Convert the polygon coordinates into pixel-level segmentation masks.

### Step 3 — Simplify the Class IDs

Convert the original dataset class IDs into compact IDs suitable for pixel-wise model training.

### Step 4 — Build the Training Dataset

Create image-mask pairs using the custom dataset class.

### Step 5 — Train the Segmentation Model

Train the DeepLabv3 model with the ResNet-50 backbone using PyTorch.

Use Cross Entropy Loss and Adam during training.

### Step 6 — Run Semantic Segmentation

Pass an RGB frame through the trained model to obtain a semantic segmentation map.

### Step 7 — Apply Walkability Mapping

Use `walkability_meta.json` to assign a walkability score to each predicted semantic class.

### Step 8 — Generate the Binary Mask

Convert the walkability representation into a binary mask:

```text
1 / White  → Walkable
0 / Black  → Non-walkable
```

### Step 9 — Apply Perspective Weighting

Apply:

```text
weight = (i / H)²
```

to prioritize regions closer to the bottom of the frame.

### Step 10 — Calculate the Walkable Centroid

Calculate the centroid of the weighted walkable region and compare its horizontal position with the center of the frame.

### Step 11 — Validate the Forward Path

Analyze the bottom-center safety strip.

If the forward region is not walkable, select an available side direction or output `STOP` when no usable region is available.

### Step 12 — Generate Audio

Convert the resulting direction into speech using `pyttsx3`.

Rate-limit the speech so that commands are produced when the direction changes or the minimum time interval has passed.

### Step 13 — Generate Visual Output

Overlay the detected walkable region onto the input frame and display the navigation output.

---

# 31. Design Principles

## Semantic Understanding Before Navigation

The system does not directly attempt to predict a direction from the image. It first understands the scene through semantic segmentation.

```text
Image
 ↓
Semantic Understanding
 ↓
Walkability
 ↓
Navigation
```

This separates scene understanding from navigation logic.

## Walkability as a Separate Layer

Semantic classes are converted into walkability scores after segmentation.

This makes the navigation definition independent from the segmentation model.

## Near-Field Priority

The perspective weighting gives greater importance to pixels near the bottom of the image because they represent the area closer to the user.

## Centroid-Based Direction

The centroid provides a geometric representation of where the available walkable area lies relative to the user.

## Safety Validation

The bottom-center safety strip provides an additional check before allowing a forward direction.

## Offline Audio

`pyttsx3` provides voice feedback without requiring an internet connection.

---

# 32. Testing and Results

The system was tested on multiple video samples.

The reported results include:

### Walkable Path Detection

The system identifies walkable regions in complex, cluttered, and uneven outdoor scenes.

### Directional Stability

The centroid-based navigation method combined with safety checks provides stable directional decisions.

### Real-Time Processing

The system produces walkable-area overlays and navigation outputs while processing video.

### Audio Guidance

The generated voice commands provide consistent directional feedback in real time.

The overall system transforms semantic segmentation output into actionable navigation information.

---

# 33. Practical Applications

The project demonstrates a foundation for assistive navigation for visually impaired users.

The segmentation and walkability information can also be used for:

- Navigation applications
- Training assistive AI systems
- Early warning systems
- Urban planning
- Autonomous wheelchairs or robots
- GIS-based pedestrian infrastructure analysis

The project documentation specifically describes the possibility of integrating walkability scores into routing systems and using segmented visual data for smart mobility aids and vision-based navigation assistants.

---

# 34. System Characteristics

The completed system provides:

```text
Real-Time Processing
        +
Semantic Scene Understanding
        +
Walkability Classification
        +
Near-Field Weighting
        +
Centroid-Based Navigation
        +
Safety Validation
        +
Visual Walkable-Area Overlay
        +
Offline Audio Guidance
```

The system does not require internet connectivity for its audio feedback.

---

# 35. Future Extensions

The project identifies several possible future improvements:

### Mobile Deployment

The system can be adapted for mobile deployment using technologies such as TensorFlow Lite or CoreML.

### Depth-Based Detection

Depth information can be incorporated for step and obstacle detection.

### Haptic Feedback

Haptic feedback can be added as another navigation modality alongside audio.

### Lightweight Models

Lighter segmentation models can be explored for faster inference and deployment on resource-constrained devices.

---

# 36. Complete System Summary

The Walkable Path Detection and Audio Navigation System converts street-level visual data into navigation instructions through a sequence of semantic and geometric processing stages.

The Mapillary Vistas dataset provides RGB images and polygon annotations. The annotations are converted into pixel-level masks using the dataset's class definitions. These masks are simplified into compact class IDs and used to train a DeepLabv3 segmentation model with a ResNet-50 backbone.

During inference, the trained model produces a semantic segmentation map for each RGB frame. The semantic classes are converted into walkability scores using `walkability_meta.json`. The resulting walkability representation is converted into a binary mask and used to identify the usable walking region.

Perspective weighting prioritizes regions near the bottom of the frame. The centroid of the weighted walkable region determines the direction of the available path relative to the center of the frame. A bottom-center safety strip provides an additional check for the immediate forward path.

The resulting navigation command is converted into offline speech using `pyttsx3`, with rate limiting to prevent repeated commands from being spoken continuously.

The complete system therefore follows:

```text
Street Scene
     ↓
Semantic Segmentation
     ↓
Walkability Mapping
     ↓
Binary Walkability Mask
     ↓
Perspective Weighting
     ↓
Centroid Navigation
     +
Safety Strip Validation
     ↓
LEFT / RIGHT / FORWARD / STOP
     ↓
Offline Audio Guidance
```

This combines semantic scene understanding, walkability analysis, geometric navigation, visual feedback, and audio guidance into a single assistive navigation pipeline.
