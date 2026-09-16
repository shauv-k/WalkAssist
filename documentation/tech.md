# Technical Documentation
## Walkable Path Detection and Audio Navigation System

This document describes the complete implementation of the Walkable Path Detection and Audio Navigation System, from Mapillary Vistas annotations through semantic segmentation, walkability classification, navigation decision-making, visualization, and audio feedback.

The intention is that someone familiar with Python and deep learning can understand the architecture and reproduce the documented implementation.

---

# 1. Project Overview

The system is a camera-based navigation system designed to help visually impaired users navigate outdoor environments. It identifies regions that can potentially be used for walking, determines a suitable direction, and provides real-time audio instructions:

- `FORWARD`
- `LEFT`
- `RIGHT`
- `STOP`

The system does not directly train a model to predict left or right. It separates the problem into semantic understanding and navigation reasoning.

The complete pipeline is:

```text
Mapillary Vistas
    ↓
Polygon Annotations
    ↓
Pixel-wise Semantic Masks
    ↓
Compact Class IDs
    ↓
DeepLabv3 + ResNet-50
    ↓
Semantic Segmentation
    ↓
Walkability Mapping
    ↓
Walkability Heatmap
    ↓
Binary Walkability Mask
    ↓
Perspective Weighting
    ↓
Walkable Region Centroid
    ↓
Safety Strip Validation
    ↓
Navigation Decision
    ↓
Text-to-Speech
```

The system was designed to operate without an internet connection during inference, including audio output.

# 2. Problem Definition

Outdoor navigation presents several challenges for visually impaired users. GPS can provide a route but cannot determine what is immediately in front of the user. Conventional proximity sensing can detect nearby objects but does not provide semantic information about whether a region is a sidewalk, road, vehicle, wall, or another part of the environment.

The objective is to obtain semantic information from a camera frame and transform it into an actionable navigation command.

Semantic segmentation alone is not sufficient. A segmentation model may identify sidewalk, road, grass, vehicle, wall, and pole, but it does not inherently know which of those classes should be considered walkable.

The project therefore introduces a separate walkability mapping layer between semantic segmentation and navigation.

# 3. Dataset

## 3.1 Mapillary Vistas

The project uses the Mapillary Vistas dataset. It contains more than 25,000 high-resolution street-level images and more than 150 semantic classes covering objects and surfaces commonly encountered in outdoor environments.

Examples include sidewalks, roads, vegetation, buildings, terrain, poles, vehicles, and pedestrians.

The dataset was selected because it contains real-world street scenes with detailed pixel-level semantic annotations.

# 4. Dataset Structure

The relevant dataset information is distributed between RGB images, annotation JSON files, and `meta.json`.

A simplified representation is:

```text
Mapillary Vistas/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── annotations/
│   ├── image_001.json
│   ├── image_002.json
│   └── ...
└── meta.json
```

The exact directory structure can vary depending on the dataset distribution, but the three important components are:

### RGB images

The original street-level images used as model inputs.

### Annotation JSON files

Each image has an associated JSON annotation containing objects and polygon coordinates describing their shapes. Conceptually:

```text
Object:
    class = sidewalk
    polygon = [(x1,y1), (x2,y2), ...]
```

These polygons must be converted into pixel-wise masks before they can be used for semantic segmentation training.

### `meta.json`

`meta.json` defines the dataset class vocabulary, including class names, IDs, and colours. It provides the mapping required to interpret the annotations.

# 5. Annotation Preprocessing

The original Mapillary annotations are polygon-based rather than directly providing the compact integer masks required by the segmentation model.

For every image:

1. Load the corresponding annotation JSON.
2. Read each annotated object.
3. Obtain its semantic class.
4. Obtain its polygon coordinates.
5. Rasterize the polygon into the image dimensions.
6. Assign the corresponding class ID to every pixel inside the polygon.

The result is a two-dimensional semantic mask corresponding to the RGB image.

```text
RGB image
    ↓
Polygon annotations
    ↓
Pixel-wise semantic mask
```

# 6. Compact Class ID Mapping

The original dataset contains a large number of semantic classes and its IDs are not necessarily convenient for direct model training. The preprocessing stage converts the original class identifiers into a compact set of IDs.

For example:

```text
Original Dataset ID    Compact ID
class A                0
class B                1
class C                2
class D                3
...
```

Every pixel belonging to the same semantic class receives the same compact integer ID.

The resulting training data is therefore a set of image-mask pairs:

```text
annotation.json
      ↓
polygon coordinates
      ↓
pixel-wise mask
      ↓
compact class IDs
      ↓
training mask
```

# 7. Walkability Classes

Semantic segmentation provides information about the environment, but semantic classes do not all have the same navigation meaning.

The project therefore defines a separate walkability representation. Relevant classes include surfaces such as:

- Sidewalk
- Road
- Road shoulder
- Bike lane
- Crosswalk
- Pedestrian area
- Parking
- Parking aisle
- Driveway
- Traffic island
- Trail
- Grass
- Stairs
- Plaza

The important design decision is that walkability is not hard-coded into the neural network. It is defined after segmentation.

# 8. Walkability Mapping

A custom file named `walkability_meta.json` associates semantic classes with numerical walkability scores from 0 to 1.

Examples documented by the project are:

```text
Sidewalk  → 1.0
Road      → 0.7
Grass     → 0.3
Vehicle   → 0.0
Wall      → 0.0
Pole      → 0.0
```

The mapping is independent of the trained segmentation model. Changing a class's walkability score changes post-processing behaviour without requiring the segmentation network to be retrained.

For example:

```text
Grass → 0.3
```

can be changed to:

```text
Grass → 0.0
```

without changing the segmentation model.

# 9. Semantic Segmentation Model

The segmentation model used in the project is DeepLabv3 with a ResNet-50 backbone. The project report describes the architecture as DeepLabv3+, while the implementation/presentation identifies it as DeepLabv3 with ResNet-50. Reproduction should follow the actual model implementation rather than assuming undocumented architectural modifications.

The model performs pixel-wise classification:

```text
RGB Image
    ↓
DeepLabv3
    ↓
ResNet-50 backbone
    ↓
Semantic segmentation
    ↓
Class prediction for every pixel
```

# 10. Training Framework

The training implementation uses Python, PyTorch, TorchVision, OpenCV, NumPy, and Matplotlib.

A custom PyTorch dataset loads the preprocessed image-mask pairs and returns an RGB image together with its corresponding semantic mask.

```text
Image path ──→ RGB image ──┐
                           ├──→ (image, mask)
Mask path  ──→ class mask ─┘
```

# 11. Model Training

The model is trained as a multi-class semantic segmentation model. The target for every pixel is the compact semantic class ID produced during preprocessing.

The documented training configuration uses Cross Entropy Loss and the Adam optimizer.

The training loop is:

```text
Load image + ground-truth mask
            ↓
       Forward pass
            ↓
      Segmentation output
            ↓
     Cross Entropy Loss
            ↓
       Backpropagation
            ↓
      Adam optimization
            ↓
       Updated model
```

After training, the resulting checkpoint is loaded for inference on new images or video frames.

## 11.1 Training Parameters Not Specified

The supplied project documentation does not specify exact values for:

- number of epochs
- batch size
- learning rate
- train/validation split
- learning-rate scheduler
- checkpoint frequency
- exact augmentation pipeline
- exact pretrained-weight configuration

Those values must be recovered from the original training code/configuration for an exact reproduction of the original training run. They should not be invented from the report.

# 12. Inference Pipeline

Once trained, the model receives an RGB frame, applies the required preprocessing, and produces a semantic segmentation output.

```text
Camera frame
     ↓
Preprocessing
     ↓
DeepLabv3 + ResNet-50
     ↓
Semantic prediction
     ↓
Predicted class for each pixel
```

The semantic prediction is then passed to the walkability-processing stage.

# 13. Semantic Mask to Walkability Heatmap

For each predicted pixel, the system obtains the predicted semantic class and uses that class to look up its walkability score in `walkability_meta.json`.

For example:

```text
Predicted pixel
      ↓
sidewalk
      ↓
walkability_meta.json
      ↓
1.0
```

and:

```text
Predicted pixel
      ↓
vehicle
      ↓
walkability_meta.json
      ↓
0.0
```

This produces a floating-point walkability map.

Conceptually:

```text
Semantic mask:

sidewalk sidewalk road vehicle wall
sidewalk sidewalk road vehicle wall
grass    grass    road vehicle pole

             ↓

Walkability map:

1.0  1.0  0.7  0.0  0.0
1.0  1.0  0.7  0.0  0.0
0.3  0.3  0.7  0.0  0.0
```

# 14. Binary Walkability Mask

The heatmap is converted into a binary walkability representation:

```text
1 → walkable
0 → non-walkable
```

For visualization, the project represents walkable pixels as white and non-walkable pixels as black.

Conceptually:

```text
Walkability scores:

1.0  1.0  0.7  0.0  0.0
1.0  1.0  0.7  0.0  0.0
0.3  0.3  0.7  0.0  0.0

             ↓ threshold

Binary mask:

1    1    1    0    0
1    1    1    0    0
0    0    1    0    0
```

The binary mask is used for both visualization and navigation.

# 15. Visual Overlay

The binary walkability mask is overlaid on the original RGB image to show which parts of the scene the system considers walkable.

```text
RGB frame
   │
   ├──────────────┐
   │              │
   ▼              ▼
Original       Binary mask
image              │
   │               │
   └───────┬───────┘
           ↓
        Overlay
```

This visualization is also useful for verifying the model's predictions during testing.

# 16. Perspective Weighting

Treating every walkable pixel equally can produce poor navigation decisions. A walkable region far in the distance may occupy a large part of the image even though the area immediately in front of the user is more important.

The project therefore applies perspective weighting. Pixels near the bottom of the image receive greater importance because they represent regions closer to the user.

The documented weighting function is:

```text
w(i) = (i / H)^2
```

where `i` is the vertical pixel coordinate and `H` is the image height.

Therefore:

```text
Top of image    → low weight
Middle          → medium weight
Bottom of image → high weight
```

The quadratic weighting increasingly emphasizes the lower part of the frame.

# 17. Weighted Walkable Region

The binary walkability mask is multiplied by the perspective weights.

For a pixel:

```text
W(x,y) = M(x,y) × w(y)
```

where `M(x,y)` is the binary walkability mask and `w(y)` is the perspective weight.

Non-walkable pixels contribute nothing, while walkable pixels closer to the bottom of the frame contribute more strongly.

# 18. Centroid-Based Direction Estimation

The navigation system determines where the available walking path is located by calculating the centroid of the weighted walkable region.

The weighted centroid is calculated as:

```text
x_c = Σ(x × W(x,y)) / ΣW(x,y)

y_c = Σ(y × W(x,y)) / ΣW(x,y)
```

The horizontal centroid `x_c` is used for directional reasoning.

The horizontal centre of the frame is:

```text
x_f = W / 2
```

The system compares the walkable centroid with the camera centre.

# 19. Direction Decision

The horizontal position of the centroid determines the basic navigation direction.

If the centroid is sufficiently to the left of the frame centre:

```text
LEFT
```

If it is sufficiently to the right:

```text
RIGHT
```

If it is near the centre:

```text
FORWARD
```

In practice, a tolerance around the centre can be used because the centroid will rarely fall exactly on the centre pixel.

```text
LEFT        FORWARD        RIGHT
  ↓             ↓             ↓
centroid     centroid      centroid
left of       near         right of
centre       centre        centre
```

# 20. Safety Strip

Centroid direction alone is not sufficient. The overall walkable region may point forward while the area immediately in front of the user is blocked.

The project therefore performs a separate safety-strip check over a bottom-centre region of the image.

```text
┌─────────────────────────┐
│                         │
│       distant area      │
│                         │
│                         │
│      ┌───────────┐      │
│      │   SAFETY  │      │
│      │   STRIP   │      │
│      └───────────┘      │
└─────────────────────────┘
```

The strip is checked against the binary walkability mask to determine whether the immediate forward region is available.

# 21. Safety Decision Logic

The navigation process therefore has two stages:

1. Determine where the walkable region points.
2. Verify that moving forward is actually possible.

If the safety strip contains walkable pixels, the normal centroid-based direction can be used.

If the safety strip is not walkable:

```text
Forward path blocked
        ↓
Check left/right walkable regions
        ↓
left available  → LEFT
right available → RIGHT
neither         → STOP
```

This prevents a `FORWARD` command from being generated solely because a walkable region exists farther away.

# 22. Complete Navigation Algorithm

```text
Input:
    Binary walkability mask

1. Calculate perspective weights for image rows.

2. Multiply the binary walkability mask
   by the perspective weights.

3. Calculate the centroid of the weighted
   walkable region.

4. Compare the centroid's horizontal position
   with the centre of the frame.

5. Inspect the bottom-centre safety strip.

6. If the safety strip is walkable:
       use centroid direction.

7. If the safety strip is not walkable:
       evaluate left and right walkable regions.

8. If an alternative exists:
       select LEFT or RIGHT.

9. If no usable region exists:
       output STOP.
```

# 23. Audio Feedback

The final navigation command is converted into speech using `pyttsx3`.

The navigation layer generates one of:

```text
left
right
forward
stop
```

The audio pipeline is:

```text
Navigation decision
        ↓
Text command
        ↓
pyttsx3
        ↓
System audio output
```

Because `pyttsx3` operates locally, speech generation does not require an internet connection.

# 24. Audio Rate Limiting

Without rate limiting, the system would attempt to announce the same direction for every processed video frame.

For example:

```text
FORWARD
FORWARD
FORWARD
FORWARD
...
```

The implementation therefore limits speech output. An instruction is spoken when the detected direction changes or when a minimum time interval has elapsed.

```text
Current direction
       ↓
Compare with previous direction
       ↓
Changed? ── Yes ──→ Speak
   │
   No
   ↓
Check elapsed time
   ↓
Interval elapsed? ── Yes ──→ Speak
                    No  ───→ Ignore
```

# 25. Real-Time Video Processing

During inference, the system operates on video frames rather than individual static images.

The high-level processing loop is:

```python
while camera_is_running:

    frame = read_frame()

    segmentation = model(frame)

    walkability = map_classes_to_scores(segmentation)

    binary_mask = create_binary_mask(walkability)

    weighted_mask = apply_perspective_weighting(binary_mask)

    direction = determine_direction(weighted_mask)

    direction = safety_check(direction, binary_mask)

    display_overlay(frame, binary_mask, direction)

    speak_when_required(direction)
```

The exact camera/video capture implementation is not specified in the supplied report, but this represents the documented processing sequence.

# 26. Complete System Architecture

The system can be divided into six logical modules.

## Module 1 — Dataset Processing

```text
Mapillary images
       +
annotation JSON
       +
meta.json
       ↓
pixel-wise class masks
```

## Module 2 — Semantic Segmentation

```text
RGB frame
    ↓
DeepLabv3
    ↓
ResNet-50
    ↓
semantic segmentation
```

## Module 3 — Walkability Processing

```text
semantic segmentation
        ↓
walkability_meta.json
        ↓
walkability scores
        ↓
binary mask
```

## Module 4 — Navigation

```text
binary mask
    ↓
perspective weighting
    ↓
centroid calculation
    ↓
safety strip
    ↓
direction
```

## Module 5 — Visualization

```text
RGB frame
+
walkability mask
+
direction
        ↓
annotated output frame
```

## Module 6 — Audio

```text
direction
    ↓
rate limiting
    ↓
pyttsx3
    ↓
spoken instruction
```

# 27. End-to-End Example

Consider a frame containing a sidewalk on the left, road in the centre, and a parked vehicle on the right.

The segmentation model identifies the corresponding semantic regions.

The walkability mapping may produce:

```text
Sidewalk → 1.0
Road     → 0.7
Vehicle  → 0.0
```

The resulting heatmap therefore favours the sidewalk and gives the road a lower walkability score, while the vehicle contributes nothing.

Perspective weighting gives greater importance to walkable pixels near the bottom of the frame. Suppose the resulting centroid lies to the left of the camera centre:

```text
Centroid → LEFT
```

The safety strip is then checked. If the immediate area is walkable, the final command is `LEFT`. If the immediate area is blocked, the system checks alternative left/right regions and produces `LEFT`, `RIGHT`, or `STOP` depending on what is available.

The final command is passed to the audio system.

# 28. Output Generation

The project generates several representations of the same frame.

### Original frame

The unmodified RGB input.

### Semantic RGB mask

A visualization in which different semantic classes are represented using different colours.

### Walkability mask

A binary representation in which walkable and non-walkable regions are separated.

### Walkability overlay

The detected walkable region is overlaid on the original camera image.

### Navigation output

The selected direction is displayed with the processed frame.

### Audio output

The same navigation decision is converted into speech.

The output pipeline is:

```text
RGB Input
   ↓
Semantic RGB Mask
   ↓
Walkability Mask
   ↓
Walkability Overlay
   ↓
Navigation + Audio
```

# 29. Software Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Vision models | TorchVision |
| Segmentation | DeepLabv3 |
| Backbone | ResNet-50 |
| Image processing | OpenCV |
| Numerical processing | NumPy |
| Visualization | Matplotlib |
| Text-to-speech | pyttsx3 |
| Dataset | Mapillary Vistas |

# 30. Reproduction Procedure

A reproduction should follow these stages.

## Step 1 — Obtain Mapillary Vistas

Obtain the RGB images, annotation JSON files, and `meta.json`.

## Step 2 — Parse `meta.json`

Build the mapping between dataset class IDs, class names, and colours.

## Step 3 — Convert annotations

For every image, load the annotation JSON, read the polygons and semantic classes, rasterize the polygons, and generate the corresponding pixel-wise mask.

## Step 4 — Compact class IDs

Convert the original class IDs into the compact IDs used by the segmentation model.

The result should be image-mask pairs where every mask pixel contains an integer semantic label.

## Step 5 — Implement the Dataset

Create a PyTorch dataset that reads an image and its corresponding mask, applies the required preprocessing, and returns the pair.

## Step 6 — Create the Segmentation Model

Instantiate DeepLabv3 with a ResNet-50 backbone and configure its output layer for the number of semantic classes used by the training masks.

## Step 7 — Train

Train using Cross Entropy Loss and the Adam optimizer. Exact training hyperparameters must be taken from the original implementation because they are not specified in the supplied documentation.

## Step 8 — Save the Trained Model

Save the trained segmentation model as a checkpoint for inference.

## Step 9 — Create Walkability Metadata

Create `walkability_meta.json` containing the desired walkability score for every semantic class.

Example:

```json
{
    "sidewalk": 1.0,
    "road": 0.7,
    "grass": 0.3,
    "vehicle": 0.0,
    "wall": 0.0,
    "pole": 0.0
}
```

The complete mapping should cover the classes used by the trained model.

## Step 10 — Run Inference

For each camera or video frame:

```text
Read frame
→ preprocess
→ segmentation model
→ semantic prediction
→ walkability mapping
→ binary mask
```

## Step 11 — Calculate Navigation

Apply perspective weighting, calculate the weighted centroid, compare its horizontal position with the frame centre, and perform the safety-strip check.

## Step 12 — Generate Visual Output

Combine the original frame, walkability mask, and navigation direction to produce the real-time visual output.

## Step 13 — Generate Audio

Pass the final navigation command to `pyttsx3` and apply the direction-change/time-interval logic to prevent repeated speech.

# 31. Important Design Decisions

Several parts of the project are deliberately separated rather than incorporated into the neural network.

The model answers:

> What is this pixel?

The walkability layer answers:

> How suitable is this class for walking?

The navigation layer answers:

> Given the walkable region, where should the user move?

The audio layer answers:

> How should that decision be communicated?

This separation makes the system easier to modify and debug.

# 32. Why Walkability Is Post-Processed

Training the model directly to predict `LEFT`, `RIGHT`, `FORWARD`, and `STOP` would tie the model to a particular navigation policy.

Instead, the project retains semantic information:

```text
Model:
    sidewalk
    road
    grass
    vehicle
    wall
    ...
```

This semantic output can then be interpreted using different walkability and navigation policies.

Therefore:

```text
Semantic model
       ↓
Walkability policy
       ↓
Navigation policy
```

remain separate.

# 33. Failure Handling

The safety-strip mechanism is the primary documented mechanism for handling cases where the normal centroid-based direction is unsafe.

The system can transition from `FORWARD` to `LEFT` or `RIGHT` when the immediate forward region is unavailable.

If neither alternative contains a usable walkable region, `STOP` is generated.

# 34. Testing

The system was tested using multiple video samples.

The documented results indicate that walkable areas were detected in complex outdoor scenes, directional logic remained stable, the overlay and navigation operated at usable real-time frame rates, and audio cues were generated consistently.

The project therefore demonstrated the complete path:

```text
Video
 ↓
Segmentation
 ↓
Walkability
 ↓
Navigation
 ↓
Audio
```

# 35. Reproducibility Notes

The project documentation specifies the main architecture and processing pipeline, but some low-level implementation details are not available in the supplied report.

For exact numerical reproduction of the original training run, the following information must be recovered from the original source code/configuration:

```text
Exact train/validation split
Number of epochs
Batch size
Learning rate
Data augmentation
Normalization values
Checkpoint used
Pretrained model weights
Exact number of output classes
Exact walkability threshold
Exact safety-strip dimensions
Exact centroid tolerance
Exact audio interval
Exact video resolution/FPS
```

These values should be taken from the original implementation rather than inferred.

The documented algorithm itself is:

```text
Mapillary Vistas
      ↓
Polygon → pixel mask
      ↓
Compact class IDs
      ↓
DeepLabv3 + ResNet-50
      ↓
Semantic segmentation
      ↓
walkability_meta.json
      ↓
Walkability heatmap
      ↓
Binary mask
      ↓
Perspective weighting
      ↓
Weighted centroid
      ↓
Safety strip
      ↓
LEFT / RIGHT / FORWARD / STOP
      ↓
pyttsx3
      ↓
Audio instruction
```

# 36. Limitations

The system provides semantic and geometric information from a monocular camera but does not explicitly perform depth estimation.

Consequently, the current implementation does not directly measure the physical distance to an obstacle or determine the height/depth of terrain.

The project identifies several areas for future development:

- mobile deployment using TensorFlow Lite or CoreML
- depth-based obstacle and step detection
- multimodal haptic feedback
- lighter segmentation models for faster inference

These extensions would add information that is not available from the current semantic segmentation and walkability pipeline alone.

# 37. Final System Summary

The project converts a conventional semantic segmentation problem into a navigation system through a sequence of deterministic processing stages.

The neural network performs scene understanding:

> What is present in this image?

The walkability mapping performs semantic interpretation:

> Which of these things can be walked on?

Perspective weighting performs spatial prioritization:

> Which walkable regions matter most right now?

The centroid performs direction estimation:

> Where is the usable walking path?

The safety strip performs immediate-path validation:

> Is it actually possible to move forward?

Finally, the audio system performs user communication:

> What instruction should the user hear?

Together, these components transform:

```text
Raw camera frame
        ↓
Semantic understanding
        ↓
Walkable region
        ↓
Safe direction
        ↓
Spoken navigation instruction
```

without requiring the segmentation network itself to learn the navigation policy.
