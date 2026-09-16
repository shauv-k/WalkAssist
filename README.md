# Walkable Path Detection and Audio Navigation System

A real-time computer vision system designed to help visually impaired users navigate outdoor environments. The system identifies walkable regions from a camera feed, determines a suitable direction to move, and provides simple audio instructions.

![Real-Time Demo](assets/demo.gif)

## Working Pipeline

The system processes a street-level image through several stages: the raw dataset annotations are first converted into pixel-wise masks, the segmentation model is trained on these masks, and the resulting predictions are then converted into walkability information. This walkability information is used to determine a suitable path and generate audio guidance.

![Working Pipeline](assets/pipeline.png)

## Data Preprocessing

The Mapillary Vistas dataset provides street images along with JSON annotations describing the objects and surfaces present in each image. `meta.json` provides the class definitions, while each image's `annotation.json` contains polygon coordinates for its labelled regions.

These polygons are converted into pixel-wise segmentation masks. The original dataset class IDs are then mapped to a compact set of class IDs that can be used by the segmentation model. Each resulting image is paired with its corresponding mask to create the training data.

This preprocessing gives the model a consistent pixel-level label for every part of the image and also allows the same segmentation output to later be interpreted in terms of walkability. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

## Model & Training

We use **DeepLabv3 with a ResNet-50 backbone** for semantic segmentation. The model takes an RGB image and predicts the semantic class of each pixel. The project uses PyTorch and TorchVision, with a custom dataset class responsible for loading the processed image-mask pairs.

The model is trained as a multi-class segmentation problem using **Cross Entropy Loss** and the **Adam optimizer**. After training, the model can produce a semantic segmentation mask for each input frame, identifying surfaces such as sidewalks, roads, grass, and other objects. :contentReference[oaicite:3]{index=3}

## Mask Processing & Walkability

The segmentation output tells us what each pixel represents, but not whether that region is suitable for walking. To bridge this gap, each semantic class is assigned a walkability score between 0 and 1.

For example, sidewalks are given a high walkability score, roads a lower score, while objects such as vehicles, walls, and poles are treated as non-walkable. These scores are used to create a walkability map, which is then thresholded into a binary mask separating walkable and non-walkable regions.

This binary mask is used both for the visual overlay and as the input to the navigation logic. An important part of the design is that the walkability mapping is separate from the trained segmentation model, so the definition of what is considered walkable can be changed without retraining the model. :contentReference[oaicite:4]{index=4}

![Walkability Overlay](assets/overlay.png)

## Navigation

The system gives greater importance to regions near the bottom of the frame because they represent areas closer to the user. It calculates the centre of the weighted walkable region and compares it with the centre of the camera view.

If the walkable region is shifted to the left or right, the corresponding direction is selected. If it is centred, the system indicates forward movement.

A separate safety check examines the bottom-centre region directly in front of the user. If that region is not walkable, the system avoids giving a forward instruction and instead looks for an available path to the left or right. If no suitable path is available, it outputs **STOP**. :contentReference[oaicite:5]{index=5}

## Audio Cues

The navigation decision is converted into simple spoken instructions using the offline `pyttsx3` text-to-speech engine. The system provides four basic cues:

**LEFT · RIGHT · FORWARD · STOP**

To avoid continuously repeating the same instruction, audio output is rate-limited and triggered when the direction changes or when a minimum time interval has passed. :contentReference[oaicite:6]{index=6}
