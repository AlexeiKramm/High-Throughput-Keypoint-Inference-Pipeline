# High-Throughput Keypoint Inference Pipeline & MLOps Suite

> **Stack:** Python · TensorFlow · OpenCV · Albumentations · NumPy · Tkinter

> [!NOTE]
> **Proprietary Notice & Code Availability:**
> Production code, model weights, and raw video recordings are property of the university and research laboratory. This document serves as a high-level engineering walkthrough of the system architecture, inference pipeline, model design, and training methodology.

---

## TLDR

Using convolutional neural networks (CNNs) to track pupil keypoints with sub-pixel precision across datasets consisting of thousands of video recordings imposes strict throughput and precision constraints on the analysis pipeline. While the industry-standard tool, DeepLabCut (DLC), delivers high tracking quality, its performance was severely CPU-bottlenecked on our analysis workstation (AMD Ryzen 9 7950X, NVIDIA RTX 4090), running at only 40–60 FPS. In a research context where datasets are frequently re-analyzed, a single re-analysis run taking over a week was not viable.

To alleviate this bottleneck, I developed a custom CNN training and inference suite in Python/TensorFlow that accelerated end-to-end processing throughput from **40–60 FPS to ~300 FPS (~5x speedup)**. This was achieved by replacing DLC’s generic ~25M parameter ResNet50 with a custom lightweight 2-stage Hourglass architecture and decoupling video decoding from model inference using a multi-threaded pipeline. The system reduced full-dataset turnaround from **approximately a week to a single day** with no loss in tracking quality.

**Core System Architecture:**
- **Asynchronous Video Decoding:** A multi-threaded video reader that overlaps disk reads and frame preprocessing with GPU inference, eliminating the idle wait times of single-threaded pipelines.
- **2-Stage Localization & Tracking:** A small Finder model locates the pupil on downscaled frames, followed by a Placer Hourglass model running on a native-resolution crop, preserving fine spatial detail without downsampling the frame.
- **Sub-Pixel Post-Processing:** Converts heatmap outputs into continuous sub-pixel coordinates using local Center-of-Mass calculations.
- **Custom Dataset & Training Suite:** Solved the cold-start data requirement of training a non-pretrained model from scratch by generating training sets from high-confidence prior DLC runs, paired with a custom Python GUI for rapid inspection and edge-case correction.

---

This project required solving **three core engineering challenges**:

1. **Inference Throughput:** Overcoming the 40–60 FPS CPU-bound processing bottleneck of DeepLabCut to enable rapid re-analysis of video datasets exceeding 10,000 recordings.
2. **Spatial Precision Without Heavy Backbones:** Resolving subtle sub-pixel dynamics without downsampling artifacts, while avoiding overparameterized ~25M parameter general-purpose backbones.
3. **Cold-Start Training Data Generation:** Developing a method to produce large-scale, high-quality training data for a custom, non-pretrained network.

---

## 1. Inference Throughput: Lightweight Architecture & Async Video Decoding

On our analysis workstation (AMD Ryzen 9 7950X, NVIDIA RTX 4090), DeepLabCut's processing throughput was capped at 40–60 FPS by two bottlenecks: single-threaded CPU video decoding and an overparameterized ~25M parameter ResNet50 backbone. To eliminate these inefficiencies, I developed a custom pipeline centered around a task-specific neural network and an asynchronous ingestion architecture.

- **Task-Specific Hourglass Architecture:** Replaced DLC's generic ~25M parameter ResNet50 with a custom-designed Hourglass U-Net (~1M parameters) built from scratch for pupil keypoint regression. The compact architecture reduced parameter count by ~96%, executing batched GPU inference in a fraction of the time.
- **Asynchronous Video Reader:** Built a dedicated worker thread that decodes raw video frames in the background and pushes them into a thread-safe queue.
- **Concurrency via GIL Release:** Because OpenCV's video decoding is implemented in C++, it explicitly releases Python's Global Interpreter Lock (GIL) during frame decompression, allowing CPU-heavy decoding to execute concurrently with GPU batch inference.
- **Pipelined GPU Batching:** The main execution thread pulls pre-decoded frames from the queue, batches them, and dispatches them directly to the RTX 4090 with zero idle wait time.

**Outcome:** Combined, these optimizations delivered a **~5x end-to-end throughput speedup** (from 40–60 FPS to ~300 FPS), reducing multi-thousand video dataset turnaround from **approximately a week to a single day**.

---

## 2. Tracking Precision: 2-Stage Architecture & Sub-Pixel Regression

Measuring subtle pupil dynamics requires resolving diameter changes on the order of tens of micrometers. A traditional single-stage pipeline presents a difficult tradeoff:
1. **Downsampling the full frame** to a manageable network size (e.g., 224×224) blurs fine boundary edges and causes tracking jitter.
2. **Processing the full 1080p frame** at native resolution avoids this downsampling loss, but significantly decreases inference throughput.

To resolve this tradeoff, I implemented a 2-stage inference architecture:
1. **Global Localization (Finder):** A lightweight Hourglass model processes a heavily downscaled global frame (64×64) to predict the coarse pupil location in the frame.
2. **Cropping Without Downsampling:** An unscaled square patch (e.g., 192×192) centered on the pupil is cropped directly from the raw 1080p frame. Because no resizing is applied, the cropped pupil retains 100% of its native resolution.
3. **High-Precision Tracking (Placer):** A second Hourglass U-Net predicts 8 perimeter keypoints and 1 corneal IR reflection directly on the unscaled crop.
4. **Sub-Pixel Center of Mass (CoM):** Heatmap activations are converted to continuous sub-pixel coordinates using a local Center-of-Mass calculation.

```
Full 1080p Video Frame
          │
          ▼ (Downscale to 64×64)
Finder Model (Global Localization)
          │
          ▼ (Coarse Pupil Coordinates)
Extract Native-Resolution Crop from Raw Frame (e.g., 192×192)
          │
          ▼ (Zero-Loss Native Pixels)
Placer Model (Hourglass U-Net Keypoint Regression)
          │
          ▼ (Sigmoid Heatmaps)
Local Center-of-Mass (Sub-Pixel Coordinates)
```

**Outcome:** Preserved 100% native pixel density on key anatomical landmarks, matching baseline DLC sub-pixel precision while operating at **~300 FPS**.

---

## 3. Training & Data: Ground-Truth Generation & Training Suite

Training a custom Hourglass network from scratch without pre-trained ImageNet weights required a diverse training dataset. Manually annotating thousands of video frames across different animals, pupil dilation states, and infrared lighting variations would have required significant repetitive labor.

To avoid the need for excessive manual annotation, I focused on two areas:

**Generating Ground Truth from Pre-Existing Data**
- **Converting Legacy DLC Outputs:** To leverage pre-existing analysis, I wrote scripts to extract high-confidence predictions ($p > 0.95$) from historical DeepLabCut runs across thousands of archived videos, converting past outputs into training annotations for the custom model.
- **Diversity Filtering:** I used K-means clustering on frame representations to filter out visually repetitive frames, ensuring the dataset covered extreme pupil dilations, blinks, and variable contrast.
- **In-House Review GUI:** I developed a custom Python/Tkinter desktop tool to inspect extracted frames, correct edge cases, and manually label additional samples when needed.

**Training Workflow & Augmentations**
- **Single Unified Dataset for Both Models:** A single ground-truth dataset (full raw frames + keypoint coordinates) was sufficient to train both models. When training the Finder model, the full frame was used as input; when training the Placer model, the data loader dynamically cropped the region of interest around the pupil, eliminating the need to maintain or store duplicate datasets.
- **Comprehensive Data Augmentation:** I implemented a comprehensive augmentation pipeline using Albumentations to cover likely variations in the recordings (such as brightness shifts from IR LED intensity changes, optical blur from defocus artifacts, and elastic tissue movement), ensuring the models remained robust during production inference.

**Outcome:** Streamlined dataset curation and model verification into an automated workflow, enabling custom models to be trained from scratch without requiring extensive manual annotation.
