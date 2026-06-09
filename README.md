# COM-4504 Digital Image Processing — Group Project
## Drone Surveillance: Multi-Class Object Detection & Tracking on VisDrone 2019

**Course:** COM-4504 Digital Image Processing  
**Dataset:** [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) — 10 classes, 6,471 train + 548 val images  
**Task:** Multi-class object detection and multi-object tracking from drone footage

---

## Overview

This project benchmarks three approaches for detecting and tracking objects (pedestrians, vehicles, cyclists, etc.) in aerial drone imagery, progressing from classical machine learning to a full deep learning pipeline.

| Method | Description | mAP@50 | FPS |
|---|---|---|---|
| **Method A** — HOG + SVM | Classical ML baseline | — (60.73% clf. acc.) | ~2 |
| **Method B** — YOLOv8n | Deep learning baseline | 41.6% | ~45 |
| **Method C** — YOLOv8s + SAHI + ByteTrack | Proposed full pipeline | 38.71% | 40.8 |

> Method C trades slightly lower static mAP for robust real-time multi-object tracking (MOTA=62.1%, IDF1=67.8%) that the other methods do not provide.

---

## Repository Structure

This repository has two feature branches, one per team member:

### `feature/yolov8s-sahi-bytetrack` — Merve Çakır (21290542)
Full end-to-end detection + tracking pipeline:
- YOLOv8s trained 50 epochs on VisDrone (mAP@50 = 38.71%)
- SAHI (Slicing Aided Hyper Inference) for small object detection — 640×640 slices, 20% overlap
- ByteTrack multi-object tracker — MOTA=62.1%, IDF1=67.8%, 89 ID switches
- Pipeline profiling, error analysis, hyperparameter grid search
- 9 Colab notebooks documenting day-by-day development

### `feature/hog-svm-yolov8n-baseline` — Emine Binay (22290370)
Classical ML and deep learning baselines:
- HOG + SVM: 64×64 crops, 3780-dim feature vector, RBF kernel — 60.73% accuracy
- YOLOv8n: lightweight baseline (3.0M params), 30 epochs — mAP@50=41.6%
- Comparison against SORT and DeepSORT trackers

---

## Key Results

**Detection (val set):**

| | HOG+SVM | YOLOv8n | YOLOv8s+SAHI+ByteTrack |
|---|---|---|---|
| mAP@50 | N/A | 41.6% | 38.71% |
| Precision | — | 41.1% | 52.1% |
| Recall | — | 31.1% | 40.0% |
| F1 | — | 35.5% | 45.2% |
| FPS | ~2 | ~45 | 40.8 (1.8 w/ SAHI) |

**Tracking (ByteTrack vs. baselines):**

| Tracker | MOTA | IDF1 | ID Switches |
|---|---|---|---|
| SORT | 42.3% | — | 312 |
| DeepSORT | 51.2% | — | 198 |
| **ByteTrack** | **62.1%** | **67.8%** | **89** |

---

## Paper

A joint IEEE-format conference paper covering all three methods is available in the `feature/yolov8s-sahi-bytetrack` branch:
- `IEEE_DIP_Paper_CakirBinay.docx` — full paper (Abstract through Conclusion + References)
- `IEEE_Results_Section.docx` — Results section with all tables and embedded figures

---

## Requirements

```bash
pip install ultralytics sahi supervision scikit-learn python-docx
```

GPU recommended (Tesla T4 or equivalent). Notebooks designed for Google Colab.

Presentation Link : https://drive.google.com/drive/folders/1F-BoQjC2m1d3hm0D_OhK4VDlx8PS_cHd?usp=sharing
