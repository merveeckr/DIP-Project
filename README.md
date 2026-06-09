# HOG+SVM & YOLOv8n Baseline — Emine Binay
## COM 4504 Digital Image Processing | Ankara University | 2026

---

## Overview

This notebook implements and evaluates the **Tier 1 (Classical ML)** and **Tier 2 (Baseline Deep Learning)** components of the three-tier methodological framework developed for real-time aerial object detection on the VisDrone2019 dataset.

The work covers the complete pipeline from dataset exploration and preprocessing through classical machine learning baseline, deep learning training, SAHI integration, and multi-tracker comparison — forming the experimental foundation for the comparative analysis in the joint project report.

**Student:** Emine Binay | ID: 22290370  
**Partner:** Merve Çakır | ID: 21290542 — [Partner's Repository](https://github.com/merveeckr/DIP-Project)  
**Supervisor:** Asst. Prof. Dr. Feyza Toktaş  
**Platform:** Kaggle Notebooks (Tesla T4 x2 GPU)

---

## Repository Contents

```
hog-svm-yolov8n-baseline/
├── hog-svm-yolov8n-baseline.ipynb   ← Main notebook (this file)
├── figures/                          ← All output figures
│   ├── BoxF1_curve.png               Training F1-Confidence curve
│   ├── BoxP_curve.png                Training Precision-Confidence curve
│   ├── BoxPR_curve.png               Precision-Recall curve
│   ├── BoxR_curve.png                Recall-Confidence curve
│   ├── confusion_matrix.png          Absolute confusion matrix
│   ├── confusion_matrix_normalized.png  Normalized confusion matrix
│   ├── labels.png                    Dataset annotation distribution
│   ├── random_erasing_examples.png   Augmentation visualization
│   ├── results.png                   Training curves (30 epochs)
│   ├── train_batch0/1/2.jpg          Early training batch examples
│   ├── train_batch8100/8101/8102.jpg Late training batch examples
│   └── val_batch0/1/2_labels/pred.jpg  Validation predictions vs GT
├── results/
│   ├── hog_svm_results.json          HOG+SVM classification metrics
│   └── results.csv                   Epoch-by-epoch training log
└── model/
    └── best.pt                       Best YOLOv8n checkpoint
```

---

## What This Notebook Does

### Step 1 — Environment Setup and Dataset Exploration
- GPU verification (Tesla T4, 14,913 MiB VRAM)
- VisDrone2019-DET dataset path resolution on Kaggle
- Dataset structure analysis: train (6,471 images), val (548 images)
- Annotation format verification — confirms YOLO-format labels already present

### Step 2 — Class Distribution Analysis
- Per-class object count across all 10 VisDrone categories
- Visualization of class imbalance: car (144,867) vs. awning-tricycle (3,246)
- Output: `figures/labels.png`, class distribution histogram

### Step 3 — Random Erasing Augmentation
- Implementation from scratch following Zhong et al. (AAAI 2020)
- Parameters: p=0.5, area ratio 0.02–0.40, aspect ratio 0.3–3.33
- Noise-based pixel replacement to simulate partial occlusion
- Output: `figures/random_erasing_examples.png`

### Step 4 — HOG+SVM Classical ML Baseline (Tier 1)
- HOG feature extraction: 9 orientations, 8×8 cells, 2×2 blocks, 64×64 patches
- LinearSVC classifier: C=0.1, max_iter=2000
- Training: 35,045 patches from 500 training images
- Validation: 4,604 patches from 100 validation images
- Result: **60.73% overall accuracy** (patch-level classification only)
- Output: `results/hog_svm_results.json`

### Step 5 — YOLOv8n Baseline Training (Tier 2, 30 Epochs)
- Pre-trained COCO weights as initialization (transfer learning)
- Training config: batch=16, imgsz=640, AdamW lr=0.000714, warmup=3 epochs
- Mosaic augmentation (p=1.0), disabled for final 10 epochs (close_mosaic=10)
- Hardware: Tesla T4 GPU, ~42 minutes total training time
- Result: **41.6% mAP@50, 28.3% mAP@50:95** on VisDrone2019 validation
- Outputs: `figures/results.png`, `figures/BoxP/R/F1_curve.png`, `results/results.csv`, `model/best.pt`

### Step 6 — SAHI Integration and Comparison
- SAHI configuration: 640×640 patches, 20% overlap, conf=0.25
- Comparison against SAHI-off baseline on 5 high-density validation images
- Result: **+135% average detections** (45.6 → 107.4 per image) at cost of throughput (15.3 → 3.7 FPS)

### Step 7 — Pipeline Latency Profiling
- Component-level timing: preprocessing, YOLOv8n inference, NMS, SAHI
- 5-run warmup + 20-run averaging with torch.cuda.synchronize()
- Result: **Full pipeline 25.77 ms → 38.8 FPS** (exceeds 25 FPS real-time target)

### Step 8 — Multi-Tracker Comparison (SORT vs DeepSORT vs ByteTrack)
- 50-frame validation sequence evaluation
- Metrics: MOTA, IDF1, ID switches, FPS
- Result: **ByteTrack best** — MOTA 62.1%, IDF1 67.8%, 89 ID switches, 35.6 FPS

---

## Key Results

### HOG+SVM vs. YOLOv8n

| Method | Accuracy / mAP@50 | Bounding Box | FPS | Notes |
|--------|-------------------|--------------|-----|-------|
| HOG+SVM | 60.73% (patch-level) | ✗ No | Slow (CPU) | Patch classification only |
| YOLOv8n (30 epochs) | **41.6% mAP@50** | ✓ Yes | **38.8** | Full detection |

### YOLOv8n Per-Class Performance

| Class | mAP@50 | Difficulty |
|-------|--------|------------|
| car | 56.3% | Low (large, frequent) |
| bus | 51.6% | Medium |
| pedestrian | 41.1% | High (tiny) |
| motor | 41.1% | Medium |
| bicycle | 23.6% | Very High (tiny, rare) |
| awning-tricycle | 23.1% | Very High (rarest) |

### SAHI Comparison (5 Images)

| Mode | Avg. Detections | FPS |
|------|----------------|-----|
| Without SAHI | 45.6 | 15.3 |
| With SAHI (640×640, 20%) | **107.4** | 3.7 |
| **Improvement** | **+135%** | **-76%** |

### Pipeline Latency

| Component | Latency | FPS |
|-----------|---------|-----|
| Preprocessing | 1.51 ms | 662 |
| YOLOv8n Inference | 24.16 ms | 41.4 |
| NMS | 0.09 ms | >10,000 |
| **Full Pipeline** | **25.77 ms** | **38.8 ✓** |

### Tracker Comparison

| Tracker | MOTA | IDF1 | ID Switches | FPS |
|---------|------|------|-------------|-----|
| SORT | 42.3% | 47.8% | 312 | 28.5 |
| DeepSORT | 51.2% | 57.4% | 198 | 18.2 |
| **ByteTrack** | **62.1%** | **67.8%** | **89** | **35.6** |

---

## How to Run

1. Open `hog-svm-yolov8n-baseline.ipynb` on [Kaggle](https://www.kaggle.com)
2. Add the following datasets as input:
   - `banuprasadb/visdrone-dataset` (VisDrone2019-DET)
   - `eminebnay/emine-dip-model2` (pre-trained YOLOv8n weights)
3. Set accelerator to **GPU T4 x2**
4. Run all cells sequentially

---

## Environment

| | |
|--|--|
| Platform | Kaggle Notebooks |
| GPU | Tesla T4 x2 (14,913 MiB each) |
| CUDA | 12.2 |
| Python | 3.12 |
| ultralytics | 8.x |
| sahi | 0.11.x |
| scikit-learn | latest |
| albumentations | 1.3.x |
| supervision | latest |

---

## Dataset

**VisDrone2019-DET** — AISKYEYE Team, Tianjin University

| Split | Images | Instances |
|-------|--------|-----------|
| Train | 6,471 | 343,205 |
| Validation | 548 | ~29,000 |

10 categories: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

---

## References

| | |
|--|--|
| [1] | Zhu et al. (2021) — VisDrone Dataset, IEEE TPAMI |
| [2] | Akyon et al. (2022) — SAHI, IEEE ICIP |
| [3] | Zhang et al. (2022) — ByteTrack, ECCV |
| [4] | Dalal & Triggs (2005) — HOG+SVM, CVPR |
| [5] | Bewley et al. (2016) — SORT, IEEE ICIP |
| [6] | Wojke et al. (2017) — DeepSORT, IEEE ICIP |
| [7] | Jocher et al. (2023) — YOLOv8, Ultralytics |
| [8] | Zhong et al. (2020) — Random Erasing, AAAI |
