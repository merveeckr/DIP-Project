# Experiment Logs
## COM 4504 Digital Image Processing — Emine Binay (22290370)
## Real-Time Object Detection in Aerial Images (UAV) — Baseline Tier
---

## 1. Hardware Configuration

| Component | Specification |
|-----------|--------------|
| **Platform** | Kaggle Notebooks (Cloud) |
| **GPU** | Tesla T4 x2 |
| **VRAM per GPU** | 14,913 MiB (≈ 15.6 GB) |
| **Total VRAM** | 29,826 MiB (≈ 30 GB) |
| **CUDA Version** | 12.2 |
| **Driver Version** | 580.105.08 |
| **CPU** | Kaggle default (4-core) |
| **RAM** | ~30 GB (Kaggle default) |
| **Storage** | ~19.5 GB output quota |
| **OS** | Linux (Ubuntu 22.04) |
| **Python** | 3.12 |

---

## 2. Dataset Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | VisDrone2019-DET |
| **Train images** | 6,471 |
| **Train instances** | 343,205 |
| **Validation images** | 548 |
| **Number of classes** | 10 |
| **Class names** | pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor |
| **Annotation format** | YOLO (normalized center coordinates) |
| **Image resolution** | Variable (960×540 to 2000×1500) |
| **Tiny objects (<32px)** | ~59.8% of all instances |

---

## 3. HOG+SVM Baseline Hyperparameters (Tier 1)

### HOG Feature Extraction
| Parameter | Value | Description |
|-----------|-------|-------------|
| `patch_size` | 64×64 px | Input patch size |
| `orientations` | 9 | Number of gradient orientation bins |
| `pixels_per_cell` | (8, 8) | Size of each cell |
| `cells_per_block` | (2, 2) | Number of cells per block |
| `color_space` | Grayscale | Input color space |
| `feature_vector` | True | Return flattened feature vector |

### SVM Classifier
| Parameter | Value | Description |
|-----------|-------|-------------|
| `classifier` | LinearSVC | Linear Support Vector Classifier |
| `C` | 0.1 | Regularization parameter |
| `max_iter` | 2000 | Maximum iterations |
| `multi_class` | ovr | One-vs-rest strategy |

### Training Details
| Parameter | Value |
|-----------|-------|
| `train_images_used` | 500 (subset) |
| `train_samples` | 35,045 patches |
| `val_samples` | 4,604 patches |
| `training_time` | 181.8 seconds (CPU) |
| `hardware` | CPU only |

---

## 4. YOLOv8n Baseline Training Hyperparameters (Tier 2)

### Model Architecture
| Parameter | Value |
|-----------|-------|
| `model` | YOLOv8n |
| `parameters` | 3,012,798 (~3.0M) |
| `GFLOPs` | 8.2 |
| `architecture` | Anchor-free, single-stage |
| `pretrained_weights` | COCO (ImageNet) |

### Training Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 30 | Total training epochs |
| `batch_size` | 16 | Images per batch (32 caused OOM) |
| `imgsz` | 640 | Input image size (640×640 px) |
| `optimizer` | AdamW | Adaptive optimizer |
| `lr0` | 0.000714 | Initial learning rate |
| `momentum` | 0.9 | Optimizer momentum |
| `weight_decay` | 0.0005 | L2 regularization |
| `warmup_epochs` | 3 | Warmup period |
| `warmup_momentum` | 0.8 | Warmup momentum |
| `patience` | 100 | Early stopping patience (not triggered) |
| `device` | 0 | GPU device index |
| `workers` | 8 | Dataloader worker threads |

### Augmentation Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `mosaic` | 1.0 | Mosaic augmentation probability |
| `close_mosaic` | 10 | Disable mosaic last N epochs |
| `flipud` | 0.0 | Vertical flip probability |
| `fliplr` | 0.5 | Horizontal flip probability |
| `degrees` | 0.0 | Rotation degrees |
| `translate` | 0.1 | Translation fraction |
| `scale` | 0.5 | Scale gain |
| `hsv_h` | 0.015 | HSV hue augmentation |
| `hsv_s` | 0.7 | HSV saturation augmentation |
| `hsv_v` | 0.4 | HSV value augmentation |
| `erasing` | 0.4 | Random erasing probability |

### Random Erasing (Custom)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `p` | 0.5 | Application probability |
| `sl` | 0.02 | Minimum erased area ratio |
| `sh` | 0.40 | Maximum erased area ratio |
| `r1` | 0.3 | Aspect ratio range min |
| `fill` | Random noise | Fill strategy |

### Training Results
| Metric | Value |
|--------|-------|
| `mAP@50 (final val)` | 41.6% |
| `mAP@50:95 (final val)` | 28.3% |
| `precision` | 41.1% |
| `recall` | 31.1% |
| `training_time` | ~42 minutes |
| `best_epoch` | 30 (no early stopping) |
| `model_size` | 6.2 MB |

---

## 5. SAHI Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `slice_height` | 640 | Patch height in pixels |
| `slice_width` | 640 | Patch width in pixels |
| `overlap_height_ratio` | 0.2 | Vertical overlap between patches |
| `overlap_width_ratio` | 0.2 | Horizontal overlap between patches |
| `confidence_threshold` | 0.25 | Minimum detection confidence |
| `device` | cuda | Inference device |
| `nms_iou_threshold` | 0.5 | IoU threshold for NMS fusion |

### SAHI Results
| Mode | Avg. Detections/Image | FPS |
|------|-----------------------|-----|
| SAHI-off (baseline) | 45.6 | 15.3 |
| SAHI-on (640×640, 20%) | 107.4 | 3.7 |

---

## 6. ByteTrack Optimal Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `track_activation_threshold` | 0.25 | Min confidence to activate track |
| `lost_track_buffer` | 30 | Frames to keep lost track (1s @ 30FPS) |
| `minimum_matching_threshold` | 0.80 | Min IoU for track association |
| `min_box_area` | 10.0 | Min bounding box area in px² |
| `frame_rate` | 30 | Expected video frame rate |

### Tracker Comparison Results
| Tracker | MOTA | IDF1 | ID Switches | FPS |
|---------|------|------|-------------|-----|
| SORT | 42.3% | 47.8% | 312 | 28.5 |
| DeepSORT | 51.2% | 57.4% | 198 | 18.2 |
| ByteTrack (optimal) | **62.1%** | **67.8%** | **89** | **35.6** |

---

## 7. Pipeline Latency Profile

All measurements: 5-run GPU warmup + 20-run average, torch.cuda.synchronize()

| Component | Mean Latency | Std Dev | FPS | VRAM |
|-----------|-------------|---------|-----|------|
| Preprocessing (resize 640) | 1.51 ms | ±0.08 ms | 662 | 94.8 MB |
| YOLOv8n Inference | 24.16 ms | ±1.79 ms | 41.4 | 151.6 MB |
| NMS Post-processing | 0.09 ms | ±0.01 ms | >10,000 | — |
| **Full Pipeline (no SAHI)** | **25.77 ms** | **±1.11 ms** | **38.8** | **152.5 MB** |
| YOLOv8n + SAHI | 243.70 ms | ±95.4 ms | 4.1 | 167.2 MB |

---

## 8. Epoch-by-Epoch Training Log (YOLOv8n)

| Epoch | mAP@50 | mAP@50:95 | Train Box Loss | Train Cls Loss | Train DFL Loss |
|-------|--------|-----------|---------------|---------------|---------------|
| 1 | 11.72% | 6.12% | 1.966 | 2.432 | 1.028 |
| 5 | 19.89% | 10.79% | 1.724 | 1.365 | 0.956 |
| 10 | 21.97% | 12.14% | 1.647 | 1.240 | 0.936 |
| 15 | 25.81% | 14.23% | 1.611 | 1.179 | 0.924 |
| 20 | 27.00% | 14.93% | 1.580 | 1.132 | 0.917 |
| 25 | 27.49% | 15.33% | 1.483 | 1.053 | 0.910 |
| 30 | 28.27% | 15.81% | 1.451 | 1.013 | 0.903 |
| **Val (best.pt)** | **41.6%** | **28.3%** | — | — | — |

Full epoch log available in `results/results.csv`

---

## 9. Output Files

| File | Description |
|------|-------------|
| `model/best.pt` | Best YOLOv8n checkpoint (6.2 MB) |
| `results/results.csv` | Full epoch-by-epoch training metrics |
| `results/hog_svm_results.json` | HOG+SVM classification results |
| `figures/results.png` | Training curves (loss + mAP) |
| `figures/BoxP_curve.png` | Precision-Confidence curve |
| `figures/BoxR_curve.png` | Recall-Confidence curve |
| `figures/BoxF1_curve.png` | F1-Confidence curve |
| `figures/confusion_matrix_normalized.png` | Normalized confusion matrix |
| `figures/random_erasing_examples.png` | Augmentation visualization |
