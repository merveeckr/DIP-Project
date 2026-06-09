# COM-4504 Digital Image Processing — Merve Çakır (21290542)
## YOLOv8s + SAHI + ByteTrack: Drone Surveillance Object Detection & Tracking

Bu repo, COM-4504 dersi kapsamında geliştirilen **drone görüntülerinde çok sınıflı nesne tespiti ve takip** projesinin tüm kodlarını, notebook'larını ve çıktı dosyalarını içerir.  
Veri kümesi: [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) — 10 sınıf, 6.471 eğitim + 548 doğrulama görüntüsü.

---


## Sonuçlar (Özet)

| Metrik | Değer |
|---|---|
| mAP@50 (YOLOv8s, no SAHI) | **38.71%** |
| Precision / Recall / F1 | 52.1% / 40.0% / **45.2%** |
| FPS (YOLO + ByteTrack, no SAHI) | **40.8 FPS** |
| FPS (YOLO + SAHI + ByteTrack) | **1.8 FPS** |
| MOTA (ByteTrack) | **62.1%** |
| IDF1 (ByteTrack) | **67.8%** |
| GPU VRAM kullanımı | 152.5 MB / 15.6 GB (%0.98) |

---

## Proje Yapısı

```
DIP_Project/
│
├── notebooks/               # Günlük çalışma notebook'ları (Google Colab)
├── runs/                    # YOLOv8 eğitim çıktıları (weights, metrics)
├── datasets/VisDrone/       # Veri kümesi (images/train, val, test-dev)
├── sahi_comparison/         # SAHI konfigürasyon karşılaştırma çıktıları
├── error_analysis/          # FP/FN görselleri
├── tracking_output/         # ByteTrack ile üretilen video/frame çıktıları
│
├── [Ana Python Scriptleri]
├── [Veri/Konfigürasyon Dosyaları]
└── [Görseller ve Raporlar]
```

---

## Kurulum ve Çalıştırma

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

> Google Colab kullanıyorsanız `torch` ve `torchvision` zaten kurulu gelir.  
> GPU: NVIDIA Tesla T4 (veya eşdeğeri, min. 4 GB VRAM) | Python: 3.10+

---

### 2. Veri Kümesi Hazırlığı

VisDrone 2019 veri kümesini indirip aşağıdaki yapıya yerleştirin:

```
datasets/VisDrone/
├── images/
│   ├── train/     # 6,471 görüntü
│   ├── val/       # 548 görüntü
│   └── test-dev/
└── labels/
    ├── train/
    └── val/
```

`data.yaml` dosyasındaki `path` satırını kendi dizininize göre güncelleyin:

```yaml
path: /content/drive/MyDrive/DIP_Project/datasets/VisDrone  # Colab
# path: ./datasets/VisDrone                                  # Yerel
```

---

### 3. Model Eğitimi

```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    optimizer='SGD',
    lr0=0.01,
    mosaic=1.0,
    close_mosaic=10,
    project='runs',
    name='yolov8s_visdrone_v1'
)
"
```

> Not: `batch=32` Tesla T4'te CUDA OOM hatasına yol açar, `batch=16` kullanın.  
> Eğitim çıktısı: `runs/yolov8s_visdrone_v1/weights/best.pt`

Adım adım notebook versiyonu: `notebooks/day5_training.ipynb`

---

### 4. Model Değerlendirme (Test)

```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/yolov8s_visdrone_v1/weights/best.pt')
metrics = model.val(data='data.yaml', imgsz=640, split='val')
print('mAP@50:', metrics.box.map50)
print('Precision:', metrics.box.p)
print('Recall:', metrics.box.r)
"
```

Detaylı değerlendirme notebook'u: `notebooks/day11_evaluation.ipynb`

---

### 5. SAHI ile Çıkarım (Küçük Nesne Tespiti)

```bash
python -c "
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='runs/yolov8s_visdrone_v1/weights/best.pt',
    confidence_threshold=0.25,
    device='cuda'
)

result = get_sliced_prediction(
    'path/to/image.jpg',
    model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
result.export_visuals(export_dir='sahi_output/')
"
```

SAHI konfigürasyon karşılaştırması: `notebooks/day12_sahi_comparison.ipynb`

---

### 6. Video Üzerinde Takip (YOLOv8s + ByteTrack)

```bash
python track_video.py \
    --source 607004_Cities_City_3840x2160.mp4 \
    --model runs/yolov8s_visdrone_v1/weights/best.pt \
    --output tracking_output/result.mp4
```

Seçili frame'leri PNG olarak kaydetmek için:

```bash
python tracking_demo.py
```

ByteTrack hiperparametre optimizasyonu: `notebooks/day8_bytetrack_hyperparams.ipynb`

---

### 7. Hızlı Demo (Tüm Pipeline)

Tüm aşamaları sırayla çalıştırmak için notebook'ları şu sırayla açın:

| Sıra | Notebook | Aşama |
|---|---|---|
| 1 | `day1_setup.ipynb` | Kurulum ve veri hazırlığı |
| 2 | `day5_training.ipynb` | YOLOv8s eğitimi |
| 3 | `day11_evaluation.ipynb` | Model değerlendirme |
| 4 | `day12_sahi_comparison.ipynb` | SAHI konfigürasyon seçimi |
| 5 | `day7_bytetrack.ipynb` | ByteTrack entegrasyonu |
| 6 | `day8_bytetrack_hyperparams.ipynb` | Hiperparametre optimizasyonu |
| 7 | `day13_profiling.ipynb` | Performans profilleme |
| 8 | `day14_error_analysis.ipynb` | Hata analizi |

---

### 8. Rapor Üretimi

```bash
# Tam IEEE makalesi
python create_ieee_paper.py
# Çıktı: IEEE_DIP_Paper_CakirBinay.docx

# Sonuçlar bölümü (tüm tablolar + 14 figür gömülü)
python create_results_word.py
# Çıktı: IEEE_Results_Section.docx
```
## Konfigürasyon Dosyaları

| Dosya | İçerik |
|---|---|
| `data.yaml` | VisDrone veri kümesi tanımı — 10 sınıf ismi, train/val/test dizin yolları |
| `bytetrack_best_config.json` | Grid search sonucunda bulunan en iyi ByteTrack parametreleri (track_thresh=0.25, buffer=30, match_thresh=0.80, min_box_area=10) |
| `bytetrack_grid_results.csv` | 12 konfigürasyonun MOTA/IDF1/IDSw sonuçları |
| `model_paths.json` | Eğitilmiş model ağırlıklarının Google Drive yolları |
| `eval_results.json` | YOLOv8s val seti değerlendirme sonuçları (mAP, P, R) |
| `error_summary.json` | FP/FN sayıları ve küçük nesne oranları |
| `profiling_summary.json` | Her pipeline bileşeninin ms/FPS/VRAM ölçümleri |
| `per_class_ap.csv` | 10 sınıf için ayrı ayrı AP@50 değerleri |
| `sahi_comparison_summary.csv` | 5 SAHI konfigürasyonunun detection sayısı karşılaştırması |
| `error_records.csv` | Frame bazında FP/FN detayları (sınıf, boyut, IoU) |
| `profiling_table.csv` | `profiling_summary.json`'ın tablo formatı |

---

## Görseller

| Dosya | Ne Gösterir |
|---|---|
| `training_curves.png` | 50 epoch eğitim sürecinde mAP@50, box loss, cls loss eğrileri |
| `per_class_ap.png` | 10 sınıf için AP@50 bar grafiği (en iyi: car %78.1, en kötü: bicycle %12.5) |
| `augmentation_grid.png` | Mosaic augmentation örneği (4 görüntü → 1 eğitim örneği) |
| `mosaic_example.png` | Mosaic augmentation'ın farklı bir örneği |
| `sahi_comparison.png` | SAHI açık/kapalı detection karşılaştırması |
| `sahi_best_result.png` | En iyi SAHI konfigürasyonu (640×640, %20 overlap) ile örnek çıkarım |
| `slice_grid.png` | SAHI'nin bir görüntüyü nasıl 12 dilime böldüğünü gösteren ızgara |
| `bytetrack_hyperparam.png` | 12 ByteTrack konfigürasyonunun MOTA/IDF1 karşılaştırma grafiği |
| `bytetrack_tracking_figure.png` | ByteTrack ile takip edilen nesnelerin ID'leriyle birlikte gösterimi |
| `tracking_frames.png` | Farklı framelerde takip sonuçları (yan yana) |
| `tracking_stats.png` | MOTA, IDF1, IDSw bar grafikleri |
| `profiling_results.png` | Pipeline bileşenlerinin ms ve FPS karşılaştırması |
| `error_distribution.png` | FP/FN'lerin sınıf ve boyut dağılımı |
| `error_analysis/false_positives.png` | FP örnekleri — modelin yanlış tespit ettiği bölgeler |
| `error_analysis/false_negatives.png` | FN örnekleri — modelin kaçırdığı küçük nesneler |

---


## Model Ağırlıkları (`runs/`)

```
runs/
└── yolov8s_visdrone_v1/
    └── weights/
        ├── best.pt      # En iyi validation mAP'e sahip checkpoint
        └── last.pt      # Son epoch checkpoint'i
```

Eğitim konfigürasyonu: `batch=16`, `imgsz=640`, `epochs=50`, `optimizer=SGD`, `lr0=0.01`, `mosaic=1.0` (ilk 40 epoch), `close_mosaic=10`.

---

## Pipeline Genel Akışı

```
Video Frame
    ↓
Preprocessing (resize 640×640)   →  1.44 ms
    ↓
YOLOv8s Inference                → 13.13 ms
    ↓
[Opsiyonel] SAHI (12 dilim)      → 541 ms  (1.8 FPS)
    ↓
NMS (Non-Maximum Suppression)    →  0.09 ms
    ↓
ByteTrack (2-stage matching)     →  9.47 ms
    ↓
Annotated Frame / Video          → Toplam: 24.5 ms = 40.8 FPS (SAHI kapalı)
```

---

## Gereksinimler

requirements.txt dosyasında gerekli gereksinimler mevcuttur.

GPU: NVIDIA Tesla T4 (veya eşdeğeri, min. 4 GB VRAM)  
Python: 3.10+  
Platform: Google Colab (önerilen) veya yerel CUDA ortamı

---
