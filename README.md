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

## Notebook'lar (`notebooks/`)

Her notebook, projenin belirli bir aşamasını temsil eder. Google Colab'da Tesla T4 GPU ile çalıştırılmıştır.

| Dosya | Ne Yapar |
|---|---|
| `day1_setup.ipynb` | Ortam kurulumu — Ultralytics, SAHI, ByteTrack yükleme; VisDrone veri kümesini Google Drive'a bağlama ve `data.yaml` oluşturma |
| `day2_sahi_test.ipynb` | SAHI'nin temel mantığını test etme — tek bir görüntü üzerinde dilim (slice) tabanlı çıkarım; slice sayısı ve overlap oranının etkisini gözlemleme |
| `day5_training.ipynb` | YOLOv8s modelini VisDrone üzerinde 50 epoch eğitme (batch=16, imgsz=640, mosaic augmentation); eğitim eğrilerini kaydetme |
| `day7_bytetrack.ipynb` | ByteTrack entegrasyonu — YOLOv8s tespitlerini ByteTrack ile birleştirip video üzerinde çok nesne takibi yapma; MOTA/IDF1/IDSw hesaplama |
| `day8_bytetrack_hyperparams.ipynb` | ByteTrack hiperparametre grid search — 12 farklı konfigürasyon (track_thresh, buffer, match_thresh kombinasyonları) denenerek en iyi config belirleme |
| `day11_evaluation.ipynb` | Model değerlendirme — VisDrone val seti üzerinde mAP@50, P, R, F1 hesaplama; per-class AP analizi; sonuçları JSON/CSV'ye kaydetme |
| `day12_sahi_comparison.ipynb` | SAHI konfigürasyon karşılaştırması — 5 farklı slice boyutu ve overlap oranı denenerek en iyi SAHI ayarını bulma (640×640, %20 overlap) |
| `day13_profiling.ipynb` | Performans profilleme — her pipeline bileşeninin (preprocessing, YOLO, SAHI, ByteTrack) ms ve FPS ölçümü; VRAM kullanımı; sonuçları JSON/CSV'ye kaydetme |
| `day14_error_analysis.ipynb` | Hata analizi — FP (yanlış pozitif) ve FN (kaçırılan nesne) tespitleri görselleştirme; küçük nesne sorunu analizi (%73.1 FN küçük nesneler) |

---

## Ana Python Scriptleri

### `track_video.py`
**Ne yapar:** Bir video dosyası üzerinde tam pipeline'ı çalıştırır — YOLOv8s ile nesne tespiti + ByteTrack ile takip — ve sonucu annotated video olarak kaydeder.

**Nasıl çalışır:**
1. Video frame'lerini okur
2. Her frame'e YOLOv8s inference uygular (conf ≥ 0.25)
3. Tespitleri ByteTrack'e iletir (iki aşamalı Hungarian eşleştirme)
4. Her nesneye kalıcı ID atar, bounding box + ID'yi frame üzerine çizer
5. Çıktıyı `tracking_output/` klasörüne yazar

**Çalıştırma:**
```bash
python track_video.py --source 607004_Cities_City_3840x2160.mp4 --output tracking_output/cities_tracked.mp4
```

---

### `tracking_demo.py`
**Ne yapar:** `track_video.py`'nin daha basit, demo amaçlı versiyonu. Belirli frame'leri (50, 150, 300, 450. frame) PNG olarak kaydeder ve takip istatistiklerini ekrana basar.

**Farkı:** Tam video yerine sadece seçili frame'leri diske yazar; hızlı görsel doğrulama için kullanılır.

**Çıktılar:** `tracking_output/detected_frame_*.jpg`

---

### `make_tracking_figure.py`
**Ne yapar:** Rapor için tracking görsellerini hazırlar — birden fazla frame'i yan yana birleştirerek tek bir figür oluşturur (`tracking_frames.png`). ByteTrack takip istatistiklerini bar grafiği olarak çizer (`tracking_stats.png`).

**Çıktılar:** `tracking_frames.png`, `tracking_stats.png`

---

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

```bash
pip install ultralytics sahi supervision python-docx
```

GPU: NVIDIA Tesla T4 (veya eşdeğeri, min. 4 GB VRAM)  
Python: 3.10+  
Platform: Google Colab (önerilen) veya yerel CUDA ortamı

---
