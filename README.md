# 🔬 Benchmarking Underexplored Small-Object Detection Models on Aerial Imagery

> **A systematic, fair-comparison study of 6 lightweight/underexplored object detection architectures on the VisDrone-DET dataset, with and without SAHI (Sliced-Inference) post-processing.**

---

## 📌 Project Overview

This research benchmarks **six underexplored small-object detection models** on the challenging **VisDrone2019-DET (Task 1)** dataset — a collection of dense UAV aerial images where objects average only ~22×22 pixels. The study evaluates each model in **two phases**: standard full-image inference (baseline) and SAHI-augmented sliced inference, providing reproducible, architecture-agnostic performance comparisons.

The full results and methodology are documented in an **IEEE-formatted research paper** and a series of Kaggle/Colab notebooks for every model.

---

## 🗂️ Repository Structure

```
 Models/
    ├── D-fine/
    │   ├── d-fine-kaggle (4).ipynb                          # D-FINE training pipeline (Kaggle)
    │   ├── d-fine-phase-2 (1).ipynb                         # D-FINE Phase 2 (SAHI inference)
    │   └── sod_pipeline_v2.html                             # Full SOD pipeline flowchart (visual)
    │
    ├── DAMO-YOLO/
    │   └── 50 Epoches Result/
    │       └── damo-yolo-visdrone-pipeline.ipynb            # DAMO-YOLO 50-epoch full pipeline
    │
    ├── GOLD-YOLO/
    │   ├── 50 Epoch/
    │   │   └── GOLD-YOLO-Visdrone_Phase1&2.ipynb            # GOLD-YOLO 50-epoch (Phase 1+2)
    │   └── 70 Epoch/
    │       └── gold-yolo-70_epoch.ipynb                     # GOLD-YOLO extended 70-epoch run
    │
    ├── Nenodet Plus/
    │   ├── Nenodet PLus.ipynb                               # NanoDet-Plus training & evaluation
    │   ├── Evaluation.png                                   # Evaluation metrics screenshot
    │   └── Result Graph.png                                 # Training result graph
    │
    └── YOLO V12N/
        └── visdrone_yolov12n (1).ipynb                      # YOLOv12n training & evaluation
```

---

## 🎯 Research Objective

Traditional object detection benchmarks focus on large models (YOLOv8/v9, DINO, RT-DETR) and standard datasets (COCO, VOC). This study **fills a gap** by:

1. Selecting **6 lightweight / underexplored architectures** that are rarely evaluated together
2. Applying a **uniform, fair training protocol** (same dataset, input size, epochs, hardware)
3. Measuring both **accuracy and efficiency** metrics (mAP, AP_S, FPS, GFLOPs, GPU Memory)
4. Quantifying the **SAHI uplift** — how much sliced inference improves small-object recall per architecture

---

## 🤖 Models Benchmarked

| Model | Architecture Type | Key Feature |
|---|---|---|
| 🟡 **GOLD-YOLO** | CNN + Gather-Distribute Neck | Enhanced feature aggregation for multi-scale |
| 🟠 **DAMO-YOLO** | NAS Backbone + RepGFPN | Neural-architecture-searched efficient backbone |
| 🟢 **NanoDet-Plus** | Ultra-light Anchor-Free | Edge-optimized, extremely low GFLOPs |
| 🔵 **D-FINE** | DETR-based Fine-Grained Regression | Transformer with progressive box refinement |
| 🟣 **YOLOv12n** | Attention-Augmented Compact YOLO | Flash-attention integrated nano YOLO |
| 🔴 **RF-DETR** | Transformer Baseline | Pure transformer reference model |

---

## 📦 Dataset — VisDrone2019-DET

| Split | Images | Size | Usage |
|---|---|---|---|
| 🟩 Train | 6,471 | 1.44 GB | Fine-tune all 6 models |
| 🟨 Validation | 548 | 0.07 GB | Monitor training / tune hyperparameters |
| 🟦 Test-Dev | 1,610 | 0.28 GB | Final evaluation & reporting (GT available ✅) |

**10 Detection Categories:**
`Pedestrian` · `Person` · `Car` · `Van` · `Bus` · `Truck` · `Motor` · `Bicycle` · `Awning-Tricycle` · `Tricycle`

> ⚠️ Only **Task 1 (Static Image Object Detection)** is used. Tasks 2–5 (video tracking, crowd counting) are excluded.

---

## ⚙️ Experimental Protocol

All models share an **identical, controlled training setup** for fair comparison:

```
Input Resolution : 640 × 640 px
Training Epochs  : 50 (early stopping on val mAP; GOLD-YOLO also extended to 70)
Annotation Format: VisDrone .txt → COCO JSON (VisDrone2COCO conversion script)
Augmentations    : Mosaic · Random Flip · HSV Shift  (train split only)
Random Seed      : Fixed (same across all runs)
Hardware         : Identical GPU for all models (fair FPS comparison)
Checkpoint       : Best val-mAP checkpoint saved per model
```

---

## 🔬 Evaluation Pipeline (Two Phases)

```
VisDrone Dataset
      │
      ▼
Format Conversion (VisDrone → COCO JSON)
      │
      ▼
Preprocessing (Resize 640×640, Augmentation)
      │
      ▼
Fine-Tune 6 Models (50 Epochs, Fixed Seed)
      │
      ├──────────────────────────────────────────────┐
      ▼                                              ▼
PHASE 1 — Baseline                        PHASE 2 — SAHI
(Full-Image Inference)                    (Sliced Inference)
      │                                              │
  mAP@0.5, AP_S, AP_M                       256/512px tiles
  Recall, FPS, GFLOPs                        ~20% overlap
  GPU Memory                              NMS tile merging
  Per-class AP_S (10 cats)               Same metrics as Ph.1
  Speed vs Accuracy Plot                  ΔAP_S · ΔRecall · ΔFPS
      │                                              │
      └──────────────┬───────────────────────────────┘
                     ▼
              Validation Gate
        (Is Phase 1 mAP acceptable?)
                     │
                     ▼
         Final Analysis & Reporting
   ┌──────────┬──────────┬──────────┬──────────┐
   │ SAHI     │ Per-Class│ Model    │  Final   │
   │ Impact   │ Heatmap  │ Ranking  │  Paper   │
   │ Bars     │(Model×Cat│(Acc/Speed│(IEEE PDF)│
   └──────────┴──────────┴──────────┴──────────┘
```

---

## 📊 Metrics Reported

| Category | Metrics |
|---|---|
| **Accuracy** | mAP@0.5, AP_S (small), AP_M (medium), Recall |
| **Efficiency** | FPS, GFLOPs, GPU Memory (MB) |
| **Per-Class** | AP_S for all 10 VisDrone categories |
| **SAHI Delta** | ΔAP_S, ΔRecall, ΔFPS (Phase 2 vs Phase 1) |
| **Rankings** | Best accuracy / best speed / best edge deployment |

---

## 📁 Notebooks Index

| Model | Notebook | Phase |
|---|---|---|
| D-FINE | [`d-fine-kaggle (4).ipynb`](Models/D-fine/d-fine-kaggle%20(4).ipynb) | Training (Phase 1) |
| D-FINE | [`d-fine-phase-2 (1).ipynb`](Models/D-fine/d-fine-phase-2%20(1).ipynb) | SAHI Inference (Phase 2) |
| DAMO-YOLO | [`damo-yolo-visdrone-pipeline.ipynb`](Models/DAMO-YOLO/50%20Epoches%20Result/damo-yolo-visdrone-pipeline.ipynb) | Full Pipeline (50 Epochs) |
| GOLD-YOLO | [`GOLD-YOLO-Visdrone_Phase1&2.ipynb`](Models/GOLD-YOLO/50%20Epoch/GOLD-YOLO-Visdrone_Phase1%262.ipynb) | Phase 1 + 2 (50 Epochs) |
| GOLD-YOLO | [`gold-yolo-70_epoch.ipynb`](Models/GOLD-YOLO/70%20Epoch/gold-yolo-70_epoch.ipynb) | Extended Run (70 Epochs) |
| NanoDet-Plus | [`Nenodet PLus.ipynb`](Models/Nenodet%20Plus/Nenodet%20PLus.ipynb) | Training & Evaluation |
| YOLOv12n | [`visdrone_yolov12n (1).ipynb`](Models/YOLO%20V12N/visdrone_yolov12n%20(1).ipynb) | Training & Evaluation |

---



---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install sahi
pip install pycocotools
```

### Dataset Setup

1. Download **VisDrone2019-DET** from the [official VisDrone website](https://github.com/VisDrone/VisDrone-Dataset)
2. Convert annotations to COCO JSON format:
   ```bash
   python visdrone2coco.py --input ./VisDrone/annotations --output ./data/coco/
   ```

### Run a Model

Open any notebook from the [Notebooks Index](#-notebooks-index) above in Kaggle or Jupyter and follow the cells sequentially. Each notebook covers:
- Data loading and preprocessing
- Model initialization with pretrained weights
- Fine-tuning on VisDrone train split
- Evaluation on test-dev split
- (Phase 2 notebooks) SAHI sliced inference

---

## 🏗️ Key Design Decisions

- **Why VisDrone?** — Industry-standard UAV aerial benchmark with extremely small, dense objects; average object size ~22×22 px makes it ideal for small-object detection research.
- **Why 50 epochs?** — Sufficient for convergence while keeping compute fair across architectures; early stopping prevents overfitting.
- **Why SAHI?** — Sliced Inference is the dominant post-processing technique for small objects in aerial imagery. Measuring its impact per architecture reveals model-specific compatibility.
- **Why these 6 models?** — Each represents a distinct architectural family (CNN, NAS, DETR-based, attention-augmented, ultra-light), enabling architecture-level comparisons beyond raw benchmarks.

---

## 👥 Authors

**Group 16** — Digital Image Processing Research Project

| Name | 
|---|
| Md. Khademul Islam Nahin |
| Irfanuzzaman Montasir |
| Md. Abdullah Al Imran |
| Md. Tahmid Shadhin |
| Argha Biswas |
| Md Shakib Al Islam |

---

## 📜 License

This project is for academic and research purposes. Dataset usage is subject to the [VisDrone Dataset License](https://github.com/VisDrone/VisDrone-Dataset).

---

## ⭐ Acknowledgements

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) — Tianjin University
- [SAHI](https://github.com/obss/sahi) — Slicing Aided Hyper Inference
- [GOLD-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) — Huawei Noah's Ark Lab
- [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) — Alibaba DAMO Academy
- [NanoDet-Plus](https://github.com/RangiLyu/nanodet) — RangiLyu
- [D-FINE](https://github.com/Peterande/D-FINE) — Peterande et al.
- [YOLOv12](https://github.com/sunsmarterjie/yolov12) — SunSmarterJie et al.
