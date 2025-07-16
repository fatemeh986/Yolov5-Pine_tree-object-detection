# Pine Tree Detection with YOLOv5

A lightweight pipeline for detecting pine trees in RGB images using YOLOv5. This repository covers:

* Converting Mask R-CNN JSON annotations to YOLO format
* Organizing a custom dataset for YOLOv5
* Running a series of training experiments
* Comparing results (mAP, recall, precision, confusion matrix)
* Analysis of performance and next steps

---

## 📂 Repository Structure

```
.
├── data
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── validations
│       ├── images
│       └── labels
├── yolov5
│   ├── train.py
│   ├── val.py
│   ├── detect.py
│   ├── dataset.yml
│   └── … (YOLOv5 code & configs)
├── convert_to_yolo.py        # Script to turn Mask R-CNN JSON → YOLO TXT
└── README.md
```

---

## 🗂 Dataset

* **One class**: `pine_tree` (class 0).

* **Structure** (root: `data/`):

  ```
  data/
  ├─ train/images
  ├─ train/labels    ← YOLO TXT files
  ├─ validations/images
  └─ validations/labels
  ```

* **`yolov5/dataset.yml`**:

  ```yaml
  # root: yolov5/
  path: ../data
  train: train/images
  val:   validations/images
  nc:    1
  names:
    0: pine_tree
  ```

---

## 🔄 Annotation Conversion

Use `convert_to_yolo.py` to turn your Mask R-CNN JSONs into YOLO `.txt` label files:

```bash
python convert_to_yolo.py \
  --json_dir data/train/annots \
  --labels_dir data/train/labels

python convert_to_yolo.py \
  --json_dir data/validations/annots \
  --labels_dir data/validations/labels
```

---

## ⚙️ Training Experiments

We ran **six** distinct experiments, each saved under `runs/train/<name>/`:

| Name            | Model    | Key Trick                                |
| --------------- | -------- | ---------------------------------------- |
| `pine_tree_run` | YOLOv5-s | default training (50 epochs)             |
| `exp_freeze`    | YOLOv5-s | first 30 ep freezing backbone → +20 ep   |
| `exp_unfreeze`  | YOLOv5-s | fine-tune all layers                     |
| `exp_anchor`    | YOLOv5-s | (auto)recomputed anchors                 |
| `exp_evolve`    | YOLOv5-s | hyperparameter evolution (`--evolve`)    |
| `exp_yolov5m2`  | YOLOv5-m | medium model + best `exp_evolve` hparams |

Each was launched with a one-liner, e.g.:

```bash
python yolov5/train.py \
  --weights yolov5s.pt \
  --data dataset.yml \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --name exp_anchor
```

---

## 📊 Results

### Quantitative Metrics

| Experiment      | Precision | Recall | mAP\@50 | mAP\@50-95 | “Accuracy” (TP/(TP+FP+FN)) |
| --------------- | :-------: | :----: | :-----: | :--------: | :------------------------: |
| pine\_tree\_run |   0.664   |  0.600 |  0.646  |    0.417   |           46.2 %           |
| exp\_yolov5m2   |     —     |  0.830 |    —    |      —     |           41.5 %           |

*(Detailed logs are in each run’s `results.txt`.)*

### Best Confusion Matrix

![](runs/val/val_exp_yolov5m2/confusion_matrix.png)

* **True Positive Rate** (recall): 83 %
* **False Negative Rate**: 17 %
* **False Positive Rate** (background→pine): 100 %
* Implicit background class yields a 2×2 matrix for single‐class detection.

---

## 🖼️ Example Detections

Here are two sample detection outputs:

![Detection Example 1](https://imgur.com/a/wsoGKaL)

<!-- ![Detection Example 2](val_batch0_pred.jpg) -->


## 🔍 Analysis

1. **Dataset Challenges**

   * **Size & diversity**: only \~150 train images with limited backgrounds
   * **Image quality**: some images are low-contrast or blurred
   * **Single‐class, implicit negatives**: no explicit “background” boxes

2. **Model Capacity & Augmentation**

   * Upgrading to YOLOv5-m and applying evolved hyperparameters boosted recall from 60 % → 83 %.
   * Precision remains limited by high false positives on complex backgrounds.

3. **Detection‐Accuracy vs. Classification**

   * In object detection, we measure precision/recall/mAP; the “accuracy” here is TP/(TP+FP+FN) ≈ 41 %.

---

## 🚀 Next Steps

* **Expand & diversify data**: collect more scenes (different lighting, angles, seasons).
* **Improve annotations**: tighter, more consistent bounding boxes; consider adding a few explicit “background” crops for negative sampling.
* **Multi-spectral inputs**: include NIR or depth channels if available.
* **Advanced augmentation**: copy-paste, stronger color/scale/perspective variants.
* **Ensemble & TTA**: combine multiple runs or test-time augmentations for gain.

---

## ▶️ Usage

1. **Install dependencies**

   ```bash
   pip install -r yolov5/requirements.txt
   ```
2. **Convert annotations**

   ```bash
   python convert_to_yolo.py ...
   ```
3. **Train**

   ```bash
   python yolov5/train.py --data yolov5/dataset.yml --name your_experiment
   ```
4. **Validate**

   ```bash
   python yolov5/val.py --weights runs/train/your_experiment/weights/best.pt \
                        --data yolov5/dataset.yml \
                        --save-conf --verbose --name val_your_experiment
   ```
5. **Detect**

   ```bash
   python yolov5/detect.py --weights runs/train/your_experiment/weights/best.pt \
                          --source path/to/images \
                          --save-txt --name detect_your_experiment
   ```

---

> **Acknowledgments**
>
> * [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
> * Early experiments with Mask R-CNN annotations
