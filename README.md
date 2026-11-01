# PAN Card Field Detection with YOLOv12

An object-detection project that uses **Ultralytics YOLOv12s** to locate important fields on PAN card images. The detector identifies the regions corresponding to the cardholder's name, father's name, date of birth, and PAN number.

> **Project status:** Initial training and validation experiments have been completed. The reported five-epoch run is a baseline/pipeline check, not a claim of production-ready performance. Dataset cleanup, longer controlled training, held-out test evaluation, and further error analysis remain important next steps.

## Table of Contents

- [Overview](#overview)
- [Detection Classes](#detection-classes)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Environment](#environment)
- [Installation](#installation)
- [Dataset Configuration](#dataset-configuration)
- [Training](#training)
- [Validation and Metrics](#validation-and-metrics)
- [Inference](#inference)
- [Understanding the Output](#understanding-the-output)
- [Experiments and Observations](#experiments-and-observations)
- [Known Issues and Next Steps](#known-issues-and-next-steps)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)

## Overview

The project implements the object-detection stage of a document-processing workflow:

1. Load a PAN card image.
2. Detect the locations of the required fields using a fine-tuned YOLOv12 model.
3. Return each detection's class, confidence score, and bounding-box coordinates.
4. Optionally pass the detected crops to an OCR system in a subsequent stage to extract their text.

**Important:** This repository focuses on field localisation. YOLO detection does not itself perform text recognition or validate the extracted PAN details.

## Detection Classes

The dataset contains four classes:

| Class ID | Class name | Description |
|---:|---|---|
| 0 | `dob` | Date of birth field |
| 1 | `father` | Father's name field |
| 2 | `name` | Cardholder's name field |
| 3 | `pan` | PAN number field |

## Dataset

The project uses the PAN Card dataset identified as [`agendra048/pan-card-dataset`](https://www.kaggle.com/datasets/agendra048/pan-card-dataset).

The dataset uses YOLO-format bounding-box annotations. Each annotation line has the format:

```text
class_id x_center y_center width height
```

The coordinates and dimensions are normalised relative to the image width and height.

### Dataset layout

The training configuration expects a structure similar to:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/                 # Optional, if a test split is provided
    ├── images/
    └── labels/
```

The dataset is not included in this repository. Download it separately and arrange the directories to match the paths in `data.yaml`.

### Dataset YAML

Example `data.yaml`:

```yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  - dob
  - father
  - name
  - pan
```

Using `path` makes the dataset root explicit. If you retain the original downloaded YAML, verify that its relative paths resolve correctly from the YAML file's location. Remove the `test` entry if no test split exists, or provide the corresponding directory before running test evaluation.

## Repository Structure

A suggested clean repository layout is:

```text
pan-card-field-detection/
├── scripts/
│   ├── check_dataset.py
│   ├── visualize_annotations.py
│   └── train.py
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

Keep the dataset, virtual environment, training outputs, and large model checkpoints outside Git unless there is a specific reason and permission to publish them. Update this structure to match the files actually committed to your repository.

## Environment

The experiments were carried out on a Windows machine using:

| Component | Environment used |
|---|---|
| Python | 3.12.6 |
| Ultralytics | 8.4.x (8.4.84 was reported during environment checks) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU memory | 6 GB |
| Framework | PyTorch with CUDA support |

The exact PyTorch build and CUDA runtime should be selected for the machine being used. An NVIDIA driver reporting CUDA capability does not guarantee that the installed PyTorch package is CUDA-enabled.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd pan-card-field-detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, you can use a process-scoped execution-policy change for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Alternatively, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install PyTorch

Install a PyTorch build that supports CUDA on your system by following the official selector:

https://pytorch.org/get-started/locally/

For GPU training, verify CUDA availability before proceeding:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If `torch.cuda.is_available()` is `False`, resolve the PyTorch/CUDA installation before expecting YOLO to train on the GPU.

### 4. Install project dependencies

Install Ultralytics:

```bash
pip install ultralytics
```

A minimal `requirements.txt` can contain:

```text
ultralytics
```

For reproducible experiments, pin the exact versions that you have tested in your own environment rather than relying indefinitely on unpinned latest versions.

## Dataset Configuration

Place the dataset in the expected directory structure and update `data.yaml`.

Before training, check:

- every image has the expected corresponding label file;
- label files contain valid class IDs (`0` to `3`);
- normalised coordinates are within valid bounds;
- boxes are placed around the correct fields;
- train, validation, and test splits do not contain duplicate or near-duplicate images.

Filename matching alone does not guarantee annotation correctness. Visualise annotated samples and inspect them manually.

A dataset-integrity script can compare image and label filename stems and report unmatched files. During the initial check for this project, the training split contained **1,448 images and 1,458 label files**, with **10 label files that had no matching image** and no images without a matching label. These unmatched annotations should be investigated before treating the dataset as clean.

## Training

The initial model choice was **YOLOv12s**. It was selected as a practical starting point for a dataset of roughly 1.4k training images and a laptop GPU with 6 GB VRAM.

Create or adapt `scripts/train.py`:

```python
from ultralytics import YOLO


def main():
    model = YOLO("yolo12s.pt")

    model.train(
        data="data.yaml",
        epochs=5,
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        pretrained=True,
        plots=True,
        val=True,
        project="runs/pan_card",
        name="yolo12s_baseline"
    )


if __name__ == "__main__":
    main()
```

Run it from the repository root:

```bash
python scripts/train.py
```

The values above reproduce the **short initial baseline configuration** (5 epochs, image size 640, batch size 4); they are not presented as final optimised hyperparameters. For a proper training experiment, increase the training duration based on validation behaviour and compare runs systematically.

### Key parameters

| Parameter | Purpose |
|---|---|
| `data` | Location of the dataset YAML |
| `epochs` | Maximum number of training epochs |
| `imgsz` | Input image size |
| `batch` | Number of images per batch |
| `workers` | Data-loader worker processes |
| `device` | `0` selects the first CUDA GPU |
| `pretrained` | Starts from pretrained weights |
| `plots` | Saves training and validation plots |
| `val` | Runs validation during training |
| `project`, `name` | Organise experiment output directories |

`workers=0` is used here as a Windows-friendly starting point because a multiprocessing bootstrapping error occurred during an earlier run. Worker count can be benchmarked after the training script runs reliably.

## Validation and Metrics

The initial five-epoch evaluation reported the following overall metrics:

| Metric | Reported value |
|---|---:|
| Precision | 0.982 |
| Recall | 0.982 |
| mAP@50 | 0.986 |
| mAP@50–95 | 0.646 |

These figures are from the initial short run and should be treated as preliminary validation results, not final test-set performance.

### Metric definitions

- **Precision:** Fraction of predicted detections that are correct.
- **Recall:** Fraction of ground-truth objects that the model detects.
- **mAP@50:** Mean average precision at an IoU threshold of 0.50.
- **mAP@50–95:** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.

The difference between mAP@50 and mAP@50–95 is a reason to inspect bounding-box tightness and annotation quality. High mAP@50 alone does not establish precise localisation or robust performance on unseen data.

Ultralytics saves plots and metrics in the run directory. Review the loss curves, precision-recall curves, confusion matrix, and example predictions rather than relying on a single metric.

## Inference

Once training is complete, load the selected checkpoint. The example below assumes the checkpoint is at the shown path; update it to the actual run directory.

```python
from ultralytics import YOLO

model = YOLO("runs/pan_card/yolo12s_baseline/weights/best.pt")

results = model.predict(
    source="path/to/pan_card_image.jpg",
    conf=0.25,
    save=True
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        print({
            "field": class_name,
            "confidence": round(confidence, 4),
            "bbox_xyxy": [x1, y1, x2, y2]
        })
```

To run inference on a folder, set `source` to the folder path. Ultralytics returns a result for each processed image.

### Checkpoints

- `best.pt`: checkpoint selected as best according to the validation fitness criterion.
- `last.pt`: checkpoint from the final training epoch.

Use `best.pt` as the usual starting point for inference and final evaluation. Keep `last.pt` if you may need to resume training.

## Understanding the Output

Each detection provides:

- **Class:** one of `dob`, `father`, `name`, or `pan`;
- **Confidence:** the model's confidence score for that detection;
- **Bounding box:** pixel coordinates in `[x1, y1, x2, y2]` format.

The confidence threshold (`conf`) controls which predictions are returned. Choose it using validation data and the relative cost of false positives versus missed fields.

## Experiments and Observations

The project involved:

- reviewing YOLO architecture and transfer learning concepts;
- inspecting YOLO-format annotations;
- writing scripts to compare image and label filenames;
- checking annotated images visually;
- setting up a local Python environment in VS Code;
- diagnosing a CPU-only PyTorch installation and moving to a CUDA-capable setup;
- handling Windows path and multiprocessing issues;
- training a YOLOv12s baseline and examining its validation metrics and loss curves;
- learning how to load `best.pt` and run inference.

The first five-epoch run was useful for verifying that the training pipeline worked. It was not sufficient to establish convergence or final model quality.

## Known Issues and Next Steps

- [ ] Investigate and resolve the 10 unmatched training label files.
- [ ] Confirm the dataset YAML paths and the existence of the configured test split.
- [ ] Inspect annotations for incorrect classes, misplaced boxes, and invalid coordinates.
- [ ] Check for duplicate or near-duplicate images across data splits.
- [ ] Run a longer, controlled YOLOv12s experiment.
- [ ] Analyse false positives, missed detections, and localisation errors.
- [ ] Tune training parameters one at a time and record each experiment.
- [ ] Evaluate the selected checkpoint on a held-out test set.
- [ ] Compare YOLOv12m only if the error analysis suggests the small model is capacity-limited.
- [ ] If required, integrate OCR to read text from detected field crops and evaluate the complete pipeline separately.

## Limitations

- This is a field-detection model, not a complete PAN verification or OCR system.
- The reported metrics are from a short validation run and are not a final independent test benchmark.
- Performance may vary with image quality, card orientation, lighting, blur, cropping, and document layout.
- Dataset licensing and usage conditions should be reviewed before redistributing the dataset or derived assets.

## Acknowledgements

- Dataset: [PAN Card Dataset on Kaggle](https://www.kaggle.com/datasets/agendra048/pan-card-dataset)
- Framework: [Ultralytics](https://docs.ultralytics.com/)
- Dataset annotation reference included in the supplied YAML: [Roboflow Universe project](https://universe.roboflow.com/panocr/ocr-qaxqg/dataset/5)

---

## Security and Repository Hygiene

PAN card images contain sensitive personal information. **Do not upload real PAN card images, annotation files containing personal data, or extracted identity information to a public GitHub repository.** Keep the dataset private and follow your internship/company's data-handling requirements.

Also avoid committing:

- `.venv/`
- `__pycache__/`
- large training run directories such as `runs/`
- datasets and personal documents
- model checkpoints unless you have permission and have checked the applicable dataset/model licence

A suitable starting `.gitignore` is:

```gitignore
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
runs/
datasets/
dataset/
*.pt
.env
```

If you intend to publish a trained checkpoint, first confirm that sharing it is permitted and that it does not expose restricted or sensitive data.
