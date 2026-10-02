# QR Code Authenticity Detector

A deep learning web application that classifies QR codes as **authentic** or **tampered/fake** using transfer learning with MobileNetV2 and a Flask web interface.

---

## Overview

QR codes are widely used in payment systems, authentication, and product verification. This project addresses the growing threat of fraudulent QR codes — where malicious actors replace legitimate payment or authentication QR codes with their own.

This system uses computer vision and deep learning to detect visual tampering patterns that are invisible to the human eye, achieving **99.33% test accuracy** on a balanced dataset of 1,000 QR code images.

---

## Demo

Upload any QR code image through the web interface and get an instant prediction with confidence scores:

- **Authentic** → Green result with confidence percentage
- **Fake/Tampered** → Red alert with confidence percentage

---

## Features

- Transfer learning with **MobileNetV2** pre-trained on ImageNet
- **99.33% accuracy** on a held-out test set
- Clean Flask web interface with drag-and-drop image upload
- Real-time confidence scores and probability breakdown
- CPU-friendly inference (no GPU required)
- Balanced dataset: 500 authentic + 500 tampered QR codes

---

## Project Structure

```
qr-authenticity-detector/
├── model/
│   ├── mobilenet_model.py      # MobileNetV2 architecture definition
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation with metrics & confusion matrix
│   ├── test_model.py           # Single-image inference test
│   └── best_qr_model.pth       # Trained model weights (8.72 MB)
├── backend/
│   ├── app.py                  # Flask web server & prediction API
│   └── templates/
│       └── index.html          # Web interface (HTML/CSS/JS)
├── dataset/
│   ├── original/               # 500 authentic QR code images (PNG)
│   └── fake/                   # 500 tampered QR code images (JPG)
├── utils/
│   ├── dataset_loader.py       # PyTorch Dataset class
│   ├── generate_qr.py          # Generates original QR codes
│   ├── generate_fake_qrs.py    # Generates tampered QR codes
│   ├── check_dataset.py        # Dataset integrity checker
│   └── metrics.py              # Custom evaluation metrics
├── preprocessing/
│   └── preprocess.py           # Image preprocessing utilities
├── requirements.txt
└── README.md
```

---

## Model Performance

| Metric    | Original | Fake  | Overall |
|-----------|----------|-------|---------|
| Precision | 0.99     | 1.00  | —       |
| Recall    | 1.00     | 0.99  | —       |
| F1-Score  | 0.99     | 0.99  | —       |
| Accuracy  | —        | —     | **99.33%** |

**Confusion Matrix (150 test images):**

```
                 Predicted
              Original   Fake
Actual  Orig     67        0
        Fake      1       82
```

Only 1 misclassification out of 150 test samples.

---

## Architecture

```
Input Image (any size)
        ↓
  Preprocess: Resize to 224×224, Normalize
        ↓
  MobileNetV2 Feature Extractor (frozen, pre-trained on ImageNet)
        ↓
  Custom Classifier: Linear(1280 → 2)
        ↓
  Softmax → [P(Original), P(Fake)]
        ↓
  Result + Confidence Score
```

**Why MobileNetV2?**
- Lightweight (8.72 MB) — suitable for web deployment
- Fast CPU inference — no GPU needed
- Pre-trained on 1.4M images — rich feature representations
- Transfer learning dramatically reduces training time

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/your-username/qr-authenticity-detector.git
cd qr-authenticity-detector

# Install dependencies (CPU-only PyTorch — safe for any machine)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask opencv-python numpy matplotlib scikit-learn pillow qrcode
```

---

## Running the Application

```bash
python backend/app.py
```

Then open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Upload a QR code image and click **Analyze** to get the prediction.

---

## Training the Model

The model is already trained and saved as `model/best_qr_model.pth`. To retrain:

```bash
cd model
python train.py
```

Training configuration:
- **Epochs**: 15
- **Batch size**: 16
- **Optimizer**: Adam (lr = 0.0001)
- **Loss**: CrossEntropyLoss
- **Split**: 70% train / 15% validation / 15% test

---

## Evaluating the Model

```bash
cd model
python evaluate.py
```

Outputs a full classification report, confusion matrix, and ROC curve.

---

## Dataset Generation

The dataset was programmatically generated for reproducibility:

```bash
# Generate original QR codes
python utils/generate_qr.py

# Generate tampered/fake QR codes  
python utils/generate_fake_qrs.py
```

Tampering techniques used: Gaussian noise injection, Gaussian blur, pixel-level modifications, rotation distortion, and JPEG compression artifacts.

---

## Technology Stack

| Component         | Technology            | Purpose                        |
|-------------------|-----------------------|--------------------------------|
| Deep Learning     | PyTorch 2.x           | Model definition & training    |
| Model Architecture| MobileNetV2           | Feature extraction             |
| Web Framework     | Flask                 | REST API & web server          |
| Image Processing  | OpenCV, Pillow        | Image loading & preprocessing  |
| ML Evaluation     | scikit-learn          | Metrics, confusion matrix      |
| Visualization     | Matplotlib            | Training curves, plots         |

---

## Safety Note for Low-End Machines

This project is configured to run on **CPU only**. It does **not** require a GPU and will not overheat or damage your laptop. The model is small (8.72 MB) and inference takes under 1 second per image on a standard laptop CPU.

---

## License

This project is open source and available under the [MIT License](LICENSE).
