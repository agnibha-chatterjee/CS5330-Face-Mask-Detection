# Face Mask Detection System

## Authors

- Om Agarwal
- Agnibha Chatterjee

## About

This project detects whether people in images or video streams are wearing face masks using deep learning. It consists of:

- A face detector (OpenCV's DNN module with Caffe model)
- A custom mask classifier (MobileNetV2-based)

## Installation

1. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download face detector model files:**
   - The required files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) are already present in `face_detector/`
   - If missing, download from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

## Dataset Structure

The dataset directory should be structured as follows:

Images of people wearing masks: dataset/with_mask

Images of people without masks: dataset/without_mask

## Script Usage

### 1. Training the Model (`train.py`)

Train the mask detection model on your dataset.

**Arguments:**

- `-d/--dataset`: Path to input dataset directory (required)
- `-p/--plot`: Path to output loss/accuracy plot (PNG) (default: "plot.png")
- `--plot_pdf`: Path to output loss/accuracy plot (PDF) (default: "plot.pdf")
- `-m/--model`: Path to output model file (default: "mask_detector.keras")
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Initial learning rate (default: 1e-4)

**Example:**

```bash
python train.py -d dataset --epochs 30 --batch_size 32
```

### 2. Image Detection (`detect_in_image.py`)

Detect masks in a single image.

**Arguments:**

- `-i/--image`: Path to input image (required)
- `-f/--face`: Path to face detector model directory (default: "face_detector")
- `-m/--model`: Path to trained mask detector model (default: "mask_detector.keras")
- `-c/--confidence`: Minimum detection confidence (default: 0.5)
- `-o/--output`: (Optional) Path to save output image

**Example:**

```bash
python detect_in_image.py -i test.jpg -o output.jpg
```

### 3. Video/Webcam Detection (`detect_in_video.py`)

Detect masks in real-time from webcam or video file.

**Arguments:**

- `-f/--face`: Path to face detector model directory (default: "face_detector")
- `-m/--model`: Path to trained mask detector model (default: "mask_detector.keras")
- `-c/--confidence`: Minimum detection confidence (default: 0.5)

**Example (webcam):**

```bash
python detect_in_video.py
```

**Example (video file):**

```bash
python detect_in_video.py --video test.mp4
```

## Model Information

The trained model (`mask_detector.keras`) is based on MobileNetV2 with:

- Input size: 224x224 pixels
- Output: Binary classification (Mask/No Mask)
- Training: 20 epochs by default with data augmentation
