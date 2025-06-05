# Face Detection Models Documentation

This document provides information about the face detection models implemented in the project, including the recently added YOLOv5 and YOLOv8 face detectors.

## Available Face Detection Models

The following face detection models are available in the project:

1. **Haar Cascade Face Detector**
   - Traditional OpenCV implementation
   - Fast but less accurate than deep learning models
   - Good for simple face detection tasks with frontal faces

2. **MMOD Face Detector**
   - Based on HOG (Histogram of Oriented Gradients) features
   - Uses Max-Margin Object Detection
   - Better at detecting faces from different angles than Haar

3. **SSD Face Detector**
   - Single Shot MultiBox Detector with ResNet-10 backbone
   - Pre-trained on WIDER FACE dataset
   - Good balance between speed and accuracy

4. **MobileNet SSD Face Detector**
   - Single Shot MultiBox Detector with MobileNet backbone
   - Optimized for mobile and edge devices
   - Faster than standard SSD but slightly less accurate

5. **YOLOv5 Face Detector** (New)
   - YOLOv5 architecture fine-tuned for face detection
   - Provides excellent performance with reasonable speed
   - Trained on WIDER FACE dataset

6. **YOLOv8 Face Detector** (New)
   - Latest YOLO architecture with improved accuracy
   - Enhanced feature extraction capability
   - Better handling of varied face poses and lighting conditions

## Model Comparison

The models can be compared using the `compare_face_detectors.py` script, which evaluates all detectors on the same dataset and provides comprehensive metrics including:

- Detection accuracy
- Processing speed
- Precision and recall
- F1 score

Additionally, the `compare_yolo_detectors.py` script provides a detailed comparison specifically between YOLOv5 and YOLOv8 face detectors.

## Usage

### Command Line Interface

To run a specific face detector:

```bash
./run_face_detection.sh [options] [detector_name]
```

Available detectors:
- `haar` - Haar Cascade detector
- `mmod` - MMOD detector  
- `ssd` - SSD Face detector
- `mobilenet` - MobileNet SSD detector
- `yolov5` - YOLOv5 face detector
- `yolov8` - YOLOv8 face detector
- `all` - Run all detectors

Options:
- `-s, --sample-size SIZE` - Number of images to sample (default: 1000)
- `-d, --dataset PATH` - Path to dataset (default: ../4/lfw-deepfunneled/lfw-deepfunneled)
- `-o, --output-dir DIR` - Output directory (default: ./results)
- `-v, --visualize NUM` - Number of images to visualize (default: 10)
- `-c, --compare` - Run comparison of all detectors

### Running YOLO Comparison

To compare YOLOv5 and YOLOv8 face detectors:

```bash
python scripts/comparison/compare_yolo_detectors.py [options]
```

Options:
- `--dataset PATH` - Path to the dataset
- `--sample_size SIZE` - Number of images to sample (default: 200)
- `--visualize NUM` - Number of images to visualize (default: 10)
- `--output_dir DIR` - Directory to save results
- `--conf_thres THRESHOLD` - Confidence threshold for detections (default: 0.5)
- `--full_dataset` - Use full dataset instead of sampling

## Implementation Details

### YOLOv5 Face Detector

The YOLOv5 face detector implementation includes:

1. Model loading with automatic detection of available models:
   - Searches for face-specific YOLOv5 models
   - Falls back to standard YOLOv5 if face model not found

2. Face detection with confidence filtering:
   - Detects faces in images
   - Can filter to keep only the highest confidence face per image
   - Configurable confidence threshold

3. Visualization and evaluation:
   - Visual output of detected faces
   - Detection speed measurement
   - Accuracy calculation against expected number of faces

### YOLOv8 Face Detector

The YOLOv8 face detector implementation includes:

1. Model loading using Ultralytics YOLO class:
   - Searches for face-specific YOLOv8 models
   - Falls back to standard YOLOv8 if face model not found

2. Face detection with confidence filtering:
   - Enhanced detection capabilities compared to previous versions
   - Can filter to keep only the highest confidence face per image
   - Configurable confidence threshold

3. Visualization and evaluation:
   - Visual output of detected faces
   - Detection speed measurement
   - Accuracy calculation against expected number of faces

## Model Performance

Initial evaluations show that:

- YOLOv8 generally provides better accuracy than YOLOv5
- YOLOv5 may offer faster inference on some hardware configurations
- Both YOLO models outperform traditional methods like Haar and MMOD in terms of accuracy
- For best speed, MobileNet SSD may still be preferred on resource-constrained devices

For detailed performance metrics, run the comparison scripts on your specific dataset and hardware.
