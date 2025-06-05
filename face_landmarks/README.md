# Facial Landmarks Detection

This directory contains resources for evaluating and using facial landmarks detection models, particularly focused on dlib's 68-point facial landmark predictor, MediaPipe's face landmarks detection, and TensorFlow.js face landmarks detection models.

## Contents

### Scripts
- `landmarks_evaluation.py`: Simplified, single-script solution for evaluating dlib facial landmarks detection
- `mediapipe_landmarks_evaluation.py`: Script for evaluating MediaPipe face landmarks detection
- `tfjs_landmarks_evaluation.py`: Script for evaluating TensorFlow.js face landmarks detection
- `run_landmarks_evaluation.sh`: Wrapper script to easily run and compare all landmarks evaluation methods
- `generate_comparison_report.py`: Script to create comparative reports across different landmark detection models
- `download_tfjs_model.py`: Script to download and convert TensorFlow.js face landmarks models for Python
- `evaluate_landmarks.py`: Original script for large-scale evaluation of facial landmarks detection
- `display_landmarks.py`: Utility for displaying facial landmarks on images
- `dlib_68points_visualizer.py`: Tool for visualizing the 68 facial landmarks using dlib
- `enhanced_landmark_visualizer.py`: Enhanced visualization with color-coded facial feature groups
- `simple_cv_visualizer.py`: Simple OpenCV-based landmark visualization 
- `generate_html_viewer.py`: Tool for generating HTML reports of landmark detection results
- `benchmark_face_detectors.py`: Benchmark script for face detection methods
- Various other visualization and utility scripts

### Documentation
- `landmark_evaluation_report.md`: Comprehensive evaluation report on dlib's 68-point landmarks detector
- `future_improvements.md`: Recommendations for future enhancements to the evaluation process

### Results
- Contains the outputs from landmark detection evaluations including:
  - Performance metrics (CSV, JSON)
  - Visualization images
  - HTML reports
  - Performance plots

### Sample Visualizations
- The `dlib_visualizations` directory contains sample images with visualized landmarks

## Usage

### Using the Evaluation Scripts

#### Dlib Landmarks Evaluation

To run the dlib facial landmarks evaluation script:

```bash
python landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/dlib
```

#### MediaPipe Landmarks Evaluation

To run the MediaPipe facial landmarks evaluation script:

```bash
python scripts/mediapipe_landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/mediapipe
```

#### TensorFlow.js Landmarks Evaluation

To run the TensorFlow.js facial landmarks evaluation script:

```bash
python scripts/tfjs_landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/tfjs
```

#### All-in-One Evaluation and Comparison

To run all landmarks evaluations and generate comparison reports:

```bash
./run_landmarks_evaluation.sh
# When prompted, select option 4 to run all models
python mediapipe_landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/mediapipe
```

#### TensorFlow.js Landmarks Evaluation

To run the TensorFlow.js facial landmarks evaluation script:

```bash
python tfjs_landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/tfjs
```

### Using the Wrapper Script

For convenience, you can use the wrapper script for the simplified evaluation:

```bash
./run_landmarks_evaluation.sh
```

You can also pass options to customize the evaluation:

```bash
./run_landmarks_evaluation.sh --sample-size 1000 --output-dir results/my_test
```

### Using the Original Script

To run the original facial landmarks evaluation on a dataset:

```bash
python scripts/evaluate_landmarks.py --dataset PATH_TO_DATASET --sample_size 10000 --output_dir results
```

Parameters:
- `--dataset`: Path to the image dataset (default: LFW dataset path)
- `--sample_size`: Number of images to sample for evaluation (default: 10000)
- `--output_dir`: Directory to save evaluation results
- `--visualize`: Number of detection results to visualize
- `--batch_size`: Batch size for processing images
- `--conf_thres`: Face detection confidence threshold
- `--landmarks_model`: Path to the dlib landmarks model
- `--seed`: Random seed for reproducible sampling

## Models

The evaluation uses the following models:
- Face detection: dlib's HOG-based face detector
- Landmarks: dlib's 68-point facial landmarks predictor (`shape_predictor_68_face_landmarks.dat`)

The landmark model file should be located in the `landmarks_models` directory.

## Results

The evaluation on 10,000 LFW images shows:
- 99.26% face detection rate
- 100% landmarks detection success rate on detected faces
- Average processing time of 10.19 ms per image
- Throughput of 98.12 FPS
