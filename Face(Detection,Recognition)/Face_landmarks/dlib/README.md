# Dlib Facial Landmarks Detection Evaluation

This directory contains the evaluation results for the Dlib 68-point facial landmark detector model (`shape_predictor_68_face_landmarks.dat`).

## Model Information

Dlib's facial landmark detector is based on Kazemi and Sullivan's paper ["One Millisecond Face Alignment with an Ensemble of Regression Trees"](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) and provides 68 points that map to specific facial features:

- Points 1-17: Jaw line
- Points 18-22: Left eyebrow
- Points 23-27: Right eyebrow
- Points 28-31: Nose bridge
- Points 32-36: Nose bottom
- Points 37-42: Left eye
- Points 43-48: Right eye
- Points 49-68: Mouth

## Evaluation Results

The evaluation results in this directory include:

- `metrics.csv`: Summary metrics for the evaluation
- `landmarks_metrics.json`: Detailed metrics in JSON format
- `landmarks_results.csv`: Per-image evaluation results
- `detection_time_distribution.png`: Histogram of detection times
- `detection_time_vs_faces.png`: Scatter plot of detection time vs. number of faces
- `face_count_distribution.png`: Distribution of face counts in the dataset
- `performance_summary.png`: Overall performance metrics visualization
- `success_rate_by_face_count.png`: Success rate based on the number of faces in an image
- `visualizations/`: Sample images with visualized landmarks

## Key Performance Indicators

- **Average Detection Time**: Time taken to detect landmarks per face
- **Success Rate**: Percentage of faces where landmarks were successfully detected
- **Accuracy**: Precision of landmark placement (when ground truth is available)

## Running Dlib Evaluation

To run the dlib facial landmarks evaluation:

```bash
python scripts/landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/dlib
```

## Comparison to Other Models

For a comparison between dlib and other facial landmark detection models (MediaPipe, TensorFlow.js), refer to the comparison reports in:

```
face_landmarks/results/comparison/
```

or run the comparison script:

```bash
python scripts/generate_comparison_report.py --dlib_results results/dlib --mediapipe_results results/mediapipe --tfjs_results results/tfjs --output_dir results/comparison
```
