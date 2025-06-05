# MediaPipe Facial Landmarks Detection Evaluation

This directory contains the evaluation results for the MediaPipe face landmarks detection model.

## Model Information

MediaPipe's face landmarks detection uses a machine learning pipeline that includes face detection and landmark regression. The model provides different landmark configurations:

- **Face Mesh**: 468 3D facial landmarks
- **Attention Mesh**: 478 landmarks (468 face landmarks + 10 iris landmarks)
- **Refined mesh landmarks**: Higher accuracy landmarks with more detail

The model is optimized for real-time mobile performance and provides a comprehensive mapping of the face geometry.

## Evaluation Results

The evaluation results in this directory include:

- `metrics.csv`: Summary metrics for the evaluation run
- `mediapipe_landmarks_metrics.csv`: Model-specific metrics in CSV format for cross-model comparison
- `mediapipe_landmarks_detailed_results.json`: Detailed evaluation results in JSON format
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
- **Accuracy**: Precision of landmark placement
- **Memory Usage**: Memory footprint during detection
- **CPU Utilization**: CPU utilization during detection

## Running MediaPipe Evaluation

To run the MediaPipe facial landmarks evaluation:

```bash
python scripts/mediapipe_landmarks_evaluation.py --dataset PATH_TO_DATASET --sample_size 500 --output_dir results/mediapipe
```

## Comparison to Other Models

For a comparison between MediaPipe and other facial landmark detection models (dlib, TensorFlow.js), refer to the comparison reports in:

```
face_landmarks/results/comparison/
```

or run the comparison script:

```bash
python scripts/generate_comparison_report.py --dlib_results results/dlib --mediapipe_results results/mediapipe --tfjs_results results/tfjs --output_dir results/comparison
```
