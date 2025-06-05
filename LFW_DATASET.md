# Dataset Details: Labeled Faces in the Wild (LFW)

![LFW Example Images](./assets/lfw_samples.png)

## Overview

The Labeled Faces in the Wild (LFW) dataset is a database of face photographs designed for studying the problem of unconstrained face recognition. It contains:

- 13,233 face images
- 5,749 different individuals
- 1,680 people with two or more distinct photos

## Key Characteristics

LFW contains images collected from the web with significant variations in:
- Pose
- Lighting
- Expression
- Background
- Camera quality
- Occlusions

## Dataset Structure

The LFW dataset is organized as follows:

```
lfw-deepfunneled/
  ├── Aaron_Eckhart/
  │     ├── Aaron_Eckhart_0001.jpg
  │     └── ...
  ├── Aaron_Guiel/
  │     └── Aaron_Guiel_0001.jpg
  ├── Aaron_Patterson/
  │     └── Aaron_Patterson_0001.jpg
  └── ...
```

Each person's images are stored in a separate directory named after the individual.

## Evaluation Protocol

For our evaluation, we used the standard verification protocol:
- 3,000 matched pairs (same person)
- 3,000 mismatched pairs (different people)
- 10-fold cross-validation

## Preprocessing

For our experiments, we used:
- LFW-deepfunneled version with improved alignment
- Face detection to extract facial regions
- Image normalization and standardization for consistent evaluation

## Usage in Our Research

The LFW dataset was used to evaluate:
1. **Face Detection**: Testing detection accuracy across different models
2. **Facial Landmarks**: Evaluating landmark placement accuracy and consistency
3. **Face Recognition**: Measuring verification accuracy of different recognition algorithms

## Statistics From Our Evaluation

From our evaluation using LFW:

| Metric | Value |
|--------|-------|
| Images with Faces Detected | 13,233 (100%) |
| Images with Multiple Faces | 1,685 (12.7%) |
| Average Face Size | 103.6 × 103.6 pixels |
| Successful Landmark Detections | 99.4% (dlib), 100% (MediaPipe) |
| Recognition Accuracy (best) | 98.6% (Dlib ResNet model) |

## Reference

For more information on the LFW dataset:
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- [Original Academic Website](http://vis-www.cs.umass.edu/lfw/)

Note: Our experimentation specifically uses the LFW-deepfunneled version, which provides better alignment compared to the original dataset.

## Contact

For questions about our dataset usage or implementation:
- Email: abdelrhman.s.elsayed@gmail.com
