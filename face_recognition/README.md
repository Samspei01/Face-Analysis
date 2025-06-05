# Face Recognition Model Evaluation on LFW Dataset

This project provides a comprehensive evaluation framework for multiple face recognition models using the Labeled Faces in the Wild (LFW) dataset. The evaluation includes balanced positive and negative pairs to provide meaningful performance metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## 🔍 Overview

This evaluation framework tests face recognition models on the LFW dataset using a balanced approach that includes both positive pairs (same person) and negative pairs (different people). This balanced evaluation provides more reliable ROC AUC metrics compared to traditional positive-only evaluations.

### Key Features

- **Balanced Dataset**: Equal numbers of positive and negative pairs (3,000 each)
- **Multiple Models**: Support for Dlib, FaceNet, MobileFaceNet, and ArcFace
- **Comprehensive Metrics**: ROC AUC, accuracy, timing analysis, and visualizations
- **Backward Compatibility**: Support for legacy pair formats
- **Detailed Analysis**: Per-pair timing, similarity distributions, and example visualizations

## 📊 Dataset

### LFW Dataset
- **Source**: Labeled Faces in the Wild (LFW) dataset
- **Format**: lfw-deepfunneled version for better face alignment
- **Structure**: Individual directories for each person containing their face images

### Balanced Pairs Dataset
- **File**: `pairs_balanced.csv`
- **Format**: `name1,imagenum1,name2,imagenum2,label`
- **Size**: 6,000 total pairs (3,000 positive + 3,000 negative)
- **Generation**: Created using `generate_balanced_pairs.py`

#### Dataset Statistics
- **Positive Pairs (label=1)**: 3,000 pairs of the same person
- **Negative Pairs (label=0)**: 3,000 pairs of different people
- **Total Persons**: Varies based on LFW dataset availability
- **Reproducible**: Uses random seed (42) for consistent results

## 🤖 Models Evaluated

### 1. Dlib ResNet Face Recognition
- **Model File**: `dlib_face_recognition_resnet_model_v1.dat`
- **Landmark Detector**: `shape_predictor_68_face_landmarks.dat`
- **Script**: `scripts/dlib_evaluation.py`
- **Performance**: ⭐⭐⭐⭐⭐ (Excellent)

### 2. FaceNet
- **Model File**: `facenet.onnx`
- **Script**: `scripts/facenet.py`
- **Performance**: ⭐⭐ (Poor - may need model debugging)

### 3. MobileFaceNet
- **Model Directory**: `mobilefacenet_scripted/`
- **Script**: `scripts/mobilefacenet_evaluation.py`
- **Performance**: ⭐⭐ (Poor - may need model debugging)

### 4. ArcFace (Optional)
- **Model File**: `arcfaceresnet100-8.onnx`
- **Status**: Ready for evaluation

## 🛠 Installation

### Prerequisites
```bash
pip install opencv-python
pip install dlib
pip install torch torchvision
pip install onnxruntime
pip install scikit-learn
pip install matplotlib
pip install numpy
pip install scipy
```

### Setup
1. Clone or download this repository
2. Download the LFW dataset (lfw-deepfunneled version)
3. Download the required model files and place them in `recognize_models/`
4. Generate balanced pairs or use the existing `pairs_balanced.csv`

## 🚀 Usage

### Generate Balanced Pairs Dataset
```bash
python generate_balanced_pairs.py
```

### Run Individual Model Evaluations

#### Dlib Evaluation
```bash
python scripts/dlib_evaluation.py \
    --lfw_dir ./lfw-deepfunneled \
    --pairs_file ./pairs_balanced.csv \
    --sample_size 500 \
    --output_dir ./results/dlib
```

#### FaceNet Evaluation
```bash
python scripts/facenet.py \
    --lfw_dir ./lfw-deepfunneled \
    --pairs_file ./pairs_balanced.csv \
    --sample_size 500 \
    --output_dir ./results/facenet
```

#### MobileFaceNet Evaluation
```bash
python scripts/mobilefacenet_evaluation.py \
    --lfw_dir ./lfw-deepfunneled \
    --pairs_file ./pairs_balanced.csv \
    --sample_size 500 \
    --output_dir ./results/mobilefacenet
```

### Command Line Options
- `--lfw_dir`: Path to LFW dataset directory
- `--pairs_file`: Path to pairs CSV file
- `--sample_size`: Number of pairs to evaluate (-1 for all)
- `--output_dir`: Directory for saving results
- `--debug`: Enable verbose debugging output

## 📈 Results

### Performance Summary (500 pairs evaluation)

| Model | ROC AUC | Accuracy | Avg. Time/Pair | Performance Rating |
|-------|---------|----------|----------------|-------------------|
| **Dlib ResNet** | **0.994** | **98.6%** | **0.12s** | ⭐⭐⭐⭐⭐ Excellent |
| FaceNet | 0.540 | 53.8% | 1.34s | ⭐⭐ Poor |
| MobileFaceNet | 0.565 | 57.2% | 0.79s | ⭐⭐ Poor |

### Detailed Results

#### Dlib ResNet (Best Performer)
- **ROC AUC**: 0.994 (Near perfect discrimination)
- **Accuracy**: 98.6% at threshold 0.9
- **Speed**: ~0.12 seconds per pair
- **Success Rate**: 100% (no failed pairs)
- **Status**: ✅ Excellent performance

#### FaceNet
- **ROC AUC**: 0.540 (Poor discrimination)
- **Accuracy**: 53.8% at threshold 0.06
- **Speed**: ~1.34 seconds per pair
- **Success Rate**: 100% (no failed pairs)
- **Status**: ⚠️ Needs investigation/debugging

#### MobileFaceNet
- **ROC AUC**: 0.565 (Poor discrimination)
- **Accuracy**: 57.2% at threshold 0.995
- **Speed**: ~0.79 seconds per pair
- **Success Rate**: 100% (no failed pairs)
- **Status**: ⚠️ Needs investigation/debugging

### Generated Visualizations

Each evaluation produces comprehensive visualizations:

- **ROC Curves**: Model discrimination capability
- **Similarity Distributions**: Distribution of similarity scores
- **Timing Analysis**: Processing time distributions and trends
- **Example Pairs**: Visual examples of processed pairs
- **Performance Summary**: Overall metrics visualization

## 📁 File Structure

```
face_recognition/
├── README.md                           # This file
├── generate_balanced_pairs.py          # Script to create balanced dataset
├── pairs_balanced.csv                  # Balanced evaluation dataset (6K pairs)
├── pairs.csv                          # Legacy pairs file
├── lfw-deepfunneled/                  # LFW dataset directory
│   └── lfw-deepfunneled/              # Actual dataset
│       ├── Aaron_Eckhart/             # Person directories
│       ├── Aaron_Guiel/
│       └── ...
├── recognize_models/                   # Pre-trained model files
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   ├── facenet.onnx
│   ├── arcfaceresnet100-8.onnx
│   ├── shape_predictor_68_face_landmarks.dat
│   └── mobilefacenet_scripted/
├── scripts/                           # Evaluation scripts
│   ├── dlib_evaluation.py             # ✅ Updated for balanced pairs
│   ├── facenet.py                     # ✅ Updated for balanced pairs
│   ├── mobilefacenet_evaluation.py    # ✅ Updated for balanced pairs
│   └── __pycache__/
└── results/                           # Evaluation results
    ├── dlib/                          # ✅ Excellent results
    │   ├── results.json               # Metrics summary
    │   ├── roc_curve.png             # ROC curve visualization
    │   ├── similarity_distribution.png
    │   ├── processing_time_analysis.png
    │   └── ...
    ├── facenet/                       # ⚠️ Poor performance
    │   ├── results.json
    │   ├── roc_curve.png
    │   └── ...
    ├── mobilefacenet/                 # ⚠️ Poor performance
    │   ├── mobilefacenet_evaluation_results.json
    │   ├── mobilefacenet_roc_curve.png
    │   └── ...
    └── arcface/                       # Ready for evaluation
```

## 🔧 Technical Details

### Balanced Pair Generation
The `generate_balanced_pairs.py` script creates a balanced evaluation dataset by:

1. **Scanning LFW Dataset**: Identifies all persons and their available images
2. **Positive Pairs**: Randomly selects two different images of the same person
3. **Negative Pairs**: Randomly selects images from two different persons
4. **Balancing**: Ensures equal numbers of positive and negative pairs
5. **Shuffling**: Randomizes pair order for unbiased evaluation

### Evaluation Metrics
- **ROC AUC**: Area Under the Receiver Operating Characteristic curve
- **Accuracy**: Percentage of correctly classified pairs at optimal threshold
- **Timing**: Average, minimum, maximum processing times per pair
- **Success Rate**: Percentage of pairs successfully processed

### CSV Format Compatibility
All evaluation scripts support both formats:
- **Balanced Format**: `name1,imagenum1,name2,imagenum2,label` (5 columns)
- **Legacy Format**: `name,imagenum1,imagenum2` (3 columns, positive pairs only)

## 🚨 Known Issues

### FaceNet & MobileFaceNet Performance
Both models show poor performance (ROC AUC ~0.55, accuracy ~55%). This suggests:
- **Model Loading Issues**: Models may not be loading correctly
- **Preprocessing Problems**: Face preprocessing may not match training requirements
- **Threshold Issues**: Default thresholds may be inappropriate
- **Model Version**: May need different model versions or formats

### Recommendations for Improvement
1. **Verify Model Loading**: Ensure models load correctly and match expected input formats
2. **Check Preprocessing**: Validate face detection and preprocessing pipeline
3. **Threshold Tuning**: Experiment with different similarity thresholds
4. **Model Versions**: Try different model versions or architectures
5. **Debug Mode**: Use `--debug` flag for detailed troubleshooting

## 🤝 Contributing

### Adding New Models
1. Create evaluation script in `scripts/` directory
2. Implement `load_pairs()` function supporting balanced CSV format
3. Generate comprehensive results and visualizations
4. Update this README with performance results

### Improving Existing Models
1. Debug poor-performing models (FaceNet, MobileFaceNet)
2. Optimize preprocessing pipelines
3. Experiment with different thresholds and parameters
4. Add additional evaluation metrics

## 📚 References

- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
- [Dlib Face Recognition](http://dlib.net/face_recognition.py.html)
- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
- [MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification](https://arxiv.org/abs/1804.07573)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

---

**Status**: ✅ Dlib evaluation complete and excellent | ⚠️ FaceNet and MobileFaceNet need debugging | 🔄 ArcFace ready for evaluation

**Last Updated**: June 5, 2025
