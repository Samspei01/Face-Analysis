#!/usr/bin/env python3
"""
MobileNet SSD Detector Model Evaluation Script with Highest Confidence Filtering

This script evaluates the performance of a MobileNet SSD model on the LFW dataset,
using a filtering approach that keeps only the highest confidence detection per image.

This improves accuracy on the LFW dataset which typically has one face per image.
"""


import os
import time
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import sys
from pathlib import Path
import csv
import shutil
import urllib.request

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test MobileNet SSD detector on LFW dataset with highest confidence filtering')
    parser.add_argument('--model_dir', type=str, default='Detect_models/public/ssd_mobilenet_v1_coco',
                      help='Path to the MobileNet SSD model directory')
    parser.add_argument('--dataset', type=str, default='4/lfw-deepfunneled/lfw-deepfunneled',
                      help='Path to the LFW dataset')
    parser.add_argument('--sample_size', type=int, default=1000,
                      help='Number of images to sample for evaluation')
    parser.add_argument('--output_dir', type=str, default='mobilenet_ssd_evaluation_results_highest_confidence',
                      help='Directory to save evaluation results')
    parser.add_argument('--visualize', type=int, default=10,
                      help='Number of images to visualize results for')
    parser.add_argument('--conf_thres', type=float, default=0.5,
                      help='Confidence threshold for detections')
    parser.add_argument('--full_dataset', action='store_true',
                      help='Process the full dataset instead of sampling')
    parser.add_argument('--image_size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                      help='Specify the size of the image as width and height (used for MobileNet input size)')
    return parser.parse_args()

def download_model_files(model_dir):
    """
    Download the pre-trained MobileNet SSD face detector model files if they don't exist.
    
    Args:
        model_dir (str): Directory to save the model files
    
    Returns:
        bool: True if files are available, False otherwise
    """
    os.makedirs(model_dir, exist_ok=True)
    
    model_files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    for filename, url in model_files.items():
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return False
    
    return True

def load_model(model_dir):
    """
    Load the MobileNet SSD face detector model.

    Args:
        model_dir (str): Path to the directory containing the model files.

    Returns:
        cv2.dnn.Net: The loaded face detector model.
        None: No classes list needed for this model.
    """
    # First try to use MobileNet model from the specified directory
    try:
        # Path to frozen detection graph
        nested_model_path = os.path.join(model_dir, 'ssd_mobilenet_v1_coco_2018_01_28', 'frozen_inference_graph.pb')
        direct_model_path = os.path.join(model_dir, 'frozen_inference_graph.pb')
        config_path = os.path.join(model_dir, 'config.pbtxt')
        
        # Use nested path if it exists, otherwise use direct path
        if os.path.exists(nested_model_path):
            model_path = nested_model_path
            print(f"Loading MobileNet SSD model from {model_path}...")
            # Test if model is loadable
            with open(model_path, 'rb') as f:
                if len(f.read(1024)) < 1024:
                    raise ValueError("Model file is too small, likely corrupted")
        elif os.path.exists(direct_model_path):
            model_path = direct_model_path
            print(f"Loading MobileNet SSD model from {model_path}...")
            # Test if model is loadable
            with open(model_path, 'rb') as f:
                if len(f.read(1024)) < 1024:
                    raise ValueError("Model file is too small, likely corrupted")
        else:
            raise FileNotFoundError(f"Model files not found. Checked: {nested_model_path} and {direct_model_path}")
        
        # Check if config exists
        if os.path.exists(config_path):
            print(f"Using config from {config_path}")
            # Load model using TensorFlow backend with config
            model = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        else:
            print("No config file found, using default configuration")
            # Load model using TensorFlow backend without config
            model = cv2.dnn.readNetFromTensorflow(model_path)
        
        # Check if CUDA is available and enable it
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA backend for inference")
        else:
            print("CUDA not available, using CPU for inference")
        
        # Test if model works
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(test_img, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True)
        model.setInput(blob)
        _ = model.forward()  # This will throw an exception if the model has issues
        
        print("MobileNet SSD model loaded successfully")
        return model, None  # No class list needed
    
    except Exception as e:
        print(f"Failed to load MobileNet SSD model: {e}")
        print("Falling back to OpenCV DNN face detector...")
    
    # Fallback to OpenCV's face detector
    face_detector_dir = os.path.join("Detect_models", "custom_detector")
    
    # Ensure face detector files are available
    if not download_model_files(face_detector_dir):
        raise FileNotFoundError("Could not download face detector model files")
    
    # Path to model files
    prototxt_path = os.path.join(face_detector_dir, "deploy.prototxt")
    model_path = os.path.join(face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(model_path) or not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Model files not found at {model_path} or {prototxt_path}")
    
    print(f"Loading OpenCV DNN face detector model...")
    
    # Load model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Check if CUDA is available and enable it
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA backend for inference")
    else:
        print("CUDA not available, using CPU for inference")
    
    print("OpenCV DNN face detector model loaded successfully")
    return model, None  # No class list needed

def load_sample_images(dataset_path, sample_size, full_dataset=False):
    print(f"Loading images from dataset at: {dataset_path}")
    all_images = []
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            image_files = glob(os.path.join(person_path, '*.jpg'))
            all_images.extend(image_files)
    
    print(f"Found {len(all_images)} total images")
    if not full_dataset and len(all_images) > sample_size:
        random.shuffle(all_images)
        sample_images = all_images[:sample_size]
        print(f"Sampled {sample_size} images for evaluation")
    else:
        sample_images = all_images
        print(f"Using all {len(sample_images)} images for evaluation")
    
    return sample_images

def detect_faces(model, classes, image_paths, conf_thres=0.5, filter_highest_confidence=True):
    detections = []
    times = []
    
    # Determine which model we're using based on the classes parameter
    if classes is not None:
        # Using MobileNet SSD object detector
        print(f"Running person detection using MobileNet SSD model with highest confidence filtering...")
        is_face_detector = False
        person_class_id = 1  # Person class ID in COCO dataset
    else:
        # Using OpenCV DNN face detector
        print(f"Running face detection using OpenCV DNN face detector with highest confidence filtering...")
        is_face_detector = True
    
    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        h, w = img.shape[:2]
        
        if is_face_detector:
            # OpenCV DNN face detector expects specific preprocessing
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                                        (300, 300), (104.0, 177.0, 123.0))
        else:
            # MobileNet SSD preprocessing
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1/127.5, 
                                        (300, 300), (127.5, 127.5, 127.5), swapRB=True)
        
        # Pass the blob through the network and get detections
        start_time = time.time()
        model.setInput(blob)
        output = model.forward()
        end_time = time.time()
        detection_time = end_time - start_time
        times.append(detection_time)
        
        # Parse results
        detected_faces = []
        for i in range(output.shape[2]):
            confidence = float(output[0, 0, i, 2])
            
            # Apply confidence threshold
            if confidence >= conf_thres:
                if is_face_detector:
                    # Face detector directly outputs face boxes
                    box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype('int')
                    
                    # Ensure coordinates are within image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Only add valid detections
                    if x2 > x1 and y2 > y1:
                        detected_faces.append({'rect': (x1, y1, x2-x1, y2-y1), 'confidence': float(confidence)})
                else:
                    # MobileNet SSD: check if detection is a person
                    class_id = int(output[0, 0, i, 1])
                    if class_id == person_class_id:
                        # Get bounding box coordinates
                        box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype('int')
                        
                        # Ensure coordinates are within image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        
                        # Only add valid detections
                        if x2 > x1 and y2 > y1:
                            detected_faces.append({'rect': (x1, y1, x2-x1, y2-y1), 'confidence': float(confidence)})
        
        # Filter to keep only the highest confidence face if requested
        if filter_highest_confidence and len(detected_faces) > 1:
            highest_conf_face = max(detected_faces, key=lambda x: x['confidence'])
            faces = [highest_conf_face]
        else:
            faces = detected_faces
            
        detections.append((image_path, img, faces))
    
    return detections, times

def evaluate_performance(detections, times, model_type="MobileNet SSD"):
    """Calculate metrics based on detection results"""
    # Timing metrics
    avg_time = sum(times) / len(times) if times else 0
    faces_per_second = 1.0 / avg_time if avg_time > 0 else 0
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2] if times else 0
    p90 = sorted_times[int(len(sorted_times) * 0.9)] if times else 0
    p95 = sorted_times[int(len(sorted_times) * 0.95)] if times else 0
    p99 = sorted_times[int(len(sorted_times) * 0.99)] if times else 0
    
    # Detection counts
    total_images = len(detections)
    images_with_faces = sum(1 for _, _, faces in detections if len(faces) > 0)
    images_with_no_faces = total_images - images_with_faces
    total_faces = sum(len(faces) for _, _, faces in detections)
    
    # Faces per image distribution
    face_counts = [len(faces) for _, _, faces in detections]
    one_face_images = sum(1 for count in face_counts if count == 1)
    multi_face_images = sum(1 for count in face_counts if count > 1)
    zero_face_images = sum(1 for count in face_counts if count == 0)
    
    # LFW should have exactly one face per image
    expected_faces = total_images
    detection_rate = images_with_faces / total_images if total_images > 0 else 0
    
    # Additional accuracy metrics
    true_accuracy = one_face_images / total_images if total_images > 0 else 0
    precision = one_face_images / total_faces if total_faces > 0 else 0
    extra_faces = total_faces - one_face_images
    false_positive_rate = extra_faces / total_images if total_images > 0 else 0
    
    # Confidence scores
    all_confidences = []
    for _, _, faces in detections:
        for face in faces:
            all_confidences.append(face['confidence'])
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    min_confidence = min(all_confidences) if all_confidences else 0
    max_confidence = max(all_confidences) if all_confidences else 0
    
    # Width/height ratio of detected faces
    face_sizes = []
    face_ratios = []
    for _, _, faces in detections:
        for face in faces:
            rect = face['rect']
            width, height = rect[2], rect[3]
            if height > 0:
                face_sizes.append((width, height))
                face_ratios.append(width / height)
    avg_ratio = sum(face_ratios) / len(face_ratios) if face_ratios else 0
    
    return {
        'model_type': model_type,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'true_accuracy': true_accuracy,
        'precision': precision,
        'avg_time': avg_time,
        'faces_per_second': faces_per_second,
        'total_images': total_images,
        'images_with_faces': images_with_faces,
        'images_with_no_faces': images_with_no_faces,
        'one_face_images': one_face_images,
        'multi_face_images': multi_face_images,
        'zero_face_images': zero_face_images,
        'total_faces': total_faces,
        'expected_faces': expected_faces,
        'extra_faces': extra_faces,
        'p50_time': p50,
        'p90_time': p90,
        'p95_time': p95,
        'p99_time': p99,
        'avg_confidence': avg_confidence,
        'min_confidence': min_confidence,
        'max_confidence': max_confidence,
        'avg_ratio': avg_ratio
    }

def visualize_results(detections, num_to_visualize, output_dir, model_type="MobileNet SSD"):
    """Create visualizations of the detection results"""
    print(f"Creating {num_to_visualize} detection visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort detections by number of faces, prioritize interesting cases
    sorted_detections = sorted(detections, key=lambda d: len(d[2]), reverse=True)
    
    # Ensure mixed sampling - get both multiple detections and single detections
    multi_detections = [d for d in sorted_detections if len(d[2]) > 1]
    single_detections = [d for d in sorted_detections if len(d[2]) == 1]
    no_detections = [d for d in sorted_detections if len(d[2]) == 0]
    
    # Select a mix of samples, prioritizing interesting cases
    visualize_detections = []
    
    # Try to get 1/3 from each category if available
    target_per_category = max(1, num_to_visualize // 3)
    
    visualize_detections.extend(multi_detections[:target_per_category])
    visualize_detections.extend(single_detections[:target_per_category])
    visualize_detections.extend(no_detections[:target_per_category])
    
    # If we don't have enough, fill with whatever is available
    remaining = num_to_visualize - len(visualize_detections)
    if remaining > 0:
        all_remaining = sorted_detections[:]
        for d in visualize_detections:
            if d in all_remaining:
                all_remaining.remove(d)
        visualize_detections.extend(all_remaining[:remaining])
    
    # Trim to exact number if we went over
    visualize_detections = visualize_detections[:num_to_visualize]
    
    # Now create the visualizations
    for i, (image_path, img, faces) in enumerate(visualize_detections):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        
        # Draw bounding boxes for all detected faces
        for face in faces:
            x, y, w, h = face['rect']
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add confidence score text
            confidence = face['confidence']
            ax.text(x, y - 10, f"Conf: {confidence:.2f}", color='white', fontsize=12, 
                    backgroundcolor='red')
        
        ax.set_title(f"{model_type} Detection - {os.path.basename(image_path)}\n"
                    f"{len(faces)} faces detected")
        ax.axis('off')
        
        # Save the figure
        output_path = os.path.join(output_dir, f"detection_{i+1}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def generate_graphs(detections, times, output_dir, model_type="MobileNet SSD"):
    """Generate graphs for visualizing the evaluation results"""
    print("Generating performance graphs...")
    os.makedirs(output_dir, exist_ok=True)

    # Graph 1: Detection Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(times, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=sum(times) / len(times), color='red', linestyle='dashed', 
                linewidth=2, label=f'Average: {sum(times) / len(times):.4f}s')
    plt.xlabel('Detection Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'{model_type} Detection Time Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_time_distribution.png'))
    plt.close()

    # Graph 2: Number of Faces per Image Distribution
    face_counts = [len(faces) for _, _, faces in detections]
    max_faces = max(face_counts) if face_counts else 0
    bins = min(max_faces + 1, 10)
    
    plt.figure(figsize=(10, 6))
    plt.hist(face_counts, bins=range(bins + 1), alpha=0.7, color='green', align='left', rwidth=0.8)
    plt.xlabel('Number of Faces Detected')
    plt.ylabel('Number of Images')
    plt.title(f'{model_type} Face Count Distribution')
    plt.xticks(range(bins))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'face_count_distribution.png'))
    plt.close()

    # Graph 3: Confidence Score Distribution
    all_confidences = []
    for _, _, faces in detections:
        for face in faces:
            all_confidences.append(face['confidence'])
    
    if all_confidences:
        plt.figure(figsize=(10, 6))
        plt.hist(all_confidences, bins=20, alpha=0.7, color='purple')
        plt.axvline(x=sum(all_confidences) / len(all_confidences), color='red', linestyle='dashed', 
                    linewidth=2, label=f'Average: {sum(all_confidences) / len(all_confidences):.4f}')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'{model_type} Confidence Score Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()

    # Graph 4: Detection Time vs Number of Faces
    plt.figure(figsize=(10, 6))
    plt.scatter(face_counts, times, alpha=0.5)
    plt.xlabel('Number of Faces')
    plt.ylabel('Detection Time (seconds)')
    plt.title(f'{model_type} Detection Time vs. Number of Faces')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_faces.png'))
    plt.close()
    
    # Graph 5: Size Distribution
    face_sizes = []
    for _, _, faces in detections:
        for face in faces:
            rect = face['rect']
            face_sizes.append(max(rect[2], rect[3]))  # Using max dimension
    
    if face_sizes:
        plt.figure(figsize=(10, 6))
        plt.hist(face_sizes, bins=30, alpha=0.7, color='orange')
        plt.xlabel('Face Size (pixels)')
        plt.ylabel('Frequency')
        plt.title(f'{model_type} Face Size Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'size_distribution.png'))
        plt.close()

def print_results(metrics):
    """Print the evaluation results to the console"""
    print("\n" + "=" * 80)
    print(f"MobileNet SSD FACE DETECTION EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nDetection Performance:")
    print(f"  • Detection Rate: {metrics['detection_rate']:.2%}")
    print(f"  • True Accuracy (exactly one face): {metrics['true_accuracy']:.2%}")
    print(f"  • False Positive Rate: {metrics['false_positive_rate']:.2%}")
    
    print(f"\nDetection Counts:")
    print(f"  • Total Images: {metrics['total_images']}")
    print(f"  • Images with Faces: {metrics['images_with_faces']} ({metrics['images_with_faces']/metrics['total_images']:.2%})")
    print(f"  • Images with No Faces: {metrics['images_with_no_faces']} ({metrics['images_with_no_faces']/metrics['total_images']:.2%})")
    print(f"  • Images with One Face: {metrics['one_face_images']} ({metrics['one_face_images']/metrics['total_images']:.2%})")
    print(f"  • Images with Multiple Faces: {metrics['multi_face_images']} ({metrics['multi_face_images']/metrics['total_images']:.2%})")
    
    print(f"\nConfidence Scores:")
    print(f"  • Average Confidence: {metrics['avg_confidence']:.4f}")
    print(f"  • Min Confidence: {metrics['min_confidence']:.4f}")
    print(f"  • Max Confidence: {metrics['max_confidence']:.4f}")
    
    print(f"\nTiming Performance:")
    print(f"  • Average Processing Time: {metrics['avg_time']:.4f} seconds per image")
    print(f"  • Processing Speed: {metrics['faces_per_second']:.2f} faces per second")
    print(f"  • P50 (Median) Time: {metrics['p50_time']:.4f} seconds")
    print(f"  • P90 Time: {metrics['p90_time']:.4f} seconds")
    print(f"  • P95 Time: {metrics['p95_time']:.4f} seconds")
    print(f"  • P99 Time: {metrics['p99_time']:.4f} seconds")

def save_metrics(metrics, output_dir):
    """Save metrics to a CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'metrics.csv')
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    
    print(f"Metrics saved to {output_file}")

def main(args=None):
    # If args is None, parse from command line
    if args is None:
        args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    try:
        model, classes = load_model(getattr(args, 'model_dir', 'Detect_models/public/ssd_mobilenet_v1_coco'))
        
        # Determine model type for reporting
        if classes is not None:
            model_type = "MobileNet SSD"
            print("Successfully loaded MobileNet SSD model")
        else:
            model_type = "OpenCV Face Detector"
            print("Successfully loaded OpenCV Face Detector model")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load sample images
    image_paths = load_sample_images(args.dataset, args.sample_size, getattr(args, 'full_dataset', False))
    
    # Run detection with highest confidence filtering
    start_time = time.time()
    detections, times = detect_faces(
        model, classes, image_paths, 
        conf_thres=getattr(args, 'conf_thres', 0.5),
        filter_highest_confidence=True
    )
    total_time = time.time() - start_time
    
    # Calculate metrics
    metrics = evaluate_performance(detections, times, model_type=model_type)
    metrics['total_processing_time'] = total_time
    metrics['model_name'] = model_type
    
    # Output results
    print_results(metrics)
    save_metrics(metrics, args.output_dir)
    
    # Generate visualizations
    try:
        visualize_results(detections, getattr(args, 'visualize', 10), args.output_dir, model_type=model_type)
        generate_graphs(detections, times, args.output_dir, model_type=model_type)
        print(f"Visualizations saved to {args.output_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        
    print("\nEvaluation complete!")
    
    return metrics

if __name__ == "__main__":
    main()
