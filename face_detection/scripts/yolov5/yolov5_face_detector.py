#!/usr/bin/env python3
"""
YOLOv5 Face Detector Evaluation Script with Highest Confidence Filtering

This script evaluates the performance of the YOLOv5 face detection model
on a dataset of images, providing metrics and visualizations. It uses a filtering
approach that keeps only the highest confidence face per image, which improves 
accuracy on datasets like LFW that typically have one face per image.

Usage:
    python test_yolov5_face_detector_highest_confidence.py [--dataset DATASET] [--sample_size SIZE] 
                                                          [--visualize NUM] [--output_dir DIR]
"""

import os
import sys
import time
import argparse
import random
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test YOLOv5 face detector on LFW dataset with highest confidence filtering')
    parser.add_argument('--dataset', type=str,
                      default='../../../4/lfw-deepfunneled/lfw-deepfunneled',
                      help='Path to the LFW dataset')
    parser.add_argument('--sample_size', type=int, default=1000,
                      help='Number of images to sample for evaluation (default: 1000)')
    parser.add_argument('--visualize', type=int, default=10,
                      help='Number of detection results to visualize')
    parser.add_argument('--output_dir', type=str, 
                      default='../../results/yolov5_face',
                      help='Directory to save evaluation results')
    parser.add_argument('--conf_thres', type=float, default=0.5,
                      help='Confidence threshold for detections')
    return parser.parse_args()

def load_images(dataset_path, sample_size=1000):
    """Load image paths from the dataset"""
    print(f"Loading images from dataset at: {dataset_path}")
    all_images = []
    
    # Handle both directory of images and directory of person directories
    if os.path.isdir(dataset_path):
        # First check if this is a directory of person directories
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        if subdirs:
            # This is likely a directory of person directories (LFW structure)
            for person_dir in subdirs:
                person_path = os.path.join(dataset_path, person_dir)
                person_images = glob(os.path.join(person_path, "*.jpg"))
                all_images.extend(person_images)
        else:
            # Direct directory of images
            all_images = glob(os.path.join(dataset_path, "*.jpg"))
    else:
        print(f"Error: {dataset_path} is not a valid directory")
        sys.exit(1)
    
    print(f"Found {len(all_images)} total images")
    
    # Sample images if needed
    if 0 < sample_size < len(all_images):
        print(f"Sampling {sample_size} images for evaluation")
        return random.sample(all_images, sample_size)
    else:
        print(f"Using all {len(all_images)} images for evaluation")
        return all_images

def load_face_detection_model():
    """Load YOLOv5 face detection model"""
    print("Loading YOLOv5 face detection model...")
    
    # Check multiple possible paths for the face model
    model_paths = [
        "../../Detect_models/yolov5-face/weights/yolov5n-face.pt",
        "../Detect_models/yolov5-face/weights/yolov5n-face.pt",
        "../../yolov5-face.pt",
        "../yolov5-face.pt",
        "./yolov5-face.pt"
    ]
    
    # Try to find and load face-specific model
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Found face model at: {model_path}")
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                print("YOLOv5-face detection model loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading from {model_path}: {e}")
    
    # If no face model found, try to find a standard YOLOv5 model
    print("No face-specific model found. Trying standard YOLOv5 model...")
    try:
        # Look for the standard model in current directory
        if os.path.exists("../../yolov5s.pt"):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path="../../yolov5s.pt", force_reload=True)
        elif os.path.exists("../yolov5s.pt"):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path="../yolov5s.pt", force_reload=True)
        elif os.path.exists("./yolov5s.pt"):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path="./yolov5s.pt", force_reload=True)
        else:
            # Download from torch hub if not found locally
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        print("Standard YOLOv5s model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        sys.exit(1)

def detect_faces(model, img, conf_thres=0.5):
    """Detect faces in an image using YOLOv5 face detector with highest confidence filtering"""
    # Run inference
    start_time = time.time()
    results = model(img)
    detection_time = time.time() - start_time
    
    # Process results
    detections = results.xyxy[0]
    
    # For standard YOLOv5 model, filter only for person class (0)
    # For face model, all detections are faces
    person_detections = []
    for detection in detections:
        class_id = int(detection[5])
        conf = float(detection[4])
        # Accept class 0 (person) or any class if we're using a face model
        # When using standard model, we're looking for faces in person detections
        if conf >= conf_thres and (class_id == 0 or "face" in str(model).lower()):
            person_detections.append(detection)
    
    # If we have detections, return only the highest confidence one
    faces = []
    if person_detections:
        # Get detection with highest confidence
        best_detection = max(person_detections, key=lambda d: d[4])
        x1, y1, x2, y2, confidence, class_id = best_detection.cpu().numpy()
        faces.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(confidence)
        })
    
    return faces, detection_time

def visualize_detection(img, faces, title="YOLOv5 Face Detection"):
    """Visualize detection results on image"""
    img_vis = img.copy()
    height, width = img_vis.shape[:2]
    
    # Add title banner
    banner_height = 40
    banner = np.zeros((banner_height, width, 3), dtype=np.uint8)
    cv2.putText(banner, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    img_vis = np.vstack([banner, img_vis])
    
    # Draw bounding boxes
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        confidence = face['confidence']
        
        # Adjust y-coordinates for banner
        y1 += banner_height
        y2 += banner_height
        
        # Draw rectangle
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with confidence
        label = f"Face: {confidence:.2f}"
        cv2.putText(img_vis, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_vis

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load face detection model
    face_detector = load_face_detection_model()
    
    # Load test images
    images = load_images(args.dataset, args.sample_size)
    
    # Stats for evaluation
    total_images = len(images)
    images_with_faces = 0
    total_faces_detected = 0
    detection_times = []
    face_counts = []
    
    # Sample images to visualize
    if args.visualize > 0:
        visualize_indices = random.sample(range(len(images)), min(args.visualize, len(images)))
    else:
        visualize_indices = []
    
    print("Running face detection using YOLOv5 face detector with highest confidence filtering...")
    
    # Process images
    for i, img_path in enumerate(tqdm(images, desc="Processing images")):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image at {img_path}")
            continue
            
        # Detect faces
        faces, detection_time = detect_faces(face_detector, img, args.conf_thres)
        detection_times.append(detection_time)
        
        # Update statistics
        face_count = len(faces)
        face_counts.append(face_count)
        
        if face_count > 0:
            images_with_faces += 1
            total_faces_detected += face_count
        
        # Visualize if selected
        if i in visualize_indices:
            vis_index = visualize_indices.index(i)
            img_vis = visualize_detection(img, faces)
            vis_path = os.path.join(args.output_dir, f"detection_{vis_index+1}.png")
            cv2.imwrite(vis_path, img_vis)
    
    # Calculate statistics
    detection_rate = images_with_faces / total_images if total_images > 0 else 0
    avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
    
    # Calculate face counts
    one_face_images = sum(1 for count in face_counts if count == 1)
    multi_face_images = sum(1 for count in face_counts if count > 1)
    zero_face_images = sum(1 for count in face_counts if count == 0)
    
    # Calculate percentiles for detection time
    if detection_times:
        percentiles = np.percentile(detection_times, [50, 90, 95, 99])
    else:
        percentiles = [0, 0, 0, 0]
    
    # Print results
    print(f"=" * 80)
    print("YOLOv5 FACE DETECTION EVALUATION RESULTS")
    print(f"=" * 80)
    print("Detection Performance:")
    print(f"  • Detection Rate: {detection_rate * 100:.2f}%")
    print(f"  • True Accuracy (exactly one face): {one_face_images / total_images * 100:.2f}%")
    print(f"  • False Positive Rate: {multi_face_images / total_images * 100:.2f}%")
    print("Detection Counts:")
    print(f"  • Total Images: {total_images}")
    print(f"  • Images with Faces: {images_with_faces} ({images_with_faces / total_images * 100:.2f}%)")
    print(f"  • Images with No Faces: {zero_face_images} ({zero_face_images / total_images * 100:.2f}%)")
    print(f"  • Images with One Face: {one_face_images} ({one_face_images / total_images * 100:.2f}%)")
    print(f"  • Images with Multiple Faces: {multi_face_images} ({multi_face_images / total_images * 100:.2f}%)")
    print("Speed Metrics:")
    print(f"  • Average detection time: {avg_detection_time:.4f} seconds per image")
    print(f"  • Processing speed: {1/avg_detection_time:.2f} images/second")
    print(f"  • P50 (median) detection time: {percentiles[0]:.4f} seconds")
    print(f"  • P90 detection time: {percentiles[1]:.4f} seconds")
    print(f"  • P95 detection time: {percentiles[2]:.4f} seconds")
    print(f"  • P99 detection time: {percentiles[3]:.4f} seconds")
    
    # Save metrics to CSV
    metrics_file = os.path.join(args.output_dir, "yolov5_face_metrics.csv")
    with open(metrics_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"model_name,YOLOv5 Face Detection\n")
        f.write(f"total_images,{total_images}\n")
        f.write(f"images_with_faces,{images_with_faces}\n")
        f.write(f"images_with_no_faces,{zero_face_images}\n")
        f.write(f"face_detection_rate,{detection_rate}\n")
        f.write(f"total_faces_detected,{total_faces_detected}\n")
        f.write(f"avg_faces_per_image,{total_faces_detected/total_images if total_images > 0 else 0}\n")
        f.write(f"one_face_images,{one_face_images}\n")
        f.write(f"multi_face_images,{multi_face_images}\n")
        f.write(f"zero_face_images,{zero_face_images}\n")
        f.write(f"avg_detection_time,{avg_detection_time}\n")
        f.write(f"median_detection_time,{percentiles[0]}\n")
        f.write(f"p90_detection_time,{percentiles[1]}\n")
        f.write(f"p95_detection_time,{percentiles[2]}\n")
        f.write(f"p99_detection_time,{percentiles[3]}\n")
        f.write(f"images_per_second,{1/avg_detection_time if avg_detection_time > 0 else 0}\n")
    
    # Create detection time histogram
    plt.figure(figsize=(10, 6))
    plt.hist(detection_times, bins=30, alpha=0.7, color='blue')
    plt.title("YOLOv5 Face Detection Time Distribution")
    plt.xlabel("Detection Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "yolov5_detection_time_distribution.png"))
    plt.close()
    
    print(f"Evaluation complete! Results saved to {args.output_dir}")
    
if __name__ == "__main__":
    main()
