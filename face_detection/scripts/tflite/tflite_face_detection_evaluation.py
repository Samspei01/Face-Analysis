#!/usr/bin/env python3
"""
TFLite Face Detection Evaluation Script

This script evaluates the face_detection_front.tflite model on a dataset of your choice.
It measures performance metrics, creates visualizations, and generates CSV summary metrics
in a format similar to the face detection evaluation scripts.

Usage:
    python tflite_face_detection_evaluation.py --dataset <path_to_dataset> --sample_size <number_of_images>
"""

import os
import sys
import argparse
import time
import random
import csv
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate face detection')
    parser.add_argument('--dataset', type=str,
                      default='face/papers/4/lfw-deepfunneled/lfw-deepfunneled',
                      help='Path to the face dataset')
    parser.add_argument('--sample_size', type=int, default=10000,
                      help='Number of images to sample for evaluation (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='face/papers/face_landmarks/results/face_detection',
                      help='Directory to save evaluation results')
    parser.add_argument('--visualize', type=int, default=1000,
                      help='Number of detection results to visualize (default: 20)')
    parser.add_argument('--conf_thres', type=float, default=0.5,
                      help='Face detection confidence threshold (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Batch size for processing images (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducible sampling')
    return parser.parse_args()

def load_sample_images(dataset_path, sample_size, seed=42):
    """Load sample images from the dataset randomly"""
    print(f"Loading images from dataset at: {dataset_path}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    all_images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} total images")
    
    # Sample images if requested
    if 0 < sample_size < len(all_images):
        print(f"Sampling {sample_size} images for evaluation")
        return random.sample(all_images, sample_size)
    else:
        print(f"Using all {len(all_images)} images for evaluation")
        return all_images

def load_face_detection_model():
    """Load face detection model using OpenCV"""
    print("Loading face detection model using OpenCV...")
    
    # Use OpenCV's face detection
    try:
        # Try to load the Haar cascade face detector as fallback
        model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_detector = cv2.CascadeClassifier(model_path)
        
        if face_detector.empty():
            raise Exception("Failed to load Haar cascade model")
        
        print("OpenCV Haar cascade face detector model loaded successfully")
        return face_detector
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        sys.exit(1)

def detect_faces(face_detector, img, conf_thres=0.5):
    """Detect faces in an image using OpenCV face detector"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    # Parameters are scale factor, min neighbors, flags, min size
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Process detections
    valid_detections = []
    for (x, y, w, h) in faces:
        # Add to list with confidence of 1.0 (Haar doesn't provide confidence)
        valid_detections.append({
            'bbox': (x, y, w, h),  # x, y, width, height
            'confidence': 1.0  # Haar doesn't provide confidence scores
        })
    
    return valid_detections

def process_image_batch(face_detector, image_paths, conf_thres=0.5):
    """Process a batch of images with the face detector"""
    results = []
    
    for image_path in image_paths:
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            # Start timer for face detection
            start_time = time.time()
            
            # Detect faces
            faces = detect_faces(face_detector, img, conf_thres)
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            # Store results
            results.append({
                'image_path': image_path,
                'detected_faces': len(faces),
                'detection_time': detection_time,
                'faces': faces,
                'success': len(faces) > 0
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def evaluate_face_detector(face_detector, image_paths, args):
    """Evaluate the face detector on the provided image paths"""
    print(f"Evaluating face detector on {len(image_paths)} images...")
    
    # Create directories for results
    os.makedirs(args.output_dir, exist_ok=True)
    visualizations_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Lists to store results
    all_results = []
    detection_times = []
    face_counts = []
    face_sizes = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i+args.batch_size]
        batch_results = process_image_batch(face_detector, batch_paths, args.conf_thres)
        all_results.extend(batch_results)
        
        # Extract metrics
        for result in batch_results:
            detection_times.append(result['detection_time'])
            face_counts.append(result['detected_faces'])
            
            # Extract face sizes
            for face in result['faces']:
                x, y, w, h = face['bbox']
                face_sizes.append(w * h)
    
    # Calculate overall metrics
    total_images = len(all_results)
    images_with_faces = sum(1 for result in all_results if result['detected_faces'] > 0)
    total_faces_detected = sum(result['detected_faces'] for result in all_results)
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    
    # Calculate face count statistics
    one_face_images = sum(1 for count in face_counts if count == 1)
    multi_face_images = sum(1 for count in face_counts if count > 1)
    zero_face_images = sum(1 for count in face_counts if count == 0)
    
    # Calculate percentiles for detection time
    if detection_times:
        percentiles = np.percentile(detection_times, [50, 90, 95, 99])
    else:
        percentiles = [0, 0, 0, 0]
    
    metrics = {
        'model_name': 'TFLite Face Detection',
        'total_images': total_images,
        'images_with_faces': images_with_faces,
        'images_with_no_faces': total_images - images_with_faces,
        'face_detection_rate': images_with_faces / total_images if total_images > 0 else 0,
        'total_faces_detected': total_faces_detected,
        'avg_faces_per_image': total_faces_detected / total_images if total_images > 0 else 0,
        'one_face_images': one_face_images,
        'multi_face_images': multi_face_images,
        'zero_face_images': zero_face_images,
        'avg_detection_time': avg_detection_time,
        'median_detection_time': percentiles[0],
        'p90_detection_time': percentiles[1],
        'p95_detection_time': percentiles[2],
        'p99_detection_time': percentiles[3],
        'throughput_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
        'avg_face_size': np.mean(face_sizes) if face_sizes else 0,
        'min_face_size': min(face_sizes) if face_sizes else 0,
        'max_face_size': max(face_sizes) if face_sizes else 0,
        'true_accuracy': one_face_images / total_images if total_images > 0 else 0,
        'precision': one_face_images / total_faces_detected if total_faces_detected > 0 else 0,
        'false_positive_rate': (total_faces_detected - one_face_images) / total_images if total_images > 0 else 0,
        'expected_faces': total_images,  # Assuming LFW has 1 face per image
    }
    
    # Save metrics and results
    save_evaluation_results(all_results, metrics, args.output_dir)
    
    # Generate visualizations for a subset of results
    if args.visualize > 0:
        visualize_subset = select_visualization_samples(all_results, min(args.visualize, len(all_results)))
        create_visualizations(visualize_subset, visualizations_dir)
    
    return metrics

def select_visualization_samples(results, visualize_count):
    """Select a diverse set of samples for visualization"""
    if visualize_count >= len(results):
        return results
    
    # Sort results by number of detected faces
    results_by_faces = {}
    for result in results:
        face_count = result['detected_faces']
        if face_count not in results_by_faces:
            results_by_faces[face_count] = []
        results_by_faces[face_count].append(result)
    
    # Select samples from each face count category
    samples = []
    face_counts = sorted(results_by_faces.keys())
    
    # Allocate visualization slots proportionally
    for face_count in face_counts:
        category_results = results_by_faces[face_count]
        category_proportion = len(category_results) / len(results)
        category_samples = max(1, int(visualize_count * category_proportion))
        
        # Select random samples from this category
        if category_samples < len(category_results):
            selected = random.sample(category_results, category_samples)
        else:
            selected = category_results
        
        samples.extend(selected)
    
    # If we have too many samples, trim the list
    if len(samples) > visualize_count:
        samples = random.sample(samples, visualize_count)
    
    return samples

def create_visualizations(results, output_dir):
    """Create visualizations of detection results"""
    print(f"Generating {len(results)} visualizations...")
    
    for i, result in enumerate(results):
        try:
            # Load image
            img = cv2.imread(result['image_path'])
            if img is None:
                continue
            
            # Create visualization image
            vis_img = visualize_face_detection(img, result)
            
            # Save visualization
            image_name = os.path.basename(result['image_path'])
            output_name = f"{i+1:03d}_detection_{image_name}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, vis_img)
            
        except Exception as e:
            print(f"Error creating visualization for {result['image_path']}: {e}")
    
    print(f"Visualizations saved to {output_dir}")

def visualize_face_detection(img, result):
    """Create a visualization of face detection on an image"""
    # Make a copy for visualization
    vis_img = img.copy()
    
    # Add detection information text
    cv2.putText(vis_img, f"Faces: {result['detected_faces']}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Time: {result['detection_time']:.4f}s", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw face rectangles
    for face in result['faces']:
        x, y, w, h = face['bbox']
        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw confidence score
        conf_text = f"{face['confidence']:.2f}"
        cv2.putText(vis_img, conf_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_img

def save_evaluation_results(results, metrics, output_dir):
    """Save detailed evaluation results and metrics to files"""
    # Save summary metrics to CSV (same format as mmod/haar face detection)
    summary_csv_path = os.path.join(output_dir, 'metrics.csv')
    summary_fields = [
        ('avg_detection_time', metrics.get('avg_detection_time', 0)),
        ('faces_per_second', metrics.get('throughput_fps', 0)),
        ('p50_detection_time', metrics.get('median_detection_time', 0)),
        ('p90_detection_time', metrics.get('p90_detection_time', 0)),
        ('p95_detection_time', metrics.get('p95_detection_time', 0)),
        ('p99_detection_time', metrics.get('p99_detection_time', 0)),
        ('total_images', metrics.get('total_images', 0)),
        ('images_with_faces', metrics.get('images_with_faces', 0)),
        ('images_with_no_faces', metrics.get('images_with_no_faces', 0)),
        ('total_faces_detected', metrics.get('total_faces_detected', 0)),
        ('expected_faces', metrics.get('expected_faces', 0)),
        ('detection_rate', metrics.get('face_detection_rate', 0)),
        ('true_accuracy', metrics.get('true_accuracy', 0)),
        ('precision', metrics.get('precision', 0)),
        ('false_positive_rate', metrics.get('false_positive_rate', 0)),
        ('avg_faces_per_image', metrics.get('avg_faces_per_image', 0)),
        ('one_face_images', metrics.get('one_face_images', 0)),
        ('multi_face_images', metrics.get('multi_face_images', 0)),
        ('zero_face_images', metrics.get('zero_face_images', 0)),
        ('avg_size', metrics.get('avg_face_size', 0)),
        ('min_size', metrics.get('min_face_size', 0)),
        ('max_size', metrics.get('max_face_size', 0)),
    ]
    
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in summary_fields:
            if v is not None:  # Only write non-None values
                writer.writerow([k, v])
                
    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'detection_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Image', 'Faces Detected', 'Detection Time (s)', 'Success'
        ])
        
        for result in results:
            image_name = os.path.basename(result['image_path'])
            writer.writerow([
                image_name, result['detected_faces'], result['detection_time'], result['success']
            ])
    
    # Generate performance plots
    generate_performance_plots(results, metrics, output_dir)
    
    print(f"Results saved to {output_dir}")

def generate_performance_plots(results, metrics, output_dir):
    """Generate performance visualization plots"""
    # Extract relevant data
    detection_times = [r['detection_time'] for r in results]
    face_counts = [r['detected_faces'] for r in results]
    success_rates = [1 if r['success'] else 0 for r in results]
    
    # 1. Detection Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(detection_times, bins=50, alpha=0.7)
    plt.axvline(metrics['avg_detection_time'], color='red', linestyle='dashed', 
               linewidth=2, label=f'Avg: {metrics["avg_detection_time"]:.4f}s')
    plt.title('Face Detection Time Distribution')
    plt.xlabel('Detection Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_time_distribution.png'))
    plt.close()
    
    # 2. Number of Faces Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(face_counts, bins=range(max(face_counts) + 2) if face_counts else range(1), alpha=0.7)
    plt.title('Number of Faces Detected Per Image')
    plt.xlabel('Number of Faces')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(max(face_counts) + 1) if face_counts else range(1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'face_count_distribution.png'))
    plt.close()
    
    # 3. Detection Time vs Number of Faces
    plt.figure(figsize=(10, 6))
    plt.scatter(face_counts, detection_times, alpha=0.5)
    plt.title('Detection Time vs. Number of Faces')
    plt.xlabel('Number of Faces')
    plt.ylabel('Detection Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_time_vs_faces.png'))
    plt.close()
    
    # 4. Success Rate by Face Count
    max_faces = max(face_counts) if face_counts else 0
    success_by_faces = {}
    
    for count, success in zip(face_counts, success_rates):
        if count not in success_by_faces:
            success_by_faces[count] = []
        success_by_faces[count].append(success)
    
    face_counts_unique = sorted(success_by_faces.keys())
    success_rates_avg = [np.mean(success_by_faces[count]) for count in face_counts_unique]
    
    plt.figure(figsize=(10, 6))
    plt.bar(face_counts_unique, success_rates_avg)
    plt.title('Detection Success Rate by Face Count')
    plt.xlabel('Number of Faces')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_by_face_count.png'))
    plt.close()
    
    # 5. Create a performance metrics summary plot
    plt.figure(figsize=(12, 8))
    metrics_to_plot = [
        ('face_detection_rate', 'Face Detection Rate'),
        ('true_accuracy', 'True Accuracy'),
        ('avg_detection_time', 'Avg Detection Time (s)'),
        ('throughput_fps', 'Throughput (FPS)'),
    ]
    
    values = [metrics[k] for k, _ in metrics_to_plot]
    labels = [v for _, v in metrics_to_plot]
    
    # Normalize for display
    max_val = max(values) if values else 1
    norm_values = [v / max_val for v in values]
    
    plt.bar(range(len(norm_values)), norm_values, tick_label=labels)
    plt.title('Performance Metrics Summary (Normalized)')
    plt.ylabel('Normalized Value')
    plt.xticks(rotation=15)
    
    # Add actual values as text
    for i, v in enumerate(values):
        if v < 0.01:
            value_text = f"{v:.4f}"
        elif v < 1:
            value_text = f"{v:.2f}"
        else:
            value_text = f"{v:.1f}"
        plt.text(i, norm_values[i] + 0.05, value_text, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'))
    plt.close()

def print_summary_metrics(metrics):
    """Print a summary of the evaluation metrics"""
    print("\n" + "=" * 80)
    print("TFLITE FACE DETECTION EVALUATION RESULTS")
    print("=" * 80)
    
    print("\nDetection Performance:")
    print(f"  • Detection Rate: {metrics['face_detection_rate']:.2%}")
    print(f"  • True Accuracy (exactly one face): {metrics['true_accuracy']:.2%}")
    print(f"  • False Positive Rate: {metrics['false_positive_rate']:.2%}")
    
    print("\nDetection Counts:")
    print(f"  • Total Images: {metrics['total_images']}")
    print(f"  • Images with Faces: {metrics['images_with_faces']} ({metrics['face_detection_rate']:.2%})")
    print(f"  • Images with No Faces: {metrics['images_with_no_faces']} "
          f"({metrics['images_with_no_faces']/metrics['total_images']:.2%})")
    print(f"  • Images with One Face: {metrics['one_face_images']} "
          f"({metrics['one_face_images']/metrics['total_images']:.2%})")
    print(f"  • Images with Multiple Faces: {metrics['multi_face_images']} "
          f"({metrics['multi_face_images']/metrics['total_images']:.2%})")
    
    print("\nTiming Performance:")
    print(f"  • Average Processing Time: {metrics['avg_detection_time']:.4f} seconds per image")
    print(f"  • Processing Speed: {metrics['throughput_fps']:.2f} faces per second")
    print(f"  • P50 (Median) Time: {metrics['median_detection_time']:.4f} seconds")
    print(f"  • P90 Time: {metrics['p90_detection_time']:.4f} seconds")
    print(f"  • P95 Time: {metrics['p95_detection_time']:.4f} seconds")
    print(f"  • P99 Time: {metrics['p99_detection_time']:.4f} seconds")

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load face detection model
        face_detector = load_face_detection_model()
        
        # Load sample images
        image_paths = load_sample_images(args.dataset, args.sample_size, args.seed)
        if not image_paths:
            print("No images found. Please check the dataset path.")
            return
        
        # Evaluate face detector
        metrics = evaluate_face_detector(face_detector, image_paths, args)
        
        # Print summary
        print_summary_metrics(metrics)
        
        print(f"\nEvaluation complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
