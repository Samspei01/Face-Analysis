#!/usr/bin/env python3
"""
Facial Landmarks Detection Evaluation

This script evaluates the dlib 68-point facial landmarks detector on a dataset of your choice.
It measures performance metrics, creates visualizations, and generates an HTML report.

Usage:
    python dlib.py --dataset <path_to_dataset> --sample_size <number_of_images>
"""

import os
import sys
import argparse
import time
import random
import csv
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate facial landmarks detection')
    parser.add_argument('--dataset', type=str,
                      default='face/papers/4/lfw-deepfunneled/lfw-deepfunneled',
                      help='Path to the face dataset')
    parser.add_argument('--sample_size', type=int, default=10000,
                      help='Number of images to sample for evaluation (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='face/papers/face_landmarks/results',
                      help='Directory to save evaluation results')
    parser.add_argument('--visualize', type=int, default=1000,
                      help='Number of detection results to visualize (default: 20)')
    parser.add_argument('--landmarks_model', type=str, 
                      default='face/papers/landmarks_models/shape_predictor_68_face_landmarks.dat',
                      help='Path to the dlib landmarks model')
    parser.add_argument('--conf_thres', type=float, default=0.0,
                      help='Face detection confidence threshold (default: 0.0)')
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
    if len(all_images) > sample_size:
        random.shuffle(all_images)
        sample_images = all_images[:sample_size]
        print(f"Sampled {sample_size} images for evaluation")
    else:
        sample_images = all_images
        print(f"Using all {len(sample_images)} available images for evaluation")
    
    return sample_images

def load_models(landmarks_model_path):
    """Load face detector and landmarks predictor"""
    print(f"Loading face detector and landmark predictor...")
    
    # Load face detector (HOG-based)
    face_detector = dlib.get_frontal_face_detector()
    
    # Check if landmarks model exists
    if not os.path.exists(landmarks_model_path):
        raise FileNotFoundError(f"Landmarks model not found at {landmarks_model_path}")
    
    # Load landmarks predictor
    landmarks_predictor = dlib.shape_predictor(landmarks_model_path)
    
    print("Models loaded successfully")
    return face_detector, landmarks_predictor

def shape_to_np(shape, dtype="int"):
    """Convert dlib shape to numpy array"""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def process_image_batch(face_detector, landmarks_predictor, image_paths, conf_thres=0.0):
    """Process a batch of images with the face detector and landmarks predictor"""
    results = []
    
    for image_path in image_paths:
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Start timer for face detection
            start_time = time.time()
            
            # Detect faces
            faces = face_detector(gray)
            
            # If no faces were detected, continue to next image
            if len(faces) == 0:
                end_time = time.time()
                detection_time = end_time - start_time
                
                results.append({
                    'image_path': image_path,
                    'detected_faces': 0,
                    'landmarks_detected': 0,
                    'detection_time': detection_time,
                    'faces': [],
                    'success': False
                })
                continue
            
            # Process landmarks for each detected face
            processed_faces = []
            for face in faces:
                # Ensure confidence is above threshold
                if conf_thres > 0 and face.confidence < conf_thres:
                    continue
                
                # Predict landmarks
                shape = landmarks_predictor(gray, face)
                landmarks = shape_to_np(shape)
                
                # Calculate face size metrics
                face_width = face.right() - face.left()
                face_height = face.bottom() - face.top()
                
                # Add to processed faces
                processed_faces.append({
                    'bbox': (face.left(), face.top(), face.right(), face.bottom()),
                    'landmarks': landmarks.tolist(),
                    'face_width': face_width,
                    'face_height': face_height
                })
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            results.append({
                'image_path': image_path,
                'detected_faces': len(faces),
                'landmarks_detected': len(processed_faces),
                'detection_time': detection_time,
                'faces': processed_faces,
                'success': len(processed_faces) > 0
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def evaluate_landmarks_detector(face_detector, landmarks_predictor, image_paths, args):
    """Evaluate the landmarks detector on the provided image paths"""
    print(f"Evaluating landmarks detector on {len(image_paths)} images...")
    
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
        batch_results = process_image_batch(face_detector, landmarks_predictor, batch_paths, args.conf_thres)
        all_results.extend(batch_results)
        
        # Extract metrics
        for result in batch_results:
            detection_times.append(result['detection_time'])
            face_counts.append(result['detected_faces'])
            
            # Extract face sizes
            for face in result['faces']:
                face_width = face['face_width']
                face_height = face['face_height']
                face_sizes.append((face_width + face_height) / 2)  # Average face size
    
    # Calculate overall metrics
    total_images = len(all_results)
    images_with_faces = sum(1 for result in all_results if result['detected_faces'] > 0)
    total_faces_detected = sum(result['detected_faces'] for result in all_results)
    landmarks_detected = sum(result['landmarks_detected'] for result in all_results)
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    
    # Calculate percentiles for detection time
    if detection_times:
        percentiles = np.percentile(detection_times, [50, 90, 95, 99])
    else:
        percentiles = [0, 0, 0, 0]
    
    metrics = {
        'total_images': total_images,
        'images_with_faces': images_with_faces,
        'face_detection_rate': images_with_faces / total_images if total_images > 0 else 0,
        'total_faces_detected': total_faces_detected,
        'landmarks_detected': landmarks_detected,
        'landmarks_success_rate': landmarks_detected / total_faces_detected if total_faces_detected > 0 else 0,
        'avg_faces_per_image': total_faces_detected / total_images if total_images > 0 else 0,
        'avg_detection_time': avg_detection_time,
        'median_detection_time': percentiles[0],
        'p90_detection_time': percentiles[1],
        'p95_detection_time': percentiles[2],
        'p99_detection_time': percentiles[3],
        'throughput_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
        'avg_face_size': np.mean(face_sizes) if face_sizes else 0
    }
    
    # Save metrics and results
    save_evaluation_results(all_results, metrics, args.output_dir)
    
    # Generate visualizations for a subset of results
    if args.visualize > 0:
        visualize_subset = select_visualization_samples(all_results, args.visualize)
        create_visualizations(visualize_subset, visualizations_dir)
    
    return metrics

def select_visualization_samples(results, visualize_count):
    """Select a diverse set of samples for visualization"""
    # Split into successful and failed detections
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    # Determine how many of each to sample
    if len(successful) < visualize_count // 2:
        successful_count = len(successful)
        failed_count = min(len(failed), visualize_count - successful_count)
    elif len(failed) < visualize_count // 2:
        failed_count = len(failed)
        successful_count = min(len(successful), visualize_count - failed_count)
    else:
        successful_count = visualize_count // 2
        failed_count = visualize_count - successful_count
    
    # Sample from each category
    sampled_successful = random.sample(successful, successful_count) if successful else []
    sampled_failed = random.sample(failed, failed_count) if failed else []
    
    # Combine and shuffle
    samples = sampled_successful + sampled_failed
    random.shuffle(samples)
    
    return samples

def create_visualizations(results, output_dir):
    """Create visualizations of landmarks detection results"""
    print(f"Generating {len(results)} visualizations...")
    
    for i, result in enumerate(results):
        try:
            # Load image
            img = cv2.imread(result['image_path'])
            if img is None:
                continue
            
            # Create visualization image
            vis_img = visualize_face_landmarks(img, result)
            
            # Save visualization
            image_name = os.path.basename(result['image_path'])
            output_name = f"{i+1:03d}_landmarks_{image_name}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, vis_img)
            
        except Exception as e:
            print(f"Error creating visualization for {result['image_path']}: {e}")
    
    print(f"Visualizations saved to {output_dir}")

def visualize_face_landmarks(img, result):
    """Create a visualization of face landmarks on an image"""
    # Make a copy for visualization
    vis_img = img.copy()
    
    # Define colors for landmark groups (BGR format)
    colors = {
        'jaw': (0, 255, 0),        # Green
        'right_eyebrow': (0, 0, 255),  # Red
        'left_eyebrow': (0, 0, 255),   # Red
        'nose_bridge': (255, 0, 0),    # Blue
        'nose_tip': (255, 0, 0),       # Blue
        'right_eye': (0, 255, 255),    # Yellow
        'left_eye': (0, 255, 255),     # Yellow
        'outer_lip': (255, 255, 0),    # Cyan
        'inner_lip': (255, 0, 255)     # Magenta
    }
    
    # Define landmark groups
    landmark_groups = [
        ('jaw', range(0, 17), False),
        ('right_eyebrow', range(17, 22), False),
        ('left_eyebrow', range(22, 27), False),
        ('nose_bridge', range(27, 31), False),
        ('nose_tip', range(31, 36), True),
        ('right_eye', range(36, 42), True),
        ('left_eye', range(42, 48), True),
        ('outer_lip', range(48, 60), True),
        ('inner_lip', range(60, 68), True)
    ]
    
    # Add header with information
    cv2.putText(vis_img, f"Faces: {result['detected_faces']}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Time: {result['detection_time']:.4f}s", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw face rectangles and landmarks
    for face in result['faces']:
        # Draw face rectangle
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert landmarks list back to numpy array
        landmarks = np.array(face['landmarks'])
        
        # Draw each landmark group
        for group_name, indices, closed in landmark_groups:
            points = landmarks[list(indices)]
            color = colors[group_name]
            
            # Draw points
            for x, y in points:
                cv2.circle(vis_img, (x, y), 2, color, -1)
            
            # Connect points with lines
            for j in range(len(points) - 1):
                p1 = tuple(points[j])
                p2 = tuple(points[j+1])
                cv2.line(vis_img, p1, p2, color, 1)
            
            # Close the shape if required
            if closed and len(points) > 1:
                cv2.line(vis_img, tuple(points[-1]), tuple(points[0]), color, 1)
    
    return vis_img

def save_evaluation_results(results, metrics, output_dir):
    """Save detailed evaluation results and metrics to files"""
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'landmarks_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save summary metrics to CSV (mmod/haar style)
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
        ('images_with_no_faces', metrics.get('total_images', 0) - metrics.get('images_with_faces', 0)),
        ('total_faces_detected', metrics.get('total_faces_detected', 0)),
        ('expected_faces', metrics.get('total_images', 0)),
        ('detection_rate', metrics.get('face_detection_rate', 0)),
        ('true_accuracy', metrics.get('face_detection_rate', 0)),
        ('precision', metrics.get('landmarks_success_rate', 0)),
        ('false_positive_rate', 0),  # Not directly available for landmarks
        ('avg_faces_per_image', metrics.get('avg_faces_per_image', 0)),
        ('one_face_images', None),  # Not tracked, can be added if needed
        ('multi_face_images', None),  # Not tracked, can be added if needed
        ('zero_face_images', None),  # Not tracked, can be added if needed
        ('avg_size', metrics.get('avg_face_size', 0)),
        ('min_size', None),  # Not tracked, can be added if needed
        ('max_size', None),  # Not tracked, can be added if needed
        ('total_processing_time', None),  # Not tracked, can be added if needed
    ]
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in summary_fields:
            if v is not None:
                writer.writerow([k, v])

    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'landmarks_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Image', 'Faces Detected', 'Landmarks Detected', 
            'Detection Time (s)', 'Success'
        ])
        for result in results:
            image_name = os.path.basename(result['image_path'])
            writer.writerow([
                image_name, result['detected_faces'], result['landmarks_detected'], 
                result['detection_time'], result['success']
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
    plt.title('Landmarks Detection Time Distribution')
    plt.xlabel('Detection Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_time_distribution.png'))
    plt.close()
    
    # 2. Number of Faces Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(face_counts, bins=range(max(face_counts) + 2), alpha=0.7)
    plt.title('Number of Faces Detected Per Image')
    plt.xlabel('Number of Faces')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(max(face_counts) + 1))
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
    max_faces = max(face_counts)
    success_by_faces = {}
    
    for count, success in zip(face_counts, success_rates):
        if count not in success_by_faces:
            success_by_faces[count] = []
        success_by_faces[count].append(success)
    
    face_counts_unique = sorted(success_by_faces.keys())
    success_rates_avg = [np.mean(success_by_faces[count]) for count in face_counts_unique]
    
    plt.figure(figsize=(10, 6))
    plt.bar(face_counts_unique, success_rates_avg)
    plt.title('Landmarks Detection Success Rate by Face Count')
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
        ('landmarks_success_rate', 'Landmarks Success Rate'),
        ('avg_detection_time', 'Avg Detection Time (s)'),
        ('throughput_fps', 'Throughput (FPS)'),
    ]
    
    values = [metrics[k] for k, _ in metrics_to_plot]
    labels = [v for _, v in metrics_to_plot]
    
    # Normalize for display
    max_val = max(values)
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

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load models
        face_detector, landmarks_predictor = load_models(args.landmarks_model)
        
        # Load sample images
        image_paths = load_sample_images(args.dataset, args.sample_size, args.seed)
        if not image_paths:
            print("No images found. Please check the dataset path.")
            return
        
        # Evaluate landmarks detector
        metrics = evaluate_landmarks_detector(face_detector, landmarks_predictor, image_paths, args)
        
        print(f"Evaluation complete! Results saved to {args.output_dir}")
        print("\nSummary Metrics:")
        print(f"Total Images: {metrics['total_images']}")
        print(f"Face Detection Rate: {metrics['face_detection_rate']:.2%}")
        print(f"Landmarks Success Rate: {metrics['landmarks_success_rate']:.2%}")
        print(f"Average Detection Time: {metrics['avg_detection_time']:.4f} seconds")
        print(f"Throughput: {metrics['throughput_fps']:.2f} images/second")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
