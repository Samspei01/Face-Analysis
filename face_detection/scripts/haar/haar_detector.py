#!/usr/bin/env python3
"""
Haar Cascade Face Detector Model Evaluation Script with Highest Confidence Filtering

This script evaluates the performance of haarcascade_frontalface_default.xml on the LFW dataset,
using a filtering approach that keeps only the highest confidence face per image.

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test Haar Cascade face detector on LFW dataset')
    parser.add_argument('--model', type=str,
                      default='/home/samsepi0l/Project/FaceRecognition/face/papers/Detect_models/haarcascade_frontalface_default.xml',
                      help='Path to the Haar Cascade face detector model')
    parser.add_argument('--dataset', type=str,
                      default='/home/samsepi0l/Project/FaceRecognition/face/papers/4/lfw-deepfunneled/lfw-deepfunneled',
                      help='Path to the LFW dataset')
    parser.add_argument('--sample_size', type=int, default=1000,
                      help='Number of images to sample for evaluation')
    parser.add_argument('--output_dir', type=str, default='haar_evaluation_results_highest_confidence',
                      help='Directory to save evaluation results')
    parser.add_argument('--visualize', type=int, default=10,
                      help='Number of images to visualize results for')
    parser.add_argument('--full_dataset', action='store_true',
                      help='Process the full dataset instead of sampling')
    parser.add_argument('--image_size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                      help='Specify the size of the image as width and height (currently not used in Haar detector)')
    return parser.parse_args()

def load_model(model_path):
    # Adjust model_path if it's a relative path
    if not os.path.isabs(model_path):
        # Try multiple locations for the model file
        possible_paths = [
            model_path,  # As is (could be relative to working dir)
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', os.path.basename(model_path)),  # From face_detection/models/
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'Detect_models', os.path.basename(model_path))  # From papers/Detect_models/
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    print(f"Loading Haar Cascade model from: {model_path}")
    detector = cv2.CascadeClassifier(model_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar Cascade model from {model_path}")
    return detector

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

def detect_faces(detector, image_paths, filter_highest_confidence=True):
    detections = []
    times = []
    print("Running face detection...")
    for image_path in tqdm(image_paths):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        start_time = time.time()
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        end_time = time.time()
        detection_time = end_time - start_time
        times.append(detection_time)
        
        # Haar does not provide confidence, so we use rectangle size as a proxy
        face_objs = []
        for (x, y, w, h) in faces:
            face_objs.append({'rect': (x, y, w, h), 'size': w * h})
        
        # Filter to keep only the highest confidence face
        if filter_highest_confidence and len(face_objs) > 1:
            largest_face = max(face_objs, key=lambda x: x['size'])
            face_objs = [largest_face]
            
        detections.append((image_path, img, face_objs))
    return detections, times

def evaluate_performance(detections, times):
    avg_time = sum(times) / len(times)
    faces_per_second = 1.0 / avg_time
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2]
    p90 = sorted_times[int(len(sorted_times) * 0.9)]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    total_images = len(detections)
    images_with_faces = sum(1 for _, _, faces in detections if len(faces) > 0)
    images_with_no_faces = total_images - images_with_faces
    total_faces = sum(len(faces) for _, _, faces in detections)
    face_counts = [len(faces) for _, _, faces in detections]
    one_face_images = sum(1 for count in face_counts if count == 1)
    multi_face_images = sum(1 for count in face_counts if count > 1)
    zero_face_images = sum(1 for count in face_counts if count == 0)
    all_sizes = [face['size'] for _, _, faces in detections for face in faces]
    avg_size = np.mean(all_sizes) if all_sizes else 0
    min_size = min(all_sizes) if all_sizes else 0
    max_size = max(all_sizes) if all_sizes else 0
    expected_faces = total_images
    detection_rate = images_with_faces / total_images if total_images > 0 else 0
    
    # Add new accuracy metrics
    true_accuracy = one_face_images / total_images if total_images > 0 else 0
    
    # Calculate precision based on how many of the detected faces are likely correct
    precision = 0
    if total_faces > 0:
        correct_detections = one_face_images + multi_face_images
        precision = correct_detections / total_images
    
    # Calculate false positive rate (extra faces detected)
    extra_faces = total_faces - total_images
    false_positive_rate = extra_faces / total_images if total_images > 0 else 0
    
    return {
        'avg_detection_time': avg_time,
        'faces_per_second': faces_per_second,
        'p50_detection_time': p50,
        'p90_detection_time': p90,
        'p95_detection_time': p95,
        'p99_detection_time': p99,
        'total_images': total_images,
        'images_with_faces': images_with_faces,
        'images_with_no_faces': images_with_no_faces,
        'total_faces_detected': total_faces,
        'expected_faces': expected_faces,
        'detection_rate': detection_rate,
        'true_accuracy': true_accuracy,
        'precision': precision, 
        'false_positive_rate': false_positive_rate,
        'avg_faces_per_image': total_faces / total_images if total_images > 0 else 0,
        'one_face_images': one_face_images,
        'multi_face_images': multi_face_images,
        'zero_face_images': zero_face_images,
        'avg_size': avg_size,
        'min_size': min_size,
        'max_size': max_size
    }

def visualize_results(detections, num_to_visualize, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    successful = [d for d in detections if len(d[2]) == 1]
    issues = [d for d in detections if len(d[2]) != 1]
    num_success = min(num_to_visualize // 2, len(successful))
    num_issues = min(num_to_visualize - num_success, len(issues))
    if num_success < num_to_visualize // 2:
        num_issues = min(num_to_visualize - num_success, len(issues))
    if num_issues < num_to_visualize - num_success:
        num_success = min(num_to_visualize - num_issues, len(successful))
    if successful and num_success > 0:
        random.shuffle(successful)
        successful_sample = successful[:num_success]
    else:
        successful_sample = []
    if issues and num_issues > 0:
        random.shuffle(issues)
        issues_sample = issues[:num_issues]
    else:
        issues_sample = []
    sample = successful_sample + issues_sample
    random.shuffle(sample)
    for i, (image_path, img, faces) in enumerate(sample):
        img_cv = img.copy()
        person_name = os.path.basename(os.path.dirname(image_path))
        image_name = os.path.basename(image_path)
        title_text = f"{person_name} - {image_name}"
        info_text = f"Faces detected: {len(faces)}"
        cv2.rectangle(img_cv, (0, 0), (img_cv.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(img_cv, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_cv, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for face in faces:
            x, y, w, h = face['rect']
            size = face['size']
            color = (0, 255, 0) if size > 5000 else (0, 165, 255) if size > 2500 else (0, 0, 255)
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_cv, f"Size: {size}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if len(faces) == 1:
            status = "CORRECT: One face detected"
            status_color = (0, 255, 0)
        elif len(faces) == 0:
            status = "ERROR: No face detected"
            status_color = (0, 0, 255)
        else:
            status = f"WARNING: {len(faces)} faces detected"
            status_color = (0, 165, 255)
        cv2.rectangle(img_cv, (0, img_cv.shape[0]-30), (img_cv.shape[1], img_cv.shape[0]), (0, 0, 0), -1)
        cv2.putText(img_cv, status, (10, img_cv.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        output_path = os.path.join(output_dir, f"detection_{i+1}.png")
        cv2.imwrite(output_path, img_cv)
        print(f"Saved visualization to {output_path}")

def print_results(metrics):
    print("\n" + "="*70)
    print(" Haar Cascade Face Detector Evaluation Results (Highest Confidence) ".center(70, "="))
    print("="*70)
    print("\nTIMING METRICS:")
    print(f"- Average detection time: {metrics['avg_detection_time']:.4f} seconds per image")
    print(f"- Processing speed: {metrics['faces_per_second']:.2f} faces/second")
    print(f"- P50 (median) detection time: {metrics['p50_detection_time']:.4f} seconds")
    print(f"- P90 detection time: {metrics['p90_detection_time']:.4f} seconds")
    print(f"- P95 detection time: {metrics['p95_detection_time']:.4f} seconds")
    print(f"- P99 detection time: {metrics['p99_detection_time']:.4f} seconds")
    print("\nDETECTION METRICS:")
    print(f"- Images processed: {metrics['total_images']}")
    print(f"- Images with faces detected: {metrics['images_with_faces']} ({metrics['detection_rate']*100:.2f}%)")
    print(f"- Images with no faces detected: {metrics['images_with_no_faces']} ({100-metrics['detection_rate']*100:.2f}%)")
    print(f"- Images with exactly one face: {metrics['one_face_images']} ({metrics['one_face_images']/metrics['total_images']*100:.2f}%)")
    print(f"- Images with multiple faces: {metrics['multi_face_images']} ({metrics['multi_face_images']/metrics['total_images']*100:.2f}%)")
    print(f"- Total faces detected: {metrics['total_faces_detected']}")
    print(f"- Expected faces (assuming 1 per image): {metrics['expected_faces']}")
    print(f"- Average faces per image: {metrics['avg_faces_per_image']:.2f}")
    
    print("\nACCURACY METRICS:")
    print(f"- True accuracy (1 face per image): {metrics['true_accuracy']*100:.2f}%")
    print(f"- Precision (at least 1 face detected): {metrics['precision']*100:.2f}%") 
    print(f"- False positive rate (extra faces): {metrics['false_positive_rate']:.4f}")
    
    print("\nSIZE METRICS (proxy for confidence):")
    print(f"- Average face size: {metrics['avg_size']:.2f} pixels")
    print(f"- Minimum face size: {metrics['min_size']:.2f} pixels")
    print(f"- Maximum face size: {metrics['max_size']:.2f} pixels")
    print("="*70)

def save_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics.csv')
    with open(output_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key},{value}\n")
    print(f"Metrics saved to {output_path}")

def generate_graphs(detections, times, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.switch_backend('Agg')
    plt.figure(figsize=(10, 6))
    plt.hist(times, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Detection Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Face Detection Times')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_time_distribution.png'))
    plt.close()
    face_counts = [len(faces) for _, _, faces in detections]
    unique_counts = sorted(set(face_counts))
    count_freq = [face_counts.count(count) for count in unique_counts]
    plt.figure(figsize=(10, 6))
    plt.bar(unique_counts, count_freq, color='green', edgecolor='black')
    for i, v in enumerate(count_freq):
        plt.text(unique_counts[i], v + 5, str(v), ha='center')
    plt.xlabel('Number of Faces Detected')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Face Counts per Image')
    plt.xticks(unique_counts)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'face_count_distribution.png'))
    plt.close()
    all_sizes = [face['size'] for _, _, faces in detections for face in faces]
    plt.figure(figsize=(10, 6))
    plt.hist(all_sizes, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Face Size (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Detected Face Sizes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_distribution.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.scatter(face_counts, times, alpha=0.6, color='red', edgecolor='black')
    plt.xlabel('Number of Faces Detected')
    plt.ylabel('Detection Time (seconds)')
    plt.title('Detection Time vs. Number of Faces')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_faces.png'))
    plt.close()
    print(f"Analysis graphs saved to {output_dir}")

def main(args=None):
    # If args is None, parse from command line
    if args is None:
        args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    detector = load_model(args.model)
    image_paths = load_sample_images(args.dataset, args.sample_size, getattr(args, 'full_dataset', False))
    start_time = time.time()
    detections, times = detect_faces(detector, image_paths, filter_highest_confidence=True)
    total_time = time.time() - start_time
    metrics = evaluate_performance(detections, times)
    metrics['total_processing_time'] = total_time
    print_results(metrics)
    save_metrics(metrics, args.output_dir)
    try:
        generate_graphs(detections, times, args.output_dir)
    except Exception as e:
        print(f"Warning: Could not generate graphs: {e}")
    if args.visualize > 0:
        visualize_results(detections, args.visualize, args.output_dir)
    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")
    
    return metrics

if __name__ == "__main__":
    main()
