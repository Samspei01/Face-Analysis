#!/usr/bin/env python3
"""
FaceNet Face Recognition Model Evaluation

This script evaluates the performance of FaceNet face recognition model on the LFW dataset.
It follows the standard pairs.txt format for evaluation.
"""

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid display issues
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine
import cv2
import onnxruntime as ort
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet face recognition model on LFW dataset")
    parser.add_argument("--lfw_dir", type=str, 
                        default="./lfw-deepfunneled/lfw-deepfunneled",
                        help="Path to LFW dataset directory")
    parser.add_argument("--pairs_file", type=str, 
                        default="./pairs.csv",
                        help="Path to pairs file (txt or csv format)")
    parser.add_argument("--model_path", type=str, 
                        default="./recognize_models/facenet.onnx",
                        help="Path to the FaceNet ONNX model")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="Number of image pairs to evaluate (default: 500, use -1 for all)")
    parser.add_argument("--output_dir", type=str, 
                        default="./results/facenet",
                        help="Output directory for results")
    parser.add_argument("--face_size", type=int, default=368,
                        help="Face image size required by FaceNet (default: 368)")
    parser.add_argument("--face_detector", type=str, default="dlib",
                        choices=["opencv", "dlib"],
                        help="Face detector to use (default: opencv)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more verbose output")
    return parser.parse_args()

def load_pairs(pairs_file, sample_size=-1):
    """Load pairs from the standard pairs.txt file or CSV format"""
    pairs = []
    
    # Check if file is CSV format based on extension
    if pairs_file.lower().endswith('.csv'):
        import csv
        with open(pairs_file, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)  # Skip header
            except StopIteration:
                print(f"Error: Empty CSV file: {pairs_file}")
                return pairs
                
            for row in reader:
                # Filter out empty strings which can cause parsing errors
                row = [item for item in row if item]
                if len(row) >= 5:
                    # Balanced pairs CSV format: name1, img1_idx, name2, img2_idx, label
                    name1, idx1, name2, idx2, label = row[:5]
                    try:
                        label = int(label)
                        pairs.append((name1, idx1, name2, idx2, label))
                    except ValueError:
                        continue
                elif len(row) >= 3:
                    # Legacy CSV format for same person: name, img1_idx, img2_idx
                    name, idx1, idx2 = row[0], row[1], row[2]
                    pairs.append((name, idx1, name, idx2, 1))  # Same person
                
                if sample_size > 0 and len(pairs) >= sample_size:
                    break
    else:
        # Standard pairs.txt format
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            
            # First line contains the number of pairs
            try:
                num_pairs = int(lines[0].strip())
            except ValueError:
                # Some pairs files don't have this header
                num_pairs = len(lines) - 1
            
            line_idx = 1  # Start from second line
            while line_idx < len(lines) and (sample_size == -1 or len(pairs) < sample_size):
                line = lines[line_idx].strip()
                if not line:
                    line_idx += 1
                    continue
                    
                parts = line.split()
                if len(parts) == 3:
                    # Same person: name idx1 idx2
                    name, idx1, idx2 = parts
                    pairs.append((name, idx1, name, idx2, 1))  # Same person
                elif len(parts) == 4:
                    # Different people: name1 idx1 name2 idx2
                    name1, idx1, name2, idx2 = parts
                    pairs.append((name1, idx1, name2, idx2, 0))  # Different people
                    
                line_idx += 1
            
    print(f"Loaded {len(pairs)} pairs from {pairs_file}")
    return pairs

def construct_image_path(lfw_dir, name, image_idx):
    """Construct path to image based on LFW directory structure"""
    # Standard LFW format: person/person_NNNN.jpg
    image_path = os.path.join(lfw_dir, name, f"{name}_{int(image_idx):04d}.jpg")
    
    # Check if the path exists
    if os.path.exists(image_path):
        return image_path
    
    # Try alternative formats if standard format doesn't exist
    alt_formats = [
        os.path.join(lfw_dir, name, f"{int(image_idx):04d}.jpg"),  # person/NNNN.jpg
        os.path.join(lfw_dir, name, f"{image_idx}.jpg"),  # person/N.jpg
    ]
    
    for alt_path in alt_formats:
        if os.path.exists(alt_path):
            return alt_path
    
    # Return standard format if no alternative found
    return image_path

def detect_and_align_face(image_path, face_size=160, face_detector="opencv"):
    """Detect and align face using the specified detector"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
        
    if face_detector == "opencv":
        # Use OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            # If no face detected, just resize the whole image
            return cv2.resize(img, (face_size, face_size))
            
        # Get the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add some margin (15%)
        margin_x = int(0.15 * w)
        margin_y = int(0.15 * h)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(img.shape[1] - x, w + 2 * margin_x)
        h = min(img.shape[0] - y, h + 2 * margin_y)
        
    elif face_detector == "dlib":
        # Use dlib's face detector
        import dlib
        detector = dlib.get_frontal_face_detector()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_img, 1)
        
        if len(faces) == 0:
            # If no face detected, just resize the whole image
            return cv2.resize(img, (face_size, face_size))
            
        # Get the largest face
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        x, y = largest_face.left(), largest_face.top()
        w, h = largest_face.width(), largest_face.height()
        
        # Add some margin (15%)
        margin_x = int(0.15 * w)
        margin_y = int(0.15 * h)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(img.shape[1] - x, w + 2 * margin_x)
        h = min(img.shape[0] - y, h + 2 * margin_y)
    
    # Extract the face and resize
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (face_size, face_size))
    
    return face

def preprocess_for_facenet(face, face_size=160):
    """Preprocess face for FaceNet model"""
    # Ensure image is RGB
    if face.shape[2] == 4:  # RGBA
        face = cv2.cvtColor(face, cv2.COLOR_RGBA2RGB)
    else:  # BGR
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if face.shape[0] != face_size or face.shape[1] != face_size:
        face = cv2.resize(face, (face_size, face_size))
    
    # Normalize to [-1, 1]
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    
    # Change from HWC to CHW format (for ONNX)
    face = np.transpose(face, (2, 0, 1))
    
    # Add batch dimension
    face = np.expand_dims(face, axis=0)
    
    return face

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print(f"LFW Directory: {args.lfw_dir}")
    print(f"Pairs File: {args.pairs_file}")
    print(f"Model Path: {args.model_path}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Face Size: {args.face_size}")
    print(f"Face Detector: {args.face_detector}")
    
    # Check if directories and files exist
    if not os.path.exists(args.lfw_dir):
        print(f"Error: LFW directory {args.lfw_dir} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.pairs_file):
        print(f"Error: Pairs file {args.pairs_file} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        sys.exit(1)
    
    # Initialize ONNX Runtime session for FaceNet
    print("Loading FaceNet model...")
    try:
        # Use CUDA execution provider if available, otherwise CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(args.model_path, providers=providers)
    except Exception as e:
        print(f"Error loading model with CUDA: {e}")
        print("Falling back to CPU execution provider only...")
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(args.model_path, providers=providers)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    
    # Load pairs
    pairs = load_pairs(args.pairs_file, args.sample_size)
    
    # Process each pair
    print("Processing pairs...")
    similarities = []
    labels = []
    processing_times = []
    failed_pairs = 0
    
    for i, pair in enumerate(tqdm(pairs)):
        # Start timing for this pair
        pair_start_time = time.time()
        
        try:
            name1, idx1, name2, idx2, label = pair
            
            # Construct image paths
            img1_path = construct_image_path(args.lfw_dir, name1, idx1)
            img2_path = construct_image_path(args.lfw_dir, name2, idx2)
            
            if args.debug:
                print(f"Processing pair {i}: {name1}_{idx1} vs {name2}_{idx2}")
                print(f"Image paths: {img1_path}, {img2_path}")
            
            # Detect and align faces
            face1 = detect_and_align_face(img1_path, args.face_size, args.face_detector)
            face2 = detect_and_align_face(img2_path, args.face_size, args.face_detector)
            
            if face1 is None or face2 is None:
                failed_pairs += 1
                continue
            
            # Preprocess faces
            input1 = preprocess_for_facenet(face1, args.face_size)
            input2 = preprocess_for_facenet(face2, args.face_size)
            
            # Get embeddings
            embedding1 = session.run(None, {input_name: input1})[0].flatten()
            embedding2 = session.run(None, {input_name: input2})[0].flatten()
            
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate similarity (1 - cosine distance)
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Save example pair images (first 5 pairs)
            if i < 5:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
                ax[0].set_title(f"{name1}_{idx1}")
                ax[0].axis('off')
                
                ax[1].imshow(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))
                ax[1].set_title(f"{name2}_{idx2}")
                ax[1].axis('off')
                
                plt.suptitle(f"Pair {i}: Similarity = {similarity:.4f}, Same Person = {label==1}")
                plt.savefig(os.path.join(args.output_dir, f"pair_{i}.png"))
                plt.close()
            
            similarities.append(similarity)
            labels.append(label)
            
            # Record processing time for this pair
            pair_end_time = time.time()
            processing_times.append(pair_end_time - pair_start_time)
            
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            failed_pairs += 1
    
    if len(similarities) == 0:
        print("No valid pairs processed. Cannot generate metrics.")
        return
    
    # Generate metrics
    print("Generating metrics...")
    
    # Try different thresholds to find the best accuracy
    thresholds = np.arange(0, 1, 0.01)
    accuracies = []
    
    for threshold in thresholds:
        predictions = [1 if s >= threshold else 0 for s in similarities]
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        accuracies.append(accuracy)
    
    # Find best threshold
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_threshold_idx]
    best_accuracy = accuracies[best_threshold_idx]
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Calculate timing statistics
    if processing_times:
        avg_processing_time = np.mean(processing_times)
        min_processing_time = np.min(processing_times)
        max_processing_time = np.max(processing_times)
        total_processing_time = np.sum(processing_times)
    else:
        avg_processing_time = min_processing_time = max_processing_time = total_processing_time = 0
    
    # Print results
    print("Evaluation complete!")
    print(f"Verification accuracy: {best_accuracy:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Failed pairs: {failed_pairs} out of {len(pairs)}")
    print(f"Average processing time per pair: {avg_processing_time:.4f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    
    # Save results
    results = {
        "model": "FaceNet",
        "accuracy": float(best_accuracy),
        "threshold": float(best_threshold),
        "roc_auc": float(roc_auc),
        "failed_pairs": failed_pairs,
        "total_pairs": len(pairs),
        "success_rate": float((len(pairs) - failed_pairs) / len(pairs)),
        "timing": {
            "avg_processing_time": float(avg_processing_time),
            "min_processing_time": float(min_processing_time),
            "max_processing_time": float(max_processing_time),
            "total_processing_time": float(total_processing_time)
        }
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('FaceNet - Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"))
    plt.close()
    
    # Plot similarity distribution
    plt.figure(figsize=(10, 8))
    same_similarities = [s for s, l in zip(similarities, labels) if l == 1]
    diff_similarities = [s for s, l in zip(similarities, labels) if l == 0]
    
    plt.hist(same_similarities, bins=30, alpha=0.5, label='Same Person', color='green')
    plt.hist(diff_similarities, bins=30, alpha=0.5, label='Different Person', color='red')
    plt.axvline(x=best_threshold, color='black', linestyle='--', 
                label=f'Threshold ({best_threshold:.2f})')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('FaceNet - Distribution of Similarity Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "similarity_distribution.png"))
    plt.close()
    
    # Plot processing time distribution and over time
    if processing_times:
        # Processing time distribution
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Processing time distribution
        plt.subplot(1, 2, 1)
        plt.hist(processing_times, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=avg_processing_time, color='red', linestyle='--', 
                    label=f'Average ({avg_processing_time:.3f}s)')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('FaceNet - Processing Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Processing time over pairs
        plt.subplot(1, 2, 2)
        pair_indices = range(len(processing_times))
        plt.plot(pair_indices, processing_times, alpha=0.7, color='blue', linewidth=1)
        plt.axhline(y=avg_processing_time, color='red', linestyle='--', 
                    label=f'Average ({avg_processing_time:.3f}s)')
        plt.xlabel('Pair Index')
        plt.ylabel('Processing Time (seconds)')
        plt.title('FaceNet - Processing Time Over Pairs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "processing_time_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a detailed timing plot
        plt.figure(figsize=(15, 8))
        
        # Main plot: Processing time with moving average
        window_size = min(50, len(processing_times) // 10) if len(processing_times) > 10 else 1
        if window_size > 1 and len(processing_times) >= window_size:
            moving_avg = np.convolve(processing_times, np.ones(window_size)/window_size, mode='valid')
            moving_avg_indices = range(window_size//2, len(processing_times) - window_size//2 + 1)
            # Ensure indices and moving_avg have the same length
            if len(moving_avg_indices) != len(moving_avg):
                moving_avg_indices = range(len(moving_avg))
        else:
            moving_avg = None
            moving_avg_indices = None
        
        plt.scatter(pair_indices, processing_times, alpha=0.6, s=10, color='lightblue', label='Individual Times')
        if moving_avg is not None and moving_avg_indices is not None:
            plt.plot(moving_avg_indices, moving_avg, color='darkblue', linewidth=2, 
                    label=f'Moving Average (window={window_size})')
        plt.axhline(y=avg_processing_time, color='red', linestyle='--', linewidth=2,
                    label=f'Overall Average ({avg_processing_time:.3f}s)')
        
        plt.xlabel('Pair Index')
        plt.ylabel('Processing Time (seconds)')
        plt.title('FaceNet - Detailed Processing Time Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Statistics:
        Min: {min_processing_time:.3f}s
        Max: {max_processing_time:.3f}s
        Mean: {avg_processing_time:.3f}s
        Std: {np.std(processing_times):.3f}s
        Total: {total_processing_time:.1f}s"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "detailed_timing_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()