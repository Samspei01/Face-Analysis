#!/usr/bin/env python3
"""
Dlib Face Recognition Model Evaluation (Simplified)

This script evaluates the performance of Dlib's face recognition model on the LFW dataset.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid display issues
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine
import cv2
import dlib
import json
import csv
import traceback

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Dlib face recognition model on LFW dataset")
    parser.add_argument("--lfw_dir", type=str, 
                        default="./lfw-deepfunneled/lfw-deepfunneled",
                        help="Path to LFW dataset directory")
    parser.add_argument("--pairs_file", type=str, 
                        default="./pairs.csv",
                        help="Path to pairs file (txt or csv format)")
    parser.add_argument("--model_path", type=str, 
                        default="./recognize_models/dlib_face_recognition_resnet_model_v1.dat",
                        help="Path to the Dlib face recognition model")
    parser.add_argument("--predictor_path", type=str,
                        default="./recognize_models/shape_predictor_68_face_landmarks.dat",
                        help="Path to the facial landmark predictor")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of image pairs to evaluate (default: 100, use -1 for all)")
    parser.add_argument("--output_dir", type=str, 
                        default="./results/dlib",
                        help="Output directory for results")
    parser.add_argument("--face_size", type=int, default=150,
                        help="Face chip size for Dlib (default: 150)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more verbose output")
    return parser.parse_args()

def load_pairs(pairs_file, sample_size=-1, debug=False):
    """Load pairs from the pairs file (CSV or txt)"""
    pairs = []
    
    if not os.path.exists(pairs_file):
        print(f"Error: Pairs file {pairs_file} does not exist")
        return pairs
        
    # Check if file is CSV format based on extension
    if pairs_file.lower().endswith('.csv'):
        try:
            with open(pairs_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                if debug:
                    print(f"CSV header: {header}")
                
                for row in reader:
                    # Filter out empty strings
                    row = [item.strip() for item in row if item.strip()]
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
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return pairs
    else:
        # Standard pairs.txt format
        try:
            with open(pairs_file, 'r') as f:
                lines = f.readlines()
                
                # Check if first line contains only a number (number of pairs)
                first_line = lines[0].strip()
                try:
                    num_pairs = int(first_line)
                    start_idx = 1  # Start from second line
                except ValueError:
                    # First line is not just a number, process all lines
                    start_idx = 0
                
                for i in range(start_idx, len(lines)):
                    if sample_size > 0 and len(pairs) >= sample_size:
                        break
                        
                    line = lines[i].strip()
                    if not line:
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
        except Exception as e:
            print(f"Error reading text file: {e}")
            return pairs
    
    print(f"Loaded {len(pairs)} pairs from {pairs_file}")
    return pairs

def construct_image_path(lfw_dir, name, image_idx, debug=False):
    """Construct path to image based on LFW directory structure"""
    
    # Try different formats to handle variations in dataset organization
    formats = [
        # Standard LFW format: person/person_NNNN.jpg
        os.path.join(lfw_dir, name, f"{name}_{int(image_idx):04d}.jpg"),
        # Alternative: person/NNNN.jpg
        os.path.join(lfw_dir, name, f"{int(image_idx):04d}.jpg"),
        # Alternative: person/N.jpg
        os.path.join(lfw_dir, name, f"{image_idx}.jpg"),
    ]
    
    for path in formats:
        if os.path.exists(path):
            if debug:
                print(f"Found image at: {path}")
            return path
            
    if debug:
        print(f"Warning: Could not find image for {name} with index {image_idx}")
        print(f"Tried paths: {formats}")
        
        # Check if person directory exists
        person_dir = os.path.join(lfw_dir, name)
        if os.path.exists(person_dir):
            print(f"Person directory exists: {person_dir}")
            print(f"Directory contents: {os.listdir(person_dir)[:10]}")
        else:
            print(f"Person directory does not exist: {person_dir}")
    
    # Return the standard format as fallback
    return formats[0]

def detect_and_align_face(image_path, detector, shape_predictor, face_size=150, debug=False):
    """Detect and align face using dlib's detector and shape predictor"""
    
    if debug:
        print(f"Processing image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        if debug:
            print(f"File not found: {image_path}")
        return None
    
    try:
        # Load image
        img = dlib.load_rgb_image(image_path)
        
        if debug:
            print(f"Loaded image, shape: {img.shape}")
        
        # Detect face
        faces = detector(img, 1)
        
        if debug:
            print(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            # Use whole image as face for LFW if detection fails
            if debug:
                print("No face detected, using whole image")
            resized_img = cv2.resize(img, (face_size, face_size))
            return resized_img
        
        # Get largest face
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get landmarks and align face
        shape = shape_predictor(img, largest_face)
        face_chip = dlib.get_face_chip(img, shape, size=face_size)
        
        if debug:
            print(f"Face aligned, shape: {face_chip.shape}")
        
        return face_chip
        
    except Exception as e:
        if debug:
            print(f"Error processing image: {e}")
        return None

def main():
    args = parse_arguments()
    
    # Print configuration
    print(f"LFW Directory: {args.lfw_dir}")
    print(f"Pairs File: {args.pairs_file}")
    print(f"Model Path: {args.model_path}")
    print(f"Predictor Path: {args.predictor_path}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Face Size: {args.face_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    if not os.path.exists(args.predictor_path):
        print(f"Error: Predictor file {args.predictor_path} does not exist")
        return
    
    # Write configuration to log file
    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        f.write(f"LFW Directory: {args.lfw_dir}\n")
        f.write(f"Pairs File: {args.pairs_file}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Predictor Path: {args.predictor_path}\n")
        f.write(f"Sample Size: {args.sample_size}\n")
        f.write(f"Output Directory: {args.output_dir}\n")
        f.write(f"Face Size: {args.face_size}\n")
        f.write(f"Debug Mode: {args.debug}\n")
    
    # Initialize models
    print("Loading models...")
    try:
        detector = dlib.get_frontal_face_detector()
        print("Face detector loaded")
        
        shape_predictor = dlib.shape_predictor(args.predictor_path)
        print("Shape predictor loaded")
        
        face_recognition_model = dlib.face_recognition_model_v1(args.model_path)
        print("Face recognition model loaded")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Load pairs
    print("Loading pairs...")
    pairs = load_pairs(args.pairs_file, args.sample_size, args.debug)
    
    if not pairs:
        print("No pairs loaded, exiting")
        return
    
    # Process pairs
    print(f"Processing {len(pairs)} pairs...")
    
    similarities = []
    labels = []
    processing_times = []
    failed_pairs = 0
    successful_pairs = 0
    
    for i, pair in enumerate(pairs):
        # Start timing for this pair
        pair_start_time = time.time()
        
        if i % 10 == 0:
            print(f"Processing pair {i+1}/{len(pairs)}")
        
        try:
            name1, idx1, name2, idx2, label = pair
            
            # Construct image paths
            img1_path = construct_image_path(args.lfw_dir, name1, idx1, args.debug)
            img2_path = construct_image_path(args.lfw_dir, name2, idx2, args.debug)
            
            if args.debug:
                print(f"Pair {i+1}: {name1}_{idx1} vs {name2}_{idx2}")
            
            # Detect and align faces
            face1 = detect_and_align_face(img1_path, detector, shape_predictor, args.face_size, args.debug)
            face2 = detect_and_align_face(img2_path, detector, shape_predictor, args.face_size, args.debug)
            
            if face1 is None or face2 is None:
                if args.debug:
                    print("Failed to process pair - face detection failed")
                failed_pairs += 1
                continue
            
            # Calculate embeddings
            embedding1 = np.array(face_recognition_model.compute_face_descriptor(face1))
            embedding2 = np.array(face_recognition_model.compute_face_descriptor(face2))
            
            # Calculate similarity (1 - cosine distance)
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Save example images
            if i < 5:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(face1)
                axes[0].set_title(f"{name1}_{idx1}")
                axes[0].axis('off')
                
                axes[1].imshow(face2)
                axes[1].set_title(f"{name2}_{idx2}")
                axes[1].axis('off')
                
                plt.suptitle(f"Pair {i+1}: Similarity = {similarity:.4f}, Same Person = {label==1}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"pair_{i+1}.png"))
                plt.close()
            
            similarities.append(similarity)
            labels.append(label)
            successful_pairs += 1
            
            # Record processing time for this pair
            pair_end_time = time.time()
            processing_times.append(pair_end_time - pair_start_time)
            
        except Exception as e:
            print(f"Error processing pair {i+1}: {e}")
            if args.debug:
                traceback.print_exc()
            failed_pairs += 1
    
    # Check if we have valid results
    if not similarities:
        print("No valid pairs processed. Cannot generate metrics.")
        return
    
    # Generate metrics
    print("Generating metrics...")
    
    # Calculate optimal threshold
    thresholds = np.arange(0, 1, 0.01)
    accuracies = []
    
    for threshold in thresholds:
        predictions = [1 if s >= threshold else 0 for s in similarities]
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        accuracies.append(accuracy)
    
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_threshold_idx]
    best_accuracy = accuracies[best_threshold_idx]
    
    # Calculate ROC
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
    print(f"Total pairs processed: {len(pairs)}")
    print(f"Successful pairs: {successful_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"Success rate: {successful_pairs / len(pairs):.4f}")
    print(f"Verification accuracy: {best_accuracy:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average processing time per pair: {avg_processing_time:.4f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    
    # Generate plots
    # 1. ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Dlib ResNet - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"))
    plt.close()
    
    # 2. Similarity Distribution
    plt.figure(figsize=(10, 8))
    same_similarities = [s for s, l in zip(similarities, labels) if l == 1]
    diff_similarities = [s for s, l in zip(similarities, labels) if l == 0]
    
    if same_similarities:
        plt.hist(same_similarities, bins=30, alpha=0.5, label='Same Person', color='green')
    if diff_similarities:
        plt.hist(diff_similarities, bins=30, alpha=0.5, label='Different Person', color='red')
        
    plt.axvline(x=best_threshold, color='black', linestyle='--', 
                label=f'Threshold ({best_threshold:.2f})')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Dlib ResNet - Distribution of Similarity Scores')
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
        plt.title('Dlib - Processing Time Distribution')
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
        plt.title('Dlib - Processing Time Over Pairs')
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
        plt.title('Dlib - Detailed Processing Time Analysis')
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
    
    # Save results to JSON
    results = {
        "model": "Dlib ResNet",
        "total_pairs": len(pairs),
        "successful_pairs": successful_pairs,
        "failed_pairs": failed_pairs,
        "success_rate": float(successful_pairs / len(pairs)),
        "accuracy": float(best_accuracy),
        "threshold": float(best_threshold),
        "roc_auc": float(roc_auc),
        "timing": {
            "avg_processing_time": float(avg_processing_time),
            "min_processing_time": float(min_processing_time),
            "max_processing_time": float(max_processing_time),
            "total_processing_time": float(total_processing_time)
        }
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
