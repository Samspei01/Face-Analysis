#!/usr/bin/env python3
"""
Image Uniqueness Checker

This script checks if all images in a given dataset are unique by using perceptual hash (pHash)
to compare image content. It can identify and report duplicate images even if they have
different filenames.

Usage:
    python check_unique_images.py --dataset <path_to_dataset> [--output <output_dir>]
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check if all images in dataset are unique')
    parser.add_argument('--dataset', type=str, 
                      default='/home/samsepi0l/Project/FaceRecognition/face/papers/data/lfw-deepfunneled/lfw-deepfunneled',
                      help='Path to the image dataset')
    parser.add_argument('--output', type=str, default='unique_images_check_results',
                      help='Directory to save results')
    parser.add_argument('--threshold', type=int, default=4,
                      help='Hash difference threshold (lower = stricter matching)')
    parser.add_argument('--sample_size', type=int, default=10000,
                      help='Number of images to sample for checking (0 for all)')
    parser.add_argument('--visualize', type=int, default=5,
                      help='Number of duplicate pairs to visualize (if found)')
    return parser.parse_args()

def load_images(dataset_path, sample_size=0):
    """Load image paths from the dataset"""
    print(f"Finding images in: {dataset_path}")
    all_images = []
    
    # Check if it's a directory structure with subdirectories (like LFW)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} total images")
    
    # Sample images if requested
    if 0 < sample_size < len(all_images):
        print(f"Sampling {sample_size} images for checking")
        return random.sample(all_images, sample_size)
    else:
        print(f"Checking all {len(all_images)} images")
        return all_images

def compute_phash(image_path):
    """Compute perceptual hash for an image"""
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_duplicate_images(image_paths, threshold=4):
    """Find duplicate images based on perceptual hash similarity"""
    print("Computing image hashes...")
    image_hashes = {}
    hash_dict = defaultdict(list)
    
    for image_path in tqdm(image_paths):
        img_hash = compute_phash(image_path)
        if img_hash is not None:
            image_hashes[image_path] = img_hash
            # Store images by their hash for exact duplicates
            hash_dict[img_hash].append(image_path)
    
    # Find exact duplicates (same hash)
    exact_duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    
    # Find near-duplicates (hash difference within threshold)
    near_duplicates = []
    
    print("Looking for near-duplicate images...")
    # Convert hash strings to hash objects for comparison
    hash_objects = {path: imagehash.hex_to_hash(hash_str) for path, hash_str in image_hashes.items()}
    
    # To avoid excessive comparisons, first check exact duplicates
    paths_to_check = [p for p in image_paths if p not in [item for sublist in exact_duplicates.values() for item in sublist]]
    
    # Only compare a subsample if there are many images
    if len(paths_to_check) > 1000:
        print(f"Many images to compare ({len(paths_to_check)}), sampling 1000 for near-duplicate check")
        paths_to_check = random.sample(paths_to_check, 1000)
    
    # Compare hashes
    for i, path1 in enumerate(tqdm(paths_to_check)):
        if path1 not in hash_objects:
            continue
            
        hash1 = hash_objects[path1]
        
        for path2 in paths_to_check[i+1:]:
            if path2 not in hash_objects:
                continue
                
            hash2 = hash_objects[path2]
            
            # Check if the hashes are similar
            if hash1 - hash2 <= threshold and hash1 - hash2 > 0:
                near_duplicates.append((path1, path2, hash1 - hash2))
    
    return exact_duplicates, near_duplicates

def visualize_duplicates(exact_duplicates, near_duplicates, output_dir, num_to_visualize=5):
    """Create visualizations of duplicate images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize exact duplicates
    exact_dup_count = min(num_to_visualize, len(exact_duplicates))
    if exact_dup_count > 0:
        print(f"Creating visualizations for {exact_dup_count} exact duplicate sets...")
        exact_keys = list(exact_duplicates.keys())[:exact_dup_count]
        
        for i, hash_key in enumerate(exact_keys):
            paths = exact_duplicates[hash_key]
            visualize_image_set(paths, f"exact_duplicate_set_{i+1}", output_dir, f"Hash: {hash_key}")
    
    # Visualize near duplicates
    near_dup_count = min(num_to_visualize, len(near_duplicates))
    if near_dup_count > 0:
        print(f"Creating visualizations for {near_dup_count} near-duplicate pairs...")
        for i, (path1, path2, diff) in enumerate(near_duplicates[:near_dup_count]):
            visualize_image_pair(path1, path2, f"near_duplicate_pair_{i+1}", output_dir, f"Hash Diff: {diff}")
    
    print(f"Visualizations saved to {output_dir}")

def visualize_image_set(image_paths, name, output_dir, title=""):
    """Create a visualization for a set of duplicate images"""
    n = len(image_paths)
    cols = min(n, 3)  # Use at most 3 columns
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(4*cols, 4*rows))
    plt.suptitle(title, fontsize=16)
    
    for i, path in enumerate(image_paths):
        plt.subplot(rows, cols, i+1)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(os.path.basename(path), fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Add space for the title
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()

def visualize_image_pair(path1, path2, name, output_dir, title=""):
    """Create a visualization for a pair of near-duplicate images"""
    plt.figure(figsize=(8, 4))
    plt.suptitle(title, fontsize=16)
    
    # Image 1
    plt.subplot(1, 2, 1)
    img1 = cv2.imread(path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.title(os.path.basename(path1), fontsize=10)
    plt.axis('off')
    
    # Image 2
    plt.subplot(1, 2, 2)
    img2 = cv2.imread(path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.title(os.path.basename(path2), fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Add space for the title
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()

def save_report(exact_duplicates, near_duplicates, total_images, output_dir):
    """Save a text report of duplicate findings"""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "duplicate_report.txt")
    
    exact_dup_count = sum(len(paths) for paths in exact_duplicates.values()) - len(exact_duplicates)
    near_dup_count = len(near_duplicates)
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("IMAGE UNIQUENESS ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total images analyzed: {total_images}\n")
        f.write(f"Exact duplicates found: {exact_dup_count}\n")
        f.write(f"Near-duplicates found: {near_dup_count}\n\n")
        
        unique_percent = ((total_images - exact_dup_count) / total_images) * 100 if total_images > 0 else 0
        f.write(f"Dataset uniqueness: {unique_percent:.2f}%\n\n")
        
        # Write exact duplicate details
        if exact_duplicates:
            f.write("EXACT DUPLICATES:\n")
            f.write("-" * 70 + "\n")
            
            for i, (hash_val, paths) in enumerate(exact_duplicates.items()):
                f.write(f"Duplicate Set #{i+1} (Hash: {hash_val}):\n")
                for path in paths:
                    f.write(f"  - {path}\n")
                f.write("\n")
        
        # Write near duplicate details
        if near_duplicates:
            f.write("NEAR DUPLICATES:\n")
            f.write("-" * 70 + "\n")
            
            for i, (path1, path2, diff) in enumerate(near_duplicates):
                f.write(f"Near-Duplicate Pair #{i+1} (Hash difference: {diff}):\n")
                f.write(f"  - {path1}\n")
                f.write(f"  - {path2}\n")
                f.write("\n")
    
    print(f"Report saved to {report_path}")
    
    return unique_percent

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load image paths
    image_paths = load_images(args.dataset, args.sample_size)
    
    if not image_paths:
        print("No images found. Please check the dataset path.")
        return
    
    # Find duplicates
    exact_duplicates, near_duplicates = find_duplicate_images(image_paths, args.threshold)
    
    # Save report
    uniqueness_score = save_report(exact_duplicates, near_duplicates, len(image_paths), args.output)
    
    # Create visualizations
    if exact_duplicates or near_duplicates:
        visualize_duplicates(exact_duplicates, near_duplicates, args.output, args.visualize)
    
    # Print summary
    exact_dup_count = sum(len(paths) for paths in exact_duplicates.values()) - len(exact_duplicates)
    near_dup_count = len(near_duplicates)
    total_dup_count = exact_dup_count + near_dup_count
    
    print("\n" + "=" * 70)
    print("IMAGE UNIQUENESS ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Total images analyzed: {len(image_paths)}")
    print(f"Exact duplicates found: {exact_dup_count}")
    print(f"Near-duplicates found: {near_dup_count}")
    print(f"Total potential duplicates: {total_dup_count}")
    print(f"Dataset uniqueness score: {uniqueness_score:.2f}%")
    
    if total_dup_count > 0:
        print(f"\nDuplicate details saved to {args.output}/duplicate_report.txt")
        print(f"Visualizations saved to {args.output}/")
    else:
        print("\nGreat news! No duplicate images were found in the dataset.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
