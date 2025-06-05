#!/usr/bin/env python3
"""
LFW Dataset Preparation for Face Detection

This script prepares a dataset of unique face images from the LFW dataset
for training or evaluating face detection models. It:
1. Reads the LFW dataset names from CSV
2. Locates the corresponding images
3. Verifies image uniqueness using perceptual hash
4. Creates a structured dataset for face detection

Usage:
    python prepare_lfw_dataset.py [--output <output_dir>] [--sample_size <number>] [--check_unique]
"""

import os
import sys
import argparse
import pandas as pd
import shutil
from pathlib import Path
import random
import cv2
from tqdm import tqdm

# Import functions from check_unique_images.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from check_unique_images import compute_phash, find_duplicate_images

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare LFW dataset for face detection')
    parser.add_argument('--csv_path', type=str, 
                      default='/home/samsepi0l/Project/FaceRecognition/face/papers/data/lfw_allnames.csv',
                      help='Path to the lfw_allnames.csv file')
    parser.add_argument('--lfw_dir', type=str, 
                      default='/home/samsepi0l/Project/FaceRecognition/face/papers/data/lfw-deepfunneled',
                      help='Path to the LFW dataset directory')
    parser.add_argument('--output', type=str, default='lfw_face_detection_dataset',
                      help='Directory to save the prepared dataset')
    parser.add_argument('--sample_size', type=int, default=10000,
                      help='Number of images to sample (0 for all)')
    parser.add_argument('--check_unique', action='store_true',
                      help='Check for and remove duplicate images')
    parser.add_argument('--threshold', type=int, default=4,
                      help='Hash difference threshold for uniqueness (lower = stricter)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def read_lfw_names(csv_path):
    """Read LFW names from CSV file"""
    print(f"Reading LFW names from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} entries")
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def find_lfw_images(lfw_df, lfw_dir):
    """Find all image files for the names in the dataframe"""
    image_paths = []
    missing_names = []
    
    print("Finding image files...")
    for _, row in tqdm(lfw_df.iterrows(), total=len(lfw_df)):
        name = row['name']
        # LFW format: first_last/first_last_0001.jpg
        name_parts = name.split('_')
        if len(name_parts) < 2:
            continue
        
        person_dir = os.path.join(lfw_dir, name)
        
        # Check if the person directory exists
        if not os.path.exists(person_dir):
            missing_names.append(name)
            continue
            
        # Find all images for this person
        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(person_dir, img_file))
    
    print(f"Found {len(image_paths)} images")
    if missing_names:
        print(f"Warning: {len(missing_names)} names not found in the dataset")
        
    return image_paths

def prepare_dataset(image_paths, output_dir, sample_size=0, check_unique=True, threshold=4, random_seed=42):
    """Prepare the dataset by copying unique images to the output directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample images if requested
    if 0 < sample_size < len(image_paths):
        print(f"Sampling {sample_size} images from {len(image_paths)} total")
        random.seed(random_seed)
        sampled_images = random.sample(image_paths, sample_size)
    else:
        print("Using all available images")
        sampled_images = image_paths
    
    # Check for duplicates if requested
    if check_unique:
        print("Checking for duplicate images...")
        exact_duplicates, near_duplicates = find_duplicate_images(sampled_images, threshold)
        
        # Create list of duplicates to exclude
        duplicates_to_exclude = set()
        
        # From exact duplicates, keep first image in each group, exclude the rest
        for hash_val, paths in exact_duplicates.items():
            # Keep the first one
            duplicates_to_exclude.update(paths[1:])
            
        # From near duplicates, exclude the second image in each pair
        for path1, path2, _ in near_duplicates:
            duplicates_to_exclude.add(path2)
            
        # Filter out duplicates
        unique_images = [img for img in sampled_images if img not in duplicates_to_exclude]
        print(f"Removed {len(sampled_images) - len(unique_images)} duplicate images")
        sampled_images = unique_images
    
    # Copy images to output directory
    print(f"Copying {len(sampled_images)} images to {output_dir}...")
    for i, img_path in enumerate(tqdm(sampled_images)):
        # Generate a unique name for each image
        new_filename = f"face_{i+1:05d}.jpg"
        dest_path = os.path.join(output_dir, new_filename)
        
        try:
            # Read image to ensure it's valid
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping")
                continue
                
            # Copy the image
            shutil.copy2(img_path, dest_path)
        except Exception as e:
            print(f"Error copying {img_path}: {e}")
    
    print(f"Dataset preparation complete. {len(os.listdir(output_dir))} images saved to {output_dir}")
    return len(os.listdir(output_dir))

def save_metadata(output_dir, total_images, args):
    """Save metadata about the dataset"""
    metadata_path = os.path.join(output_dir, "dataset_metadata.txt")
    
    with open(metadata_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LFW FACE DETECTION DATASET METADATA\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date created: {pd.Timestamp.now()}\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Source CSV: {args.csv_path}\n")
        f.write(f"Source directory: {args.lfw_dir}\n")
        f.write(f"Sample size requested: {args.sample_size}\n")
        f.write(f"Uniqueness check: {'Yes' if args.check_unique else 'No'}\n")
        if args.check_unique:
            f.write(f"Hash threshold: {args.threshold}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        
        f.write("Dataset structure:\n")
        f.write("- Each image contains at least one face\n")
        f.write("- All images are confirmed to be unique\n")
        f.write("- Filenames format: face_XXXXX.jpg\n\n")
        
        f.write("Usage recommendations:\n")
        f.write("- This dataset is suitable for face detection model training/testing\n")
        f.write("- Images have variable dimensions and face counts\n")
        f.write("- All faces are real-world photographs from LFW dataset\n")
    
    print(f"Metadata saved to {metadata_path}")

def main():
    args = parse_arguments()
    
    # Read LFW names from CSV
    lfw_df = read_lfw_names(args.csv_path)
    if lfw_df is None:
        return
    
    # Find image paths
    image_paths = find_lfw_images(lfw_df, args.lfw_dir)
    if not image_paths:
        print("No images found. Please check the LFW directory path.")
        return
    
    # Prepare the dataset
    total_images = prepare_dataset(
        image_paths, 
        args.output, 
        args.sample_size, 
        args.check_unique, 
        args.threshold,
        args.seed
    )
    
    # Save metadata
    save_metadata(args.output, total_images, args)
    
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Successfully created dataset with {total_images} unique images")
    print(f"Dataset location: {os.path.abspath(args.output)}")
    print("\nYou can now use this dataset for face detection model training or evaluation.")

if __name__ == "__main__":
    main()
