#!/usr/bin/env python3
"""
Generate Balanced Pairs for LFW Evaluation

This script creates a balanced dataset with equal numbers of positive (same person) 
and negative (different person) pairs from the LFW dataset.
"""

import os
import csv
import random
from collections import defaultdict

def get_person_images(lfw_dir):
    """Get all available images for each person in the LFW dataset"""
    person_images = defaultdict(list)
    
    lfw_deepfunneled_path = os.path.join(lfw_dir, "lfw-deepfunneled")
    if not os.path.exists(lfw_deepfunneled_path):
        print(f"Error: LFW directory {lfw_deepfunneled_path} does not exist")
        return person_images
    
    for person_dir in os.listdir(lfw_deepfunneled_path):
        person_path = os.path.join(lfw_deepfunneled_path, person_dir)
        if os.path.isdir(person_path):
            # Get all image files in the person directory
            images = []
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract image number from filename (e.g., "Abel_Pacheco_0001.jpg" -> 1)
                    try:
                        img_num = int(img_file.split('_')[-1].split('.')[0])
                        images.append(img_num)
                    except (ValueError, IndexError):
                        continue
            
            if images:
                person_images[person_dir] = sorted(images)
    
    return person_images

def generate_positive_pairs(person_images, num_pairs):
    """Generate positive pairs (same person)"""
    positive_pairs = []
    persons_with_multiple_images = {name: imgs for name, imgs in person_images.items() if len(imgs) >= 2}
    
    if not persons_with_multiple_images:
        print("Warning: No persons with multiple images found")
        return positive_pairs
    
    while len(positive_pairs) < num_pairs:
        # Randomly select a person with multiple images
        person = random.choice(list(persons_with_multiple_images.keys()))
        images = persons_with_multiple_images[person]
        
        # Randomly select two different images
        img1, img2 = random.sample(images, 2)
        
        # Create the pair
        pair = (person, img1, person, img2, 1)
        if pair not in positive_pairs:
            positive_pairs.append(pair)
    
    return positive_pairs

def generate_negative_pairs(person_images, num_pairs):
    """Generate negative pairs (different persons)"""
    negative_pairs = []
    person_names = list(person_images.keys())
    
    if len(person_names) < 2:
        print("Warning: Not enough persons to generate negative pairs")
        return negative_pairs
    
    while len(negative_pairs) < num_pairs:
        # Randomly select two different persons
        person1, person2 = random.sample(person_names, 2)
        
        # Randomly select one image from each person
        img1 = random.choice(person_images[person1])
        img2 = random.choice(person_images[person2])
        
        # Create the pair
        pair = (person1, img1, person2, img2, 0)
        if pair not in negative_pairs:
            negative_pairs.append(pair)
    
    return negative_pairs

def main():
    print("Starting balanced pairs generation...")
    lfw_dir = "./lfw-deepfunneled"
    output_file = "./pairs_balanced.csv"
    num_positive_pairs = 3000  # Half of the original dataset
    num_negative_pairs = 3000  # Equal number of negative pairs
    
    print("Scanning LFW dataset...")
    person_images = get_person_images(lfw_dir)
    
    if not person_images:
        print("Error: No person images found in LFW dataset")
        return
    
    print(f"Found {len(person_images)} persons in the dataset")
    
    # Print some statistics
    persons_with_multiple = sum(1 for imgs in person_images.values() if len(imgs) >= 2)
    print(f"Persons with multiple images: {persons_with_multiple}")
    
    print("Generating positive pairs...")
    positive_pairs = generate_positive_pairs(person_images, num_positive_pairs)
    print(f"Generated {len(positive_pairs)} positive pairs")
    
    print("Generating negative pairs...")
    negative_pairs = generate_negative_pairs(person_images, num_negative_pairs)
    print(f"Generated {len(negative_pairs)} negative pairs")
    
    # Combine and shuffle pairs
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    print(f"Writing {len(all_pairs)} pairs to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name1', 'imagenum1', 'name2', 'imagenum2', 'label'])
        
        for pair in all_pairs:
            writer.writerow(pair)
    
    print(f"Balanced pairs dataset created: {output_file}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Positive pairs: {len(positive_pairs)} (label=1)")
    print(f"Negative pairs: {len(negative_pairs)} (label=0)")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    main()
