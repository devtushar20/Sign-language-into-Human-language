import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from tqdm import tqdm
from config import JSON_PATH, VIDEO_DIR, LANDMARKS_DIR, LABEL_MAP_PATH
from utils import extract_landmarks_from_video, load_wlasl_json, get_label_map

def preprocess_dataset():
    # 1. Load label map
    label_to_idx, idx_to_label = get_label_map(JSON_PATH)

    # Save label map for future use
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_to_idx, f)

    # 2. Load samples
    samples = load_wlasl_json(JSON_PATH, VIDEO_DIR)
    print(f"Total samples to process: {len(samples)}")

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for video_path, gloss, split in tqdm(samples, desc="Processing videos"):
        # Create gloss directory
        gloss_dir = os.path.join(LANDMARKS_DIR, gloss)
        os.makedirs(gloss_dir, exist_ok=True)

        video_id = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(gloss_dir, f"{video_id}.npy")

        # Resume support: skip if file exists
        if os.path.exists(output_path):
            skipped_count += 1
            continue

        # Extract landmarks
        landmarks = extract_landmarks_from_video(video_path)

        if landmarks is not None:
            if np.all(landmarks == 0):
                failed_count += 1
            else:
                np.save(output_path, landmarks)
                processed_count += 1
        else:
            failed_count += 1

    print("\nPreprocessing Complete!")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped (already exist): {skipped_count}")
    print(f"Total failed: {failed_count}")

if __name__ == "__main__":
    preprocess_dataset()
