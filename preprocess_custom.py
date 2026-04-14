"""
preprocess_custom.py
-------------------
1. Reads custom_subset.json (30 target signs from WLASL_v0.3.json format)
2. Extracts hand landmarks using MediaPipe for each video
3. Copies usable landmarks from existing `landmarks/` dir if already available
4. Runs data augmentation to reach >= 5000 total training samples
5. Saves label_map.json

Run ONCE before training:
    python preprocess_custom.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import shutil
import numpy as np
from tqdm import tqdm

from config import (
    VIDEO_DIR, LANDMARKS_DIR, CUSTOM_LANDMARKS_DIR,
    CUSTOM_SUBSET_JSON, LABEL_MAP_PATH, MODELS_DIR,
    CUSTOM_SIGNS, AUGMENTATIONS_PER_SAMPLE
)
from utils import extract_landmarks_from_video


def build_label_map():
    """Build label map from CUSTOM_SIGNS list."""
    signs = sorted([s.lower() for s in CUSTOM_SIGNS])
    label_to_idx = {gloss: i for i, gloss in enumerate(signs)}
    return label_to_idx


def copy_existing_landmarks(label_to_idx):
    """
    Copy already-processed .npy files from the old landmarks/ dir
    for signs that overlap with our 30-sign vocabulary.
    Returns count of copied files.
    """
    copied = 0
    for gloss in label_to_idx:
        src_dir = os.path.join(LANDMARKS_DIR, gloss)
        dst_dir = os.path.join(CUSTOM_LANDMARKS_DIR, gloss)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src_dir):
            for fname in os.listdir(src_dir):
                if fname.endswith('.npy') and '_aug' not in fname:
                    src = os.path.join(src_dir, fname)
                    dst = os.path.join(dst_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied += 1
    return copied


def process_custom_subset(label_to_idx):
    """
    Process videos listed in custom_subset.json.
    Returns (processed_count, skipped_count, failed_count).
    """
    if not os.path.exists(CUSTOM_SUBSET_JSON):
        print(f"custom_subset.json not found at {CUSTOM_SUBSET_JSON}")
        print("Run create_custom_subset.py first.")
        return 0, 0, 0

    with open(CUSTOM_SUBSET_JSON, 'r') as f:
        data = json.load(f)

    processed = skipped = failed = 0

    samples = []
    for video_id, info in data.items():
        gloss = info['action'][0].lower()
        if gloss not in label_to_idx:
            continue
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        samples.append((video_id, gloss, video_path))

    for video_id, gloss, video_path in tqdm(samples, desc="Processing videos"):
        dst_dir = os.path.join(CUSTOM_LANDMARKS_DIR, gloss)
        os.makedirs(dst_dir, exist_ok=True)
        out_path = os.path.join(dst_dir, f"{video_id}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        if not os.path.exists(video_path):
            failed += 1
            continue

        landmarks = extract_landmarks_from_video(video_path)
        if landmarks is not None and not np.all(landmarks == 0):
            np.save(out_path, landmarks)
            processed += 1
        else:
            failed += 1

    return processed, skipped, failed


def count_samples():
    """Count total .npy files in custom_landmarks/."""
    total = 0
    per_class = {}
    for gloss in os.listdir(CUSTOM_LANDMARKS_DIR):
        gloss_dir = os.path.join(CUSTOM_LANDMARKS_DIR, gloss)
        if os.path.isdir(gloss_dir):
            count = len([f for f in os.listdir(gloss_dir) if f.endswith('.npy')])
            per_class[gloss] = count
            total += count
    return total, per_class


def main():
    print("=" * 60)
    print("  Sign Language Custom Preprocessing Pipeline")
    print("=" * 60)

    # 1. Build label map
    label_to_idx = build_label_map()
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    print(f"Label map saved: {len(label_to_idx)} classes")
    print(f"Signs: {list(label_to_idx.keys())}\n")

    # 2. Copy existing landmarks
    print("Step 1/3: Copying existing landmarks from landmarks/ ...")
    copied = copy_existing_landmarks(label_to_idx)
    print(f"  Copied {copied} existing .npy files.\n")

    # 3. Process remaining videos from custom_subset.json
    print("Step 2/3: Processing videos from custom_subset.json ...")
    proc, skip, fail = process_custom_subset(label_to_idx)
    print(f"  Processed: {proc}  |  Skipped: {skip}  |  Failed: {fail}\n")

    # 4. Count before augmentation
    total_before, per_class = count_samples()
    print(f"Step 3/3: Real samples before augmentation: {total_before}")
    print("  Samples per class:")
    for g, c in sorted(per_class.items()):
        print(f"    {g:20s}: {c}")

    # 5. Run augmentation
    print(f"\nRunning augmentation ({AUGMENTATIONS_PER_SAMPLE} copies per sample)...")
    from augment import run_augmentation
    total_after = run_augmentation(
        landmarks_dir=CUSTOM_LANDMARKS_DIR,
        augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE
    )

    print(f"\nDone! Total training samples available: {total_after}")
    if total_after >= 5000:
        print("  >= 5000 samples achieved!")
    else:
        print(f"  Warning: Only {total_after} samples. Consider adding more videos.")


if __name__ == "__main__":
    main()
