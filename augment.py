"""
augment.py - Data Augmentation for Sign Language Landmark Sequences

Generates augmented .npy files from existing landmark files.
Augmentation strategies:
  - Jitter:      Add Gaussian noise to coordinates
  - Scale:       Random scale of x/y coordinates
  - Mirror:      Flip x-axis (simulate left-hand signing)
  - Time Warp:   Slight temporal stretching/compression
  - Frame Shift: Roll the sequence slightly in time
"""

import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CUSTOM_LANDMARKS_DIR, AUGMENTATIONS_PER_SAMPLE, SEQUENCE_LENGTH, NUM_LANDMARKS


# ---------- augmentation primitives ----------

def augment_jitter(seq, sigma=0.006):
    """Add Gaussian noise to all landmark coordinates."""
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise


def augment_scale(seq, scale_range=(0.88, 1.12)):
    """Randomly scale x and y coordinates (every 3rd col is z, leave it alone)."""
    aug = seq.copy()
    sx = np.random.uniform(*scale_range)
    sy = np.random.uniform(*scale_range)
    # columns: x=0,3,6,…  y=1,4,7,…  z=2,5,8,…
    aug[:, 0::3] *= sx
    aug[:, 1::3] *= sy
    # clip to reasonable landmark range
    aug = np.clip(aug, -1.5, 2.5)
    return aug


def augment_mirror(seq):
    """Mirror x-coordinates (simulate opposite hand)."""
    aug = seq.copy()
    aug[:, 0::3] = 1.0 - aug[:, 0::3]
    return aug


def augment_time_warp(seq, warp_factor_range=(0.85, 1.15)):
    """Stretch or compress the temporal axis slightly."""
    T = seq.shape[0]
    warp = np.random.uniform(*warp_factor_range)
    src_indices = np.linspace(0, T - 1, int(T * warp))
    src_indices = np.clip(src_indices, 0, T - 1)
    # resample to original length
    tgt_indices = np.linspace(0, len(src_indices) - 1, T)
    warped = np.zeros_like(seq)
    for j in range(seq.shape[1]):
        warped[:, j] = np.interp(tgt_indices, np.arange(len(src_indices)), seq[src_indices.astype(int), j])
    return warped


def augment_frame_shift(seq, max_shift=3):
    """Shift frames forward/backward, padding with zeros."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(seq, shift, axis=0)


def augment_dropout(seq, drop_prob=0.08):
    """Randomly zero out entire frames to simulate occlusion."""
    aug = seq.copy()
    mask = np.random.rand(seq.shape[0]) < drop_prob
    aug[mask] = 0
    return aug


# All augmentation functions
AUGMENTERS = [
    augment_jitter,
    augment_scale,
    augment_mirror,
    augment_time_warp,
    augment_frame_shift,
    augment_dropout,
]


def augment_sequence(seq):
    """
    Apply a random combination of 1–3 augmentations to a sequence.
    Returns one augmented copy.
    """
    num_augs = np.random.randint(1, 4)
    chosen = np.random.choice(len(AUGMENTERS), size=num_augs, replace=False)
    aug_seq = seq.copy().astype(np.float32)
    for idx in chosen:
        aug_seq = AUGMENTERS[idx](aug_seq)
    return aug_seq


# ---------- main augmentation runner ----------

def run_augmentation(landmarks_dir=None, augmentations_per_sample=None, check_only=False):
    """
    For each .npy file in landmarks_dir, generate augmented copies
    saved as  <video_id>_aug<n>.npy  in the same folder.

    If check_only=True, just print counts without writing files.
    """
    if landmarks_dir is None:
        landmarks_dir = CUSTOM_LANDMARKS_DIR
    if augmentations_per_sample is None:
        augmentations_per_sample = AUGMENTATIONS_PER_SAMPLE

    total_real = 0
    total_existing_aug = 0
    total_new_aug = 0

    classes = [d for d in os.listdir(landmarks_dir)
               if os.path.isdir(os.path.join(landmarks_dir, d))]

    for gloss in tqdm(classes, desc="Augmenting classes"):
        gloss_dir = os.path.join(landmarks_dir, gloss)
        all_files = os.listdir(gloss_dir)
        real_files = [f for f in all_files if f.endswith('.npy') and '_aug' not in f]
        aug_files  = [f for f in all_files if f.endswith('.npy') and '_aug' in f]

        total_real += len(real_files)
        total_existing_aug += len(aug_files)

        if check_only:
            continue

        for real_file in real_files:
            real_path = os.path.join(gloss_dir, real_file)
            base_name = real_file.replace('.npy', '')
            seq = np.load(real_path).astype(np.float32)

            for n in range(augmentations_per_sample):
                aug_filename = f"{base_name}_aug{n}.npy"
                aug_path = os.path.join(gloss_dir, aug_filename)
                if os.path.exists(aug_path):
                    continue  # skip already-augmented
                aug_seq = augment_sequence(seq)
                np.save(aug_path, aug_seq.astype(np.float32))
                total_new_aug += 1

    total_after = total_real + total_existing_aug + total_new_aug
    print(f"\n{'='*50}")
    print(f"Real samples      : {total_real}")
    print(f"Pre-existing augs : {total_existing_aug}")
    print(f"Newly created augs: {total_new_aug}")
    print(f"TOTAL samples     : {total_after}")
    print(f"{'='*50}")
    return total_after


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='Only print counts, do not write files')
    parser.add_argument('--dir', default=None,
                        help='Override landmarks directory')
    args = parser.parse_args()
    run_augmentation(landmarks_dir=args.dir, check_only=args.check)
