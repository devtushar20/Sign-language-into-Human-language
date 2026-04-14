"""
utils.py - Core utility functions for landmark extraction
----------------------------------------------------------
KEY UPDATE: Landmark Normalization
  - All coordinates are normalized relative to WRIST position (origin)
  - Scaled by wrist→middle-finger-MCP distance (hand-size invariant)
  - Applied identically in both training (video) and inference (webcam)
  - This eliminates the domain gap caused by different hand positions / distances
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from config import VIDEO_DIR, SEQUENCE_LENGTH, NUM_LANDMARKS

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# MediaPipe landmark indices
WRIST_IDX   = 0   # wrist
MIDDLE_MCP  = 9   # middle finger MCP — used for scale reference


def normalize_hand(raw_lms_21x3):
    """
    Normalize a single hand's 21 landmarks (shape 21x3).
    Steps:
      1. Translate so wrist is at (0,0,0)
      2. Scale so wrist→middle-MCP distance = 1.0
    Returns flattened array of shape (63,).
    """
    pts = raw_lms_21x3.copy()           # (21, 3)

    # 1. Translate: wrist becomes origin
    wrist = pts[WRIST_IDX].copy()
    pts -= wrist

    # 2. Scale: hand-size invariant
    scale = np.linalg.norm(pts[MIDDLE_MCP])  # distance wrist→middle MCP
    if scale > 1e-6:
        pts /= scale

    return pts.flatten()                # (63,)


def extract_landmarks_from_frame(frame):
    """
    Extracts and NORMALIZES hand landmarks from a single BGR frame.
    Returns:
      frame_landmarks : np.array of shape (126,) — normalized
      results         : raw MediaPipe results (for drawing)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame_landmarks = np.zeros(NUM_LANDMARKS, dtype=np.float32)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2:
                break
            raw = np.array([[lm.x, lm.y, lm.z]
                             for lm in hand_landmarks.landmark],
                            dtype=np.float32)       # (21, 3)
            norm = normalize_hand(raw)              # (63,)
            start = idx * 63
            frame_landmarks[start:start + 63] = norm

    return frame_landmarks, results


def extract_landmarks_from_video(video_path):
    """
    Extracts NORMALIZED hand landmarks from a video file.
    Samples SEQUENCE_LENGTH frames evenly.
    Returns numpy array of shape (SEQUENCE_LENGTH, 126).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    indices = set(np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int).tolist())
    video_landmarks = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.resize(frame, (640, 480))
            lm, _ = extract_landmarks_from_frame(frame)
            video_landmarks.append(lm)

    cap.release()

    if len(video_landmarks) < SEQUENCE_LENGTH:
        padding = [np.zeros(NUM_LANDMARKS, dtype=np.float32)] * (SEQUENCE_LENGTH - len(video_landmarks))
        video_landmarks.extend(padding)

    return np.array(video_landmarks[:SEQUENCE_LENGTH], dtype=np.float32)


def load_wlasl_json(json_path, video_dir):
    """Parses WLASL nslt_100.json. Returns list of (video_path, label, split)."""
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return []

    class_list_path = os.path.join(os.path.dirname(json_path), "wlasl_class_list.txt")
    idx_to_gloss = {}
    if os.path.exists(class_list_path):
        with open(class_list_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    idx_to_gloss[int(parts[0])] = parts[1]

    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for video_id, info in data.items():
        split = info['subset']
        gloss_index = info['action'][0]
        gloss = idx_to_gloss.get(gloss_index, str(gloss_index))
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if video_dir == "" or os.path.exists(video_path):
            samples.append((video_path, gloss, split))

    return samples


def get_label_map(json_path):
    """Returns gloss→index and index→gloss mappings."""
    samples = load_wlasl_json(json_path, "")
    glosses = sorted(list(set(s[1] for s in samples)))
    label_to_idx = {gloss: i for i, gloss in enumerate(glosses)}
    idx_to_label = {i: gloss for gloss, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


if __name__ == "__main__":
    from config import JSON_PATH, VIDEO_DIR
    l2i, i2l = get_label_map(JSON_PATH)
    print(f"Number of classes: {len(l2i)}")
    samples = load_wlasl_json(JSON_PATH, VIDEO_DIR)
    print(f"Number of available videos: {len(samples)}")
