"""
collect_data.py  —  Record YOUR OWN sign data via webcam
---------------------------------------------------------
This is the single most effective way to improve real-world accuracy.
The model will learn YOUR hand shape, YOUR camera angle, YOUR style.

HOW TO USE:
  python collect_data.py

Steps in the UI:
  1. A sign name appears on screen (e.g. "HELLO")
  2. Press SPACE to start recording when ready
  3. Hold the sign for ~2 seconds while the progress bar fills
  4. Repeat for each of the 30 samples per sign
  5. Press 'S' to skip a sign, 'Q' to quit early

After collection → run:
  python augment.py          (augment new data)
  python train.py            (retrain model)
  python predict.py          (test improved model)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import json
import time
from collections import deque

from config import CUSTOM_LANDMARKS_DIR, LABEL_MAP_PATH, SEQUENCE_LENGTH, CUSTOM_SIGNS
from utils import extract_landmarks_from_frame, mp_hands, mp_drawing, mp_draw_styles

# ── Settings ────────────────────────────────────────────────────
SAMPLES_PER_SIGN   = 30    # how many sequences to record per sign
COOLDOWN_FRAMES    = 15    # frames to wait between recordings
HAND_REQUIRED      = True  # skip recording if no hand detected

# ── Colours ─────────────────────────────────────────────────────
CLR_BG       = (20,  20,  20)
CLR_GREEN    = (0,  220, 100)
CLR_ORANGE   = (0,  165, 255)
CLR_RED      = (0,   60, 200)
CLR_WHITE    = (255, 255, 255)
CLR_GREY     = (140, 140, 140)
CLR_YELLOW   = (0,  230, 230)


def draw_ui(frame, sign, sample_idx, total_samples, state,
            progress, frame_buf_len, hand_detected, h, w):
    """Draw the collection UI on frame."""
    overlay = frame.copy()

    # Top banner
    cv2.rectangle(overlay, (0, 0), (w, 70), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Sign name
    cv2.putText(frame, f"SIGN: {sign.upper()}", (15, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, CLR_YELLOW, 2, cv2.LINE_AA)

    # Sample counter (top right)
    counter_text = f"{sample_idx}/{total_samples}"
    cv2.putText(frame, counter_text, (w - 130, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, CLR_GREEN, 2, cv2.LINE_AA)

    # Status message
    if state == 'WAITING':
        status = "SPACE = Start recording" if hand_detected else "Show your hand first..."
        color  = CLR_WHITE if hand_detected else CLR_ORANGE
    elif state == 'COUNTDOWN':
        status = f"Get ready..."
        color  = CLR_ORANGE
    elif state == 'RECORDING':
        status = "RECORDING — hold the sign!"
        color  = CLR_RED
    elif state == 'SAVED':
        status = "Saved! Get ready for next..."
        color  = CLR_GREEN
    else:
        status = state
        color  = CLR_WHITE

    cv2.putText(frame, status, (15, h - 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Progress bar (frames collected in current recording)
    if state == 'RECORDING':
        bar_w = int((frame_buf_len / SEQUENCE_LENGTH) * (w - 30))
        cv2.rectangle(frame, (15, h - 75), (w - 15, h - 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, h - 75), (15 + bar_w, h - 50), CLR_RED, -1)
        pct = int(frame_buf_len / SEQUENCE_LENGTH * 100)
        cv2.putText(frame, f"{pct}%", (w // 2 - 20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 2, cv2.LINE_AA)

    # Overall progress bar (samples done)
    overall_w = int((sample_idx / total_samples) * (w - 30))
    cv2.rectangle(frame, (15, h - 30), (w - 15, h - 12), (40, 40, 40), -1)
    cv2.rectangle(frame, (15, h - 30), (15 + overall_w, h - 12), CLR_GREEN, -1)

    # Hand detection indicator
    dot_color = CLR_GREEN if hand_detected else CLR_RED
    cv2.circle(frame, (w - 20, 20), 10, dot_color, -1)

    # Controls hint
    cv2.putText(frame, "S=Skip sign  Q=Quit", (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREY, 1, cv2.LINE_AA)

    return frame


def record_sign(cap, sign, sign_dir, existing_count, h, w):
    """
    Interactively collect SAMPLES_PER_SIGN sequences for one sign.
    Returns number of new samples saved.
    """
    sample_idx = existing_count
    target     = existing_count + SAMPLES_PER_SIGN
    saved_this_session = 0

    state        = 'WAITING'
    frame_buffer = []
    cooldown     = 0

    print(f"\n  Recording: {sign.upper()}  (need {SAMPLES_PER_SIGN} samples)")

    while sample_idx < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        lm, results = extract_landmarks_from_frame(frame)
        hand_detected = results.multi_hand_landmarks is not None

        # Draw skeleton
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )

        # ── State machine ───────────────────────────────────────
        if state == 'WAITING':
            frame_buffer = []
            if cooldown > 0:
                cooldown -= 1

        elif state == 'RECORDING':
            if hand_detected:
                frame_buffer.append(lm)
            # Auto-pad if hand briefly disappears
            else:
                frame_buffer.append(np.zeros(len(lm), dtype=np.float32))

            if len(frame_buffer) >= SEQUENCE_LENGTH:
                seq = np.array(frame_buffer[:SEQUENCE_LENGTH], dtype=np.float32)
                fname = f"user_{sample_idx:04d}.npy"
                np.save(os.path.join(sign_dir, fname), seq)
                sample_idx += 1
                saved_this_session += 1
                print(f"    Saved sample {sample_idx}/{target}")
                state    = 'SAVED'
                cooldown = COOLDOWN_FRAMES

        elif state == 'SAVED':
            cooldown -= 1
            if cooldown <= 0:
                state = 'WAITING'

        # ── Draw UI ─────────────────────────────────────────────
        draw_ui(frame, sign, sample_idx - existing_count,
                SAMPLES_PER_SIGN, state,
                0, len(frame_buffer), hand_detected, h, w)
        cv2.imshow('Sign Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return saved_this_session, True   # quit flag

        elif key == ord('s'):
            print(f"    Skipped {sign}")
            return saved_this_session, False

        elif key == ord(' '):
            if state == 'WAITING' and cooldown == 0:
                if hand_detected or not HAND_REQUIRED:
                    state = 'RECORDING'
                    frame_buffer = []

    print(f"    Completed {sign}: {saved_this_session} new samples")
    return saved_this_session, False


def main():
    print("=" * 60)
    print("  Sign Language Data Collector")
    print("=" * 60)
    print(f"  Signs to collect : {len(CUSTOM_SIGNS)}")
    print(f"  Samples per sign : {SAMPLES_PER_SIGN}")
    print(f"  Total sequences  : {len(CUSTOM_SIGNS) * SAMPLES_PER_SIGN}")
    print()
    print("  Controls:")
    print("    SPACE  = Start recording a sample")
    print("    S      = Skip current sign")
    print("    Q      = Quit early")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640

    os.makedirs(CUSTOM_LANDMARKS_DIR, exist_ok=True)
    total_saved = 0
    signs = sorted([s.lower() for s in CUSTOM_SIGNS])

    for sign in signs:
        sign_dir = os.path.join(CUSTOM_LANDMARKS_DIR, sign)
        os.makedirs(sign_dir, exist_ok=True)

        # Count already-recorded user samples for this sign
        existing_user = [f for f in os.listdir(sign_dir)
                         if f.startswith('user_') and f.endswith('.npy')]
        existing_count = len(existing_user)

        if existing_count >= SAMPLES_PER_SIGN:
            print(f"  {sign:20s}: already has {existing_count} samples — skipping")
            continue

        saved, quit_flag = record_sign(cap, sign, sign_dir, existing_count, h, w)
        total_saved += saved

        if quit_flag:
            print("\nStopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  Collection done! Total new samples: {total_saved}")
    print(f"{'='*60}")

    if total_saved > 0:
        print("\nNext steps:")
        print("  1. python augment.py          (augment your new data)")
        print("  2. python train.py            (retrain the model)")
        print("  3. python predict.py          (test your improved model)")


if __name__ == "__main__":
    main()
