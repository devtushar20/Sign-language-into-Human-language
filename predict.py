"""
predict.py - Real-time Sign Language Recognition (Improved)
------------------------------------------------------------
Improvements:
  - NORMALIZED landmarks (matches training pipeline exactly)
  - Predict every 5 frames (was 10) → more responsive
  - Buffer resets when hand disappears for >10 frames
  - 5-vote majority smoothing before committing a sign
  - Confidence threshold: 0.55 (normalized landmarks are more discriminative)
  - Buffer fill bar shows progress toward a full 30-frame window
  - Top-3 predictions shown with colour-coded bars
  - Press SPACE to manually trigger prediction
  - Press 'c' to clear sentence, 'q' to quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
import numpy as np
import json
from collections import deque, Counter
from tensorflow.keras.models import load_model

from config import MODEL_SAVE_PATH, LABEL_MAP_PATH, SEQUENCE_LENGTH, NUM_LANDMARKS
from utils import extract_landmarks_from_frame, mp_draw_styles
from train import AttentionLayer

# ── Tuning knobs ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.55   # lowered — normalized landmarks are tighter
PREDICT_EVERY_N_FRAMES = 5      # predict more often
VOTE_WINDOW            = 5      # need 5 matching predictions to commit
HAND_ABSENT_RESET      = 12     # frames without hand before buffer resets

# ── MediaPipe ───────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ── UI helpers ──────────────────────────────────────────────────
CLR_GREEN  = (0, 220, 100)
CLR_BLUE   = (200, 150, 0)
CLR_PURPLE = (200, 100, 100)
CLR_RED    = (0, 60, 230)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = (25, 25, 25)
CLR_GREY   = (100, 100, 100)
CLR_YELLOW = (0, 230, 230)


def draw_top3(frame, labels, probs, w):
    """Draw top-3 prediction bars in the top-right corner."""
    bar_x       = w - 270
    bar_y_start = 60
    bar_h       = 28
    bar_max_w   = 230
    colors      = [CLR_GREEN, CLR_BLUE, CLR_PURPLE]

    for i, (label, prob) in enumerate(zip(labels, probs)):
        y = bar_y_start + i * (bar_h + 8)
        cv2.rectangle(frame, (bar_x, y),
                      (bar_x + bar_max_w, y + bar_h), (40, 40, 40), -1)
        fill = int(prob * bar_max_w)
        cv2.rectangle(frame, (bar_x, y),
                      (bar_x + fill, y + bar_h), colors[i], -1)
        cv2.putText(frame, f"{label}: {prob*100:.1f}%",
                    (bar_x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_WHITE, 2, cv2.LINE_AA)


def draw_buffer_bar(frame, buf_len, h, w):
    """Show how full the 30-frame buffer is (bottom-left)."""
    fill_w = int((buf_len / SEQUENCE_LENGTH) * (w // 2))
    cv2.rectangle(frame, (0, h - 8), (w // 2, h), CLR_GREY, -1)
    cv2.rectangle(frame, (0, h - 8), (fill_w, h),
                  CLR_GREEN if buf_len == SEQUENCE_LENGTH else CLR_BLUE, -1)
    cv2.putText(frame, f"Buffer: {buf_len}/{SEQUENCE_LENGTH}",
                (5, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)


# ── Main inference loop ─────────────────────────────────────────
def run_inference():
    # Load model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found: {MODEL_SAVE_PATH}")
        print("Run train.py first!")
        return

    print("Loading model...")
    model = load_model(MODEL_SAVE_PATH,
                       custom_objects={'AttentionLayer': AttentionLayer})

    with open(LABEL_MAP_PATH, 'r') as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # ── State ──────────────────────────────────────────────────
    sequence        = deque(maxlen=SEQUENCE_LENGTH)   # rolling 30-frame buffer
    vote_buffer     = deque(maxlen=VOTE_WINDOW)        # last N predictions
    sentence        = []
    frame_count     = 0
    hand_absent_cnt = 0

    # display state
    current_label = "..."
    current_conf  = 0.0
    top3_labels   = ["—", "—", "—"]
    top3_probs    = [0.0, 0.0, 0.0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    print(f"Webcam: {w}x{h}")
    print("Press  Q=quit  C=clear  SPACE=force predict")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Extract normalized landmarks ────────────────────────
        lm, results = extract_landmarks_from_frame(frame)
        hand_detected = results.multi_hand_landmarks is not None

        # Draw skeleton
        if hand_detected:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )

        # ── Buffer management ───────────────────────────────────
        if hand_detected:
            hand_absent_cnt = 0
            sequence.append(lm)
        else:
            hand_absent_cnt += 1
            if hand_absent_cnt >= HAND_ABSENT_RESET:
                # Reset buffer — hand gone long enough
                sequence.clear()
                vote_buffer.clear()
                hand_absent_cnt = 0
            else:
                # Pad with zeros briefly (hand momentarily obscured)
                sequence.append(np.zeros(NUM_LANDMARKS, dtype=np.float32))

        frame_count += 1

        # ── Prediction ──────────────────────────────────────────
        run_pred = (
            len(sequence) == SEQUENCE_LENGTH and
            frame_count % PREDICT_EVERY_N_FRAMES == 0
        )

        if run_pred:
            seq_arr = np.array(sequence, dtype=np.float32)  # (30, 126)
            res     = model.predict(seq_arr[np.newaxis], verbose=0)[0]

            top3_idx    = np.argsort(res)[-3:][::-1]
            top3_labels = [idx_to_label.get(i, str(i)) for i in top3_idx]
            top3_probs  = [float(res[i]) for i in top3_idx]

            pred_idx   = int(top3_idx[0])
            confidence = top3_probs[0]

            if confidence >= CONFIDENCE_THRESHOLD:
                # ── Confusion Gap Check ────────────────────────
                # If top1 and top2 are too close, it's likely a "Same Sign" confusion.
                # We skip these until the model is more certain.
                if len(top3_probs) > 1:
                    gap = top3_probs[0] - top3_probs[1]
                    if gap < 0.15:  # 15% margin
                        current_label = f"Confused... ({top3_labels[0]}?)"
                        current_conf = confidence
                        vote_buffer.clear()
                        continue

                pred_label = top3_labels[0]
                current_label = pred_label
                current_conf  = confidence

                vote_buffer.append(pred_label)

                # Commit to sentence only if VOTE_WINDOW agree
                if len(vote_buffer) == VOTE_WINDOW:
                    most_common, count = Counter(vote_buffer).most_common(1)[0]
                    if count >= VOTE_WINDOW - 1:          # ≥4/5 agree
                        if not sentence or sentence[-1] != most_common:
                            sentence.append(most_common)
                            vote_buffer.clear()           # fresh window
                        if len(sentence) > 6:
                            sentence = sentence[-6:]
            else:
                current_label = "..."
                current_conf  = confidence
                vote_buffer.clear()

        # ── Render UI ───────────────────────────────────────────
        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 55), CLR_DARK, -1)
        label_text = f"{current_label.upper()}  ({current_conf*100:.1f}%)"
        cv2.putText(frame, label_text, (12, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, CLR_YELLOW, 2, cv2.LINE_AA)

        # Hand indicator dot
        dot_col = CLR_GREEN if hand_detected else CLR_RED
        cv2.circle(frame, (w - 18, 15), 10, dot_col, -1)
        cv2.putText(frame, "hand" if hand_detected else "no hand",
                    (w - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_col, 1, cv2.LINE_AA)

        # Top-3 panel
        draw_top3(frame, top3_labels, top3_probs, w)

        # Sentence strip (bottom)
        cv2.rectangle(frame, (0, h - 50), (w, h - 8), CLR_DARK, -1)
        sentence_text = "  ".join(sentence) if sentence else "(sentence appears here)"
        cv2.putText(frame, sentence_text, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, CLR_WHITE, 2, cv2.LINE_AA)

        # Buffer fill bar
        draw_buffer_bar(frame, len(sequence), h, w)

        # Vote progress dots (top-left row)
        for vi in range(VOTE_WINDOW):
            filled = vi < len(vote_buffer)
            col    = CLR_GREEN if filled else (60, 60, 60)
            cv2.circle(frame, (10 + vi * 22, h - 65), 8, col, -1)

        # Controls hint
        cv2.putText(frame, "Q=quit  C=clear  SPACE=predict",
                    (w // 2 - 120, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREY, 1, cv2.LINE_AA)

        cv2.imshow('Sign Language Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            vote_buffer.clear()
        elif key == ord(' '):
            # Force an immediate prediction
            if len(sequence) == SEQUENCE_LENGTH:
                frame_count = 0   # next frame will trigger pred

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    run_inference()
