"""
train.py - Enhanced Sign Language Model Training
-------------------------------------------------
Features:
  - Loads data from custom_landmarks/ (30-sign vocabulary)
  - BiLSTM + Attention architecture
  - Label smoothing (0.1)
  - Class weights for imbalanced data
  - Cosine annealing LR schedule
  - EarlyStopping with patience=20
  - Targets >= 90% validation accuracy
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input,
    Layer, Multiply, Softmax, Lambda, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    LearningRateScheduler
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy

from config import (
    CUSTOM_LANDMARKS_DIR, LABEL_MAP_PATH,
    MODEL_SAVE_PATH, TRAINING_PLOT_PATH,
    NUM_CLASSES, SEQUENCE_LENGTH, INPUT_SHAPE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)


# ─────────────────────────────────────────────
#  Attention Layer
# ─────────────────────────────────────────────
class AttentionLayer(Layer):
    """Bahdanau-style temporal attention over LSTM outputs."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_W',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_b',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, features)
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, T, 1)
        a = tf.nn.softmax(e, axis=1)                    # (batch, T, 1)
        output = tf.reduce_sum(x * a, axis=1)           # (batch, features)
        return output

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────
#  Load Dataset
# ─────────────────────────────────────────────
def load_custom_dataset():
    """
    Load all .npy landmark files from custom_landmarks/ directory.
    Returns X, y arrays and the label map.
    """
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Label map not found at {LABEL_MAP_PATH}. "
            "Run preprocess_custom.py first."
        )

    with open(LABEL_MAP_PATH, 'r') as f:
        label_to_idx = json.load(f)

    if not os.path.exists(CUSTOM_LANDMARKS_DIR):
        raise FileNotFoundError(
            f"custom_landmarks/ directory not found. "
            "Run preprocess_custom.py first."
        )

    X, y = [], []
    classes_loaded = 0

    for gloss, idx in sorted(label_to_idx.items()):
        gloss_dir = os.path.join(CUSTOM_LANDMARKS_DIR, gloss)
        if not os.path.exists(gloss_dir):
            print(f"  [WARN] No folder for class '{gloss}'")
            continue

        files = [f for f in os.listdir(gloss_dir) if f.endswith('.npy')]
        if not files:
            print(f"  [WARN] No .npy files for class '{gloss}'")
            continue

        for fname in files:
            fpath = os.path.join(gloss_dir, fname)
            try:
                landmarks = np.load(fpath).astype(np.float32)
                if landmarks.shape == (SEQUENCE_LENGTH, 126):
                    X.append(landmarks)
                    y.append(idx)
            except Exception as e:
                print(f"  [ERR] Could not load {fpath}: {e}")

        classes_loaded += 1

    print(f"Classes loaded: {classes_loaded} / {len(label_to_idx)}")
    return np.array(X), np.array(y), label_to_idx


# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────
def build_model(num_classes):
    """
    BiLSTM + Attention model for sign language recognition.
    Targets >= 90% accuracy on 30-class problem.
    """
    inp = Input(shape=INPUT_SHAPE, name='landmark_input')

    # BiLSTM stack
    x = Bidirectional(LSTM(128, return_sequences=True), name='bilstm_1')(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(256, return_sequences=True), name='bilstm_2')(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True), name='bilstm_3')(x)
    x = Dropout(0.25)(x)

    # Attention
    x = AttentionLayer(name='attention')(x)

    # Dense head
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_1')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inp, outputs=out)

    # Label-smoothed categorical cross-entropy for better generalisation
    loss = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
#  Cosine Annealing LR Schedule
# ─────────────────────────────────────────────
def cosine_lr_schedule(epoch, total_epochs=EPOCHS, min_lr=1e-6, max_lr=LEARNING_RATE):
    cosine = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    return float(min_lr + (max_lr - min_lr) * cosine)


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Sign Language Model Training")
    print("=" * 60)

    # 1. Load data
    print("\nLoading dataset from custom_landmarks/ ...")
    X, y, label_to_idx = load_custom_dataset()

    if len(X) == 0:
        print("No data found! Run preprocess_custom.py first.")
        return

    num_classes = len(label_to_idx)
    print(f"Total samples : {len(X)}")
    print(f"Classes       : {num_classes}")
    print(f"Input shape   : {X.shape}")

    # 2. Train / Val / Test split (stratified)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.11, stratify=y_tv, random_state=42)

    print(f"Train : {len(X_train)}")
    print(f"Val   : {len(X_val)}")
    print(f"Test  : {len(X_test)}")

    # 3. Class weights (only for classes present in y_train)
    present_classes = np.unique(y_train)
    cw_values = compute_class_weight('balanced', classes=present_classes, y=y_train)
    class_weights = {cls: w for cls, w in zip(present_classes, cw_values)}
    # Fill in weight=1.0 for any classes not in training set
    for c in range(num_classes):
        if c not in class_weights:
            class_weights[c] = 1.0

    # 4. One-hot encode
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes=num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

    # 5. Build model
    print("\nBuilding model...")
    model = build_model(num_classes)
    model.summary()

    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        LearningRateScheduler(cosine_lr_schedule, verbose=0),
    ]

    # 7. Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # 8. Plot
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.axhline(0.9, color='red', linestyle='--', label='90% target')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH)
    print(f"\nTraining plots saved to: {TRAINING_PLOT_PATH}")

    # 9. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)
    y_true_cls = y_test

    test_acc = accuracy_score(y_true_cls, y_pred_cls)
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

    if test_acc >= 0.90:
        print("  TARGET ACHIEVED: >= 90% accuracy!")
    else:
        print(f"  Note: {(0.90 - test_acc)*100:.1f}% below target. "
              "Consider adding more data or running more epochs.")

    # 10. Per-class accuracy (only show classes that exist in test set)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    present_labels = sorted(list(set(y_true_cls.tolist()) | set(y_pred_cls.tolist())))
    target_names = [idx_to_label.get(i, str(i)) for i in present_labels]
    print("\nPer-class accuracy on test set:")
    from sklearn.metrics import classification_report
    report = classification_report(y_true_cls, y_pred_cls,
                                   labels=present_labels,
                                   target_names=target_names,
                                   zero_division=0)
    print(report)

    # 11. Confusion matrix (only for present classes, saved to file)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=present_labels)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = MODEL_SAVE_PATH.replace('.h5', '_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    train()
