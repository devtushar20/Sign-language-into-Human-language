import os

# Base directory for the dataset
BASE_DIR = r"c:\Users\Aakash\Desktop\devtushar\Siign lang into human lang\new pro\wlasl-complete"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

# Project directories
PROJECT_DIR = r"c:\Users\Aakash\Desktop\devtushar\Siign lang into human lang\new pro\sign-language-project"
LANDMARKS_DIR = os.path.join(PROJECT_DIR, "landmarks")
CUSTOM_LANDMARKS_DIR = os.path.join(PROJECT_DIR, "custom_landmarks")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
JSON_PATH = os.path.join(BASE_DIR, "nslt_100.json")
WLASL_JSON_PATH = os.path.join(BASE_DIR, "WLASL_v0.3.json")
CUSTOM_SUBSET_JSON = os.path.join(PROJECT_DIR, "custom_subset.json")

# Ensure directories exist
os.makedirs(LANDMARKS_DIR, exist_ok=True)
os.makedirs(CUSTOM_LANDMARKS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 30 Common Signs (focused vocabulary) ---
CUSTOM_SIGNS = [
    'hello', 'goodbye', 'yes', 'no', 'please', 'sorry', 'help',
    'good', 'bad', 'me', 'you', 'want', 'drink', 'eat', 'like',
    'school', 'what', 'how', 'where', 'name', 'work', 'go',
    'come', 'stop', 'wait', 'more', 'home', 'friend', 'thank you', 'love'
]

# Training Parameters
NUM_CLASSES = len(CUSTOM_SIGNS)   # 30
SEQUENCE_LENGTH = 30              # frames per video
# mediapipe hand: 21 points x 3 (x,y,z) = 63 per hand. Both hands = 126
NUM_LANDMARKS = 126
INPUT_SHAPE = (SEQUENCE_LENGTH, NUM_LANDMARKS)

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001

# Data Augmentation
DATA_AUGMENTATION = True
AUGMENTATIONS_PER_SAMPLE = 14    # generate 14 extra copies per real sample (361 x 14 = ~5054)
MIN_SAMPLES_PER_CLASS = 150      # target before augmentation (150 x 30 = 4500)

# Model & output paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "sign_lstm_model.h5")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
TRAINING_PLOT_PATH = os.path.join(MODELS_DIR, "training_plot.png")
