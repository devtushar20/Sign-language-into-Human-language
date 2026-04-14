# Sign Language Recognition - Real-time Gesture to Human Language Translator

A deep learning project that recognizes sign language gestures in real-time using a webcam and translates them into human language text. Built with MediaPipe for hand landmark detection and a BiLSTM+Attention neural network for classification.

## 🎯 Features

- **Real-time Recognition**: Webcam-based sign language detection with 30-frame buffering
- **High Accuracy**: BiLSTM with temporal attention mechanism for sequential gesture recognition
- **30-Sign Vocabulary**: Recognizes common signs (hello, goodbye, yes, no, thank you, etc.)
- **Normalized Landmarks**: Hand landmarks normalized by wrist position for scale/position invariance
- **Smooth Predictions**: 5-vote majority smoothing for stable, confident predictions
- **Interactive UI**: Live visualization with top-3 predictions and confidence bars

## 📋 Vocabulary (30 Signs)

```
hello, goodbye, yes, no, please, sorry, help, good, bad, me, you, want, 
drink, eat, like, school, what, how, where, name, work, go, come, stop, 
wait, more, home, friend, thank you, love
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Webcam (for real-time prediction)
- 2GB+ free disk space (for model and dependencies)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/devtushar20/Sign-language-into-Human-language.git
cd sign-language-project
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv_clean
# On Windows:
.\venv_clean\Scripts\activate
# On macOS/Linux:
source venv_clean/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Real-time Sign Recognition

```bash
python predict.py
```

**Controls:**
- **SPACE** - Manually trigger prediction
- **C** - Clear current sentence
- **Q** - Quit application

**Screenshot:**
```
┌─────────────────────────────────┐
│  Sign Language Recognition      │
│                                 │
│  Predicted Sign: HELLO          │
│  Confidence: 87.3%              │
│                                 │
│  Top-3 Predictions:             │
│  1. hello     ████████░  87%    │
│  2. goodbye   ███░      30%     │
│  3. me        ██░       15%     │
│                                 │
│  Current Sentence:              │
│  "HELLO FRIEND HOW ARE YOU"     │
│                                 │
│  Buffer: 30/30 [████████████]   │
└─────────────────────────────────┘
```

### Train Custom Model

```bash
python train.py
```

This loads data from `custom_landmarks/` directory and trains the BiLSTM+Attention model.

## 📁 Project Structure

```
sign-language-project/
├── predict.py                 # Real-time recognition (webcam)
├── train.py                   # Model training script
├── utils.py                   # Landmark extraction utilities
├── config.py                  # Configuration (paths, hyperparameters)
├── preprocess_custom.py       # Preprocess custom video data
├── requirements.txt           # Python dependencies
│
├── models/
│   ├── sign_lstm_model.h5     # Pre-trained model
│   ├── label_map.json         # Sign vocabulary mapping
│   └── training_plot.png       # Training history plot
│
├── custom_landmarks/          # Training data (normalized landmarks)
│   ├── hello/
│   ├── goodbye/
│   ├── yes/
│   └── ... (other signs)
│
└── landmarks/                 # Raw video landmarks (for reference)
```

## 🧠 Model Architecture

### BiLSTM + Attention

```
Input (30 frames, 126 landmarks per frame)
    ↓
Bidirectional LSTM (256 units)
    ↓
Dropout (0.5)
    ↓
Bahdanau Attention Layer (temporal attention)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (30 units, Softmax)
    ↓
Output (30 sign classes)
```

**Key Features:**
- Bidirectional LSTM captures temporal dependencies forward and backward
- Attention mechanism identifies key frames within the 30-frame window
- Label smoothing (0.1) for better generalization
- Class weights for imbalanced data

## 📊 Training Details

| Hyperparameter | Value |
|---|---|
| Sequence Length | 30 frames |
| Landmarks | 126 (21 × 3 per hand) |
| Batch Size | 32 |
| Epochs | 200 |
| Learning Rate | 0.001 (with cosine annealing) |
| Loss | Categorical Crossentropy |
| Optimizer | Adam |
| Early Stopping | patience=20 |
| Target Accuracy | ≥90% on validation |

## 🎬 Data Collection & Preprocessing

### Landmark Extraction

MediaPipe Hands detects 21 keypoints per hand:
- 1 wrist, 4 fingers × 4 joints, 1 palm center

**Normalization:**
```python
# Normalize relative to wrist position
normalized = (landmarks - wrist_position) / wrist_to_middle_mcp_distance
```

This makes the model invariant to hand position and size.

### Data Augmentation

- 14 augmentations per sample
- Random rotations (±15°)
- Random scaling (0.8-1.2×)
- Random translations (±10%)
- Target: 150+ samples per sign before augmentation

## ⚙️ Configuration

Edit `config.py` to customize:

```python
BASE_DIR = r"path/to/wlasl-complete"  # Dataset directory
PROJECT_DIR = r"path/to/sign-language-project"

CUSTOM_SIGNS = [...]  # List of sign labels
NUM_CLASSES = 30      # Number of signs
SEQUENCE_LENGTH = 30  # Frames per gesture
BATCH_SIZE = 32
EPOCHS = 200
```

## 🔧 Dependencies

```
mediapipe==0.10.9
opencv-python==4.13.0
tensorflow>=2.13
numpy<2.0
scikit-learn
seaborn
matplotlib
tqdm
```

See `requirements.txt` for complete list.

## 📈 Performance

- **Validation Accuracy**: 91.2%
- **Inference Speed**: ~30ms per frame (CPU)
- **Latency**: ~300ms (10 frames at 30fps)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

### To Add New Signs:
1. Create a folder in `custom_landmarks/sign_name/`
2. Add `.npy` files with normalized landmarks
3. Run `train.py` to retrain the model

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Tushar Dev**  
Email: devtushar20@gmail.com  
GitHub: [@devtushar20](https://github.com/devtushar20)

## 🙏 Acknowledgments

- MediaPipe for hand detection
- TensorFlow/Keras for neural networks
- WLASL dataset inspiration
- OpenCV for computer vision utilities

## 📞 Support

For issues, questions, or suggestions:
1. Open an issue on GitHub
2. Email: devtushar20@gmail.com
3. Check existing issues and discussions

---

**Last Updated**: April 14, 2026  
**Version**: 1.0.0
