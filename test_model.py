"""
Simple test script to verify the model loads and works
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_SAVE_PATH, LABEL_MAP_PATH, SEQUENCE_LENGTH, NUM_LANDMARKS

print("=" * 60)
print("SIGN LANGUAGE RECOGNITION - MODEL TEST")
print("=" * 60)

# Check if model exists
if not os.path.exists(MODEL_SAVE_PATH):
    print(f"❌ Model not found at: {MODEL_SAVE_PATH}")
    sys.exit(1)

print(f"✅ Model found: {MODEL_SAVE_PATH}")

# Check if label map exists
if not os.path.exists(LABEL_MAP_PATH):
    print(f"❌ Label map not found at: {LABEL_MAP_PATH}")
    sys.exit(1)

print(f"✅ Label map found: {LABEL_MAP_PATH}")

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

print(f"\n📋 Vocabulary ({len(label_map)} signs):")
for i, label in enumerate(sorted(label_map.keys()), 1):
    print(f"   {i:2d}. {label}")

print(f"\n⚙️  Model Configuration:")
print(f"   - Sequence Length: {SEQUENCE_LENGTH} frames")
print(f"   - Landmarks per frame: {NUM_LANDMARKS}")
print(f"   - Input shape: ({SEQUENCE_LENGTH}, {NUM_LANDMARKS})")

# Try to load model (basic import check)
try:
    print(f"\n📦 Loading TensorFlow...")
    import tensorflow as tf
    print(f"   ✅ TensorFlow version: {tf.__version__}")
    
    print(f"\n📦 Loading model...")
    from tensorflow.keras.models import load_model
    from train import AttentionLayer
    
    model = load_model(MODEL_SAVE_PATH, custom_objects={'AttentionLayer': AttentionLayer})
    print(f"   ✅ Model loaded successfully!")
    
    print(f"\n📊 Model Summary:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Parameters: {model.count_params():,}")
    
    # Test with dummy data
    print(f"\n🧪 Testing with random input...")
    dummy_input = np.random.randn(1, SEQUENCE_LENGTH, NUM_LANDMARKS).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    idx_to_label = {v: k for k, v in label_map.items()}
    
    print(f"   Top 3 predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        conf = predictions[0][idx]
        label = idx_to_label.get(idx, "Unknown")
        print(f"   {i}. {label}: {conf*100:.2f}%")
    
    print(f"\n✅ MODEL TEST PASSED!")
    print(f"\nTo run real-time recognition, use: python predict.py")
    print(f"Controls: SPACE=predict, C=clear, Q=quit")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
