import os

# TensorFlow reads these flags at import time, so they must be set first.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# Load model once when server starts
if os.getenv("TF_ENABLE_XLA", "0") == "1":
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "efficientnet_model.keras")
)
model_eff = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Class labels
class_names = [
    "air_pollution",
    "garbage_dirty",
    "hygienic_environment",
    "stagnant_water"
]

# Image preprocessing
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    return img_array


def warmup_model():
    # Run one dummy inference so first user request avoids cold-start latency.
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy = tf.keras.applications.efficientnet.preprocess_input(dummy)
    model_eff.predict(dummy, verbose=0)

# Prediction function
def predict_environment(img_path):
    img_array = preprocess_image(img_path)
    predictions = model_eff.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]) * 100)

    return predicted_class, confidence
