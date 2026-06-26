import os
import numpy as np
import tensorflow as tf
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "nutriscan_model.keras")


def model_exists():
    return os.path.exists(MODEL_PATH)


@st.cache_resource
def load_model():
    if not model_exists():
        raise FileNotFoundError(
            "Modelo no encontrado. Entrena el modelo primero "
            "usando el notebook de entrenamiento."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, preprocessed_img):
    preds = model.predict(preprocessed_img, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    return class_idx, confidence
