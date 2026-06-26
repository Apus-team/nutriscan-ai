import numpy as np
from PIL import Image


TARGET_SIZE = (224, 224)


def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(TARGET_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array
