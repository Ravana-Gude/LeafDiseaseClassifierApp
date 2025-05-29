import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
import joblib
import os
from PIL import Image

# Load models
@st.cache_resource
def load_cnn_model():
    return load_model("cnn_model.keras")

@st.cache_resource
def load_svm_model():
    return joblib.load("svm_model.pkl")

cnn_model = load_cnn_model()
svm_model = load_svm_model()

# Feature extractor
def build_feature_model(cnn_model):
    input_tensor = Input(shape=(128, 128, 3))
    x = input_tensor
    for layer in cnn_model.layers[:-1]:
        x = layer(x)
    return Model(inputs=input_tensor, outputs=x)

feature_model = build_feature_model(cnn_model)

# Class labels
data_dir = r"C:\Users\ravan\Downloads\archive (2)\PlantVillage\PlantVillage"
class_names = sorted(os.listdir(data_dir))  # Ensure sorted class names

# Streamlit UI
st.title("🌿 Leaf Disease Classification")
st.markdown("Upload an image of a plant leaf to classify disease using CNN + SVM.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features
    features = feature_model.predict(img_array)

    # Predict using SVM
    prediction = svm_model.predict(features)[0]

    # Show final result only
    st.subheader("Predicted Disease:")
    st.success(f"**{class_names[prediction]}**")
