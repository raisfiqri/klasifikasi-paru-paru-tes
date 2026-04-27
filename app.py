import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_parurasio801010.h5")

# Label kelas (sesuaikan dengan urutan training kamu!)
class_names = ["covid", "lung normal", "lung opacity", "viral pneumonia"]

st.title("Prediksi Penyakit Paru-Paru (X-ray)")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.write("### Hasil Prediksi:")
    st.write(f"Kelas: **{class_names[predicted_class]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
