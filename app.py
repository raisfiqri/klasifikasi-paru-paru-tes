import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

st.title("Prediksi Model H5")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing (sesuaikan dengan model kamu!)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)

    st.write("Hasil Prediksi:", prediction)
    st.write("Confidence:", np.max(prediction))