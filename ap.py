import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("model (1).h5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'Clothes_Waste',
            1: 'E_Waste',
            2: 'Glass_Waste',
            3: 'Metal_Waste',
            4: 'Organic_Waste',
            5: 'Paper_Waste',
            6:  'Plastic_Waste'}
        


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    #resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Classify")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Classified Label for the image is {}".format(map_dict [prediction]))