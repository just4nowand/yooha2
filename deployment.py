import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
st.title("유재석/하도영 분류기")
st.header("분류할 이미지를 업로드해주세요:")

def dog_cat_classifier(img, model):
    '''
    Teachable machine learning classifier for dog-cat classification:
    Parameters
    {
    img: Image to be classified
    model : trained model
    }
    '''
    # Load the model that was saved earlier
    model = keras.models.load_model(model)
    '''Define the array of the right shape to feed into the keras model'''
    data = np.ndarray(shape=(1, 96, 96, 3), dtype=np.float32)
    image = img
    #resizing the image
    size = (96, 96)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #convert the image into a numpy array
    image_array = np.asarray(image)
    # Image processing (normalization)
    normalized_image = (image_array.astype(np.float32) / 255)
    # Load the image into the array
    data[0] = normalized_image
    # carryout predictions
    prediction_rate = model.predict(data)
    prediction = prediction_rate.round()
    return  prediction,prediction_rate[0][0]
#prompt user for an image
uploaded_image = st.file_uploader("유재석 또는 하도영의 이미지를 선택해주세요...", type=['webp','jfif','png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("분류 중입니다. 잠시만 기다려주세요...")
    label,conf = dog_cat_classifier(image, 'yooha.h5')
    ## st.write("label:",label,"conf:",conf)
    if label == 1:
        st.write("이 사진은 ",round(conf *100,2), "% 확률로 유재석입니다.")
    else:
        st.write("이 사진은 ",round((1-conf)*100,2), "% 확률로 하도영입니다.")
