#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:45:50 2024

@author: gn
"""

import cv2
from fer import FER
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Streamlit UI setup
st.title("Facial Emotion Detection")
st.write("Upload an image to detect emotions")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

def detect_emotions(image):
    # Initialize the FER detector
    detector = FER(mtcnn=True)

    # Detect emotions
    result = detector.detect_emotions(image)

    # Print the results
    if result:
        for face in result:
            emotions = face["emotions"]
            reaction = max(emotions, key=emotions.get)
            reaction_map = {
                "happy": 1,
                "angry": 2,
                "disgust": 3,
                "sad": 4,
                "neutral": 5,
                "surprise": 6,
                "fear": 7
            }
            reaction_number = reaction_map.get(reaction, "Unknown reaction")
            st.markdown(f"<h3 style='font-size:24px; color:green;'>Detected reaction: {reaction} {emotions[reaction]*100:.2f}%</h3>", unsafe_allow_html=True)
            
            # Draw bounding box and emotion on the image
            (x, y, w, h) = face["box"]
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            emotion_text = f"{reaction}: {emotions[reaction]*100:.2f}%"
            cv2.putText(image, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image
    else:
        st.write("No faces detected.")
        return image

if uploaded_file is not None:
    # To read image file buffer as a PIL Image:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert RGB to BGR
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Detect emotions
    processed_image = detect_emotions(image_np)
    
    # Convert BGR to RGB for displaying with Streamlit
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Display the image with detected emotions
    st.image(processed_image_rgb, caption='Processed Image', use_column_width=True)
