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
st.title("Facial Emotion Detection")  # Title of the Streamlit app

# Adding LinkedIn link and creator information
st.markdown("""
<div style="display: flex; align-items: center;">
    <span style="font-size: 18px; color: #2e8b57; font-weight: bold; margin-right: 10px;">Created by:</span>
    <a href="https://www.linkedin.com/in/gn-raavanan" target="_blank" style="text-decoration: none; display: flex; align-items: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width: 20px; height: 20px; margin-right: 5px;">
        <span style="font-size: 18px; font-weight: bold; color: #fff;">Gokul nath</span>
    </a>
</div>
""", unsafe_allow_html=True)  # Adding a LinkedIn link with an icon and creator's name

st.write("Upload an image to detect emotions")  # Instructions for the user
# File uploader to allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

def detect_emotions(image):
    # Initialize the FER (Facial Emotion Recognition) detector with MTCNN for better face detection
    detector = FER(mtcnn=True)

    # Detect emotions in the provided image
    result = detector.detect_emotions(image)

    # Check if any faces were detected
    if result:
        for face in result:
            emotions = face["emotions"]  # Extract the emotions dictionary for the detected face
            reaction = max(emotions, key=emotions.get)  # Get the emotion with the highest probability
            
            # Mapping emotions to corresponding numbers (not currently used but could be useful for future purposes)
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
            
            # Display the detected emotion and its confidence level on the Streamlit app
            st.markdown(f"<h3 style='font-size:24px; color:#2e8b57;'>Detected reaction: {reaction} {emotions[reaction]*100:.2f}%</h3>", unsafe_allow_html=True)
            
            # Draw bounding box around the detected face and annotate it with the dominant emotion
            (x, y, w, h) = face["box"]  # Get the coordinates of the bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face with a green border
            emotion_text = f"{reaction}: {emotions[reaction]*100:.2f}%"  # Prepare text for annotation
            cv2.putText(image, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Put text above the bounding box
        return image
    else:
        # If no faces are detected, display a message on the Streamlit app
        st.write("No faces detected.")
        return image

if uploaded_file is not None:
    # Read the uploaded image file buffer as a PIL Image
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert the PIL image to a NumPy array

    # Convert the image from RGB (PIL format) to BGR (OpenCV format)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Detect emotions in the image
    processed_image = detect_emotions(image_np)
    
    # Convert the processed image from BGR (OpenCV format) back to RGB for displaying with Streamlit
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Display the processed image with detected emotions and bounding boxes
    st.image(processed_image_rgb, caption='Processed Image', use_column_width=True)

