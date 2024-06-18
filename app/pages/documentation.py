import streamlit as st
st.set_page_config(
        page_title="TensorVision-Documentation",
        page_icon=":camera:",
        layout="wide",
)
        
st.write("Welcome to the TensorVision documentation!!")

# Title and Introduction
st.markdown(
    """
TensorVision is a Streamlit web application designed for image classification using the pre-trained MobileNetV2 deep learning model. It empowers users to effortlessly upload images and discover their most likely categories with remarkable precision.
"""
)

# Key Features
st.header("Key Features")
st.write(
    """
* **Leverage MobileNetV2:** TensorVision harnesses the power of MobileNetV2, a state-of-the-art deep learning model, to deliver accurate image classification.
* **Seamless User Experience:** The user-friendly interface allows users to upload images with ease and receive classification results instantaneously.
* **Top  Predictions:** TensorVision generates the  most probable classifications for each uploaded image, providing valuable insights.
"""
)

# How to Use
st.header("How to Use")
st.write(
    """
1. **Launch the App:** Access the TensorVision web application through the provided URL (or instructions on how to run it locally).
2. **Upload an Image:** Click the "Choose an image..." button in the sidebar and select an image file (JPG, JPEG, or PNG format) from your device.
3. **Classify Your Image:** Press the "Classify" button to initiate the image classification process.
4. **View Results:** Upon successful classification, TensorVision displays a "Classification Complete!" message. Below that, you'll see the "Classification Results:" header followed by the top predicted category for the uploaded image.
"""
)

# Technical Specifications
st.header("Technical Specifications")
st.write(
    """
* **Frontend Framework:** Streamlit
* **Deep Learning Model:** MobileNetV2
* **Supported Image Formats:** JPG, JPEG, PNG
"""
)

# Code Structure Breakdown
st.header("Code Structure Breakdown")
st.write(
    """
The TensorVision web app is built using Python and leverages the Streamlit framework for creating the user interface. Here's a breakdown of the key components:

1. **Imports:** Necessary libraries like Streamlit, TensorFlow, OpenCV, NumPy, and others are imported for functionality.
2. **Model Loading:** The pre-trained MobileNetV2 model is loaded using `mobilenet_v2.MobileNetV2(weights='imagenet')`.
3. **`classify_image` Function:** This function handles the image classification process. ... 
4. **`is_valid_image` Function:** This function verifies if the uploaded file is a valid image format (JPG, JPEG, or PNG).
5. **`main` Function:** This core function builds the Streamlit app layout. ...
"""
)

# Additional
