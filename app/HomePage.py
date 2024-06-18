import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications import mobilenet_v2, imagenet_utils


# Load the pre-trained MobileNetV2 model
try:
  # Load the pre-trained MobileNetV2 model
  model = mobilenet_v2.MobileNetV2(weights='imagenet')
except Exception as e:
  st.error(f"Error loading MobileNetV2 model: {e}")
  raise  # Re-raise the error to stop execution 


@st.cache_data
def classify_image(image_file):
    if not image_file:
        return []
    
    img_array = np.frombuffer(image_file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet.preprocess_input(img)

    predictions = model.predict(img)
    results = imagenet_utils.decode_predictions(predictions, top=3)  # Show top 3 predictions
    return results


import io
from PIL import Image

def is_valid_image(image_bytes):
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        return True
    except (IOError, SyntaxError):
        return False

def main():
    
    """Builds the Streamlit app for image classification."""
    
    

    # Set page title, icon, and layout (valid arguments)
    st.set_page_config(
        page_title="TensorVision",
        page_icon=":camera:",
        layout="wide",
        
    )
    gif_animation = 'animation/Animation - 1718720528123.gif'

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
             "<h1 style='animation: fadein 1s ease-in;'>TensorVision</h1>",
            unsafe_allow_html=True,
        )
        description='''Leverage the power of MobileNetV2, a state-of-the-art deep learning model, to effortlessly classify your images.
                       Simply upload a picture, and TensorVision will reveal its most likely category with remarkable precision. '''
        st.markdown(
            description,
            unsafe_allow_html=True,    
       )
      
    with col2:
      st.image(gif_animation)


    # Sidebar for image upload and classification button
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        classify_button = st.button("Classify")
        

    
    if uploaded_file is not None:

        st.image(uploaded_file)
        if classify_button:
            with st.spinner("Classifying..."):
                results = classify_image(uploaded_file)
            st.success("Classification Complete!")
            st.header("Classification Results:")
            max_index = max(range(len(results[0])), key=lambda i: results[0][i][2])
            class_name = results[0][max_index][1]
            class_name=class_name.replace("_"," ")
            st.write(f"The object in the image provided is- {class_name}")
           
        
if __name__ == '__main__':
    main()