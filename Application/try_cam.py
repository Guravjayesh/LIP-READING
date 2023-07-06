# Import all the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Set the page layout
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

# Main app title
st.title('LipNet Full Stack App')

# Create two columns for the app layout
col1, col2 = st.columns(2)

# Column 1: Webcam streaming
with col1:
    cam = webrtc_streamer(key="key")

# Column 2: Machine learning model predictions
with col2:
    st.info('This is all the machine learning model sees when making a prediction')

    # Load video data from the webcam stream
    video, annotations = load_data(tf.convert_to_tensor(cam))
    # imageio.mimsave('animation.gif', video, fps=10)
    # st.image('animation.gif', width=400)

    st.info('This is the output of the machine learning model as tokens')

    # Load the pre-trained model
    model = load_model()

    # Make predictions using the loaded model
    yhat = model.predict(tf.expand_dims(video, axis=0))

    # Perform CTC decoding to obtain the predicted tokens
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    st.text(decoder)

    st.info('Decode the raw tokens into words')

    # Convert the predicted tokens to text
    if decoder:
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
