# Import all of the dependencies
import streamlit as st
import os 
# import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

from streamlit_webrtc import webrtc_streamer, RTCConfiguration
# import av
# import cv2

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
# options = os.listdir(os.path.join('..', 'data', 's1'))
# selected_video = st.selectbox('Choose video', options)

webrtc_streamer(key="key")