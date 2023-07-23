import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

# st.set_page_config(page_title="Registration Form")
st.subheader("Registration Form")

## Init registration form
registration_form = face_rec.RegistrationForm()

# 1. collect person information
name = st.text_input("Name", placeholder="Enter your name")
role = st.selectbox("Select Role", ["Student", "Teacher"])


# 2. collect facial embedding
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")  # 3 dimensional array
    reg_img, embedding = registration_form.get_embedding(img)

    if embedding is not None:
        with open("face_embedding.txt", mode="ab") as f:
            np.savetxt(f, embedding)

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")


webrtc_streamer(
    key="registration",
    video_frame_callback=video_callback_func,
    rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)


if st.button("Submit"):
    return_val = registration_form.save_data_in_redis_db(name, role)
    if return_val == True:
        st.success("Data saved successfully")
    else:
        st.error("Something went wrong")
