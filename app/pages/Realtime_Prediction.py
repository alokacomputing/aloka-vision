import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

# st.set_page_config(page_title="Prediction")
st.subheader("Realtime Attendance System")

with st.spinner("Retriving data from Redis"):
    redis_face_db = face_rec.retrive_data(name="academy:register")
#     st.dataframe(redis_face_db)

st.success("Data retrived successfully")

waitTime = 30 # seconds
setTime = time.time()
realtimepred = face_rec.RealTimePred() #realtime predict class

def video_frame_callback(frame):
  global setTime

  img = frame.to_ndarray(format="bgr24") # 3 dimensional array

  pred_img = realtimepred.face_prediction(img, redis_face_db, feature_column="facial_features", thresh=0.5)
  timenow = time.time()
  difftime = timenow - setTime

  if difftime >= waitTime:
    realtimepred.saveLogs_redis()
    setTime = time.time() # reset the time

    print('saved to redis')

  return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)