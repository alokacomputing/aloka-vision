import streamlit as st

st.set_page_config(
    page_title="Kodiklatal Attendance System", page_icon=":smiley:", layout="wide"
)

st.header("Kodiklatal Attendance System")

with st.spinner("Loading model and connecting to redis"):
  import face_rec

st.success('Model loaded successfully')
st.success('Connected to Redis')
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """

# st.markdown(hide_st_style, unsafe_allow_html=True)
