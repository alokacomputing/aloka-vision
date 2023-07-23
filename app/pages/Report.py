import streamlit as st
from Home import face_rec

# st.set_page_config(page_title="Reporting")
st.subheader("Report")

# extract data from redis
name = 'attendance:logs'
def load_logs(name, end=-1):
    data = face_rec.r.lrange(name, 0, end)
    data = [x.decode('utf-8') for x in data]
    return data

# tabs to show the information
tab1, tab2 = st.tabs(['Registered Data', 'Attendace Logs'])

with tab1:
  if st.button('Refresh Data'):
    with st.spinner("Retriving data from Redis"):
      redis_face_db = face_rec.retrive_data(name="academy:register")
      # st.dataframe(redis_face_db)
      st.dataframe(redis_face_db[['name', 'role']])

with tab2:
  if st.button('Refresh Logs'):
    st.write(load_logs(name))