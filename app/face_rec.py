import numpy as np
import pandas as pd
import cv2
import redis
import os
from dotenv import load_dotenv

load_dotenv()

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
# time
import time
from datetime import datetime

# Connect to Redis
hostname = os.getenv("REDIS_HOST")
port = os.getenv("REDIS_PORT")
password = os.getenv("REDIS_PASSWORD")

r = redis.Redis(host=hostname, port=port, password=password)

# retrive data from Redis
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode('utf-8'), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['role_name', 'facial_features']
    retrive_df[['role', 'name']] = retrive_df['role_name'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['name', 'role', 'facial_features']]

# configure face analysis
app = FaceAnalysis(name="buffalo_sc", root="insightface_model", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, thresh=0.5):
    """
    Cosine similarity search algorithm
    """
    # 1. take dataframe (collection of all data)
    dataframe = dataframe.copy()

    # 2. index face embedding from dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    X = np.array(X_list)

    # 3. Calculate cosine similarity between face embedding from dataframe and face embedding from test image
    similarity = pairwise.cosine_similarity(X, test_vector.reshape(1, -1))
    similarity_arr = np.array(similarity).flatten()
    dataframe["cosine_similarity"] = similarity_arr

    # 4. filter the dataframe based on cosine similarity
    data_filter = dataframe.query("cosine_similarity >= @thresh")
    if len(data_filter) > 0:
        # 5. get the most similar person from filtered dataframe
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter["cosine_similarity"].argmax()
        name, role = data_filter.loc[argmax][
            ["name", "role"]
        ]
        # print('The most similar person is ', name, 'with similarity ', similarity)
    else:
        name, role,  = "Unknown", "Unknown"
        # print('No person found')
    return name, role

# Realtime Prediction
# save logs for every 1 mins
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def saveLogs_redis(self):
        # create logs dataframe
        logs_df = pd.DataFrame(self.logs)

        # drop the duplicate
        logs_df.drop_duplicates('name', inplace=True)

        # push to redis
        # encode the data
        name_list = logs_df['name'].tolist()
        role_list = logs_df['role'].tolist()
        ctime_list = logs_df['current_time'].tolist()
        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = role + '@' + name + '@' + ctime
                encoded_data.append(concat_string)

        if len(encoded_data):
            r.lpush('attendance:logs', *encoded_data)
        
        # reset the dict
        self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column, thresh=0.5):
        name='Unknown'
        y1, y2, x1, x2 = 0, 0, 0, 0
        # 0. find the time
        current_time = str(datetime.now())

        # 1. take test image and apply face detection using insightface
        results = app.get(test_image)
        test_copy = test_image.copy()

        # 2. use for loop and extract face embedding from each face detected and pass it to ml_search_algorithm function
        for i, res in enumerate(results):
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embedding = res['embedding']
            name, role = ml_search_algorithm(dataframe, feature_column, test_vector=embedding, thresh=thresh)

        if name == 'Unknown':
            color =(0,0,255) # bgr
        else:
            color = (0,255,0)

        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

        cv2.putText(test_copy,name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
        cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)

        # save logs
        self.logs['name'].append(name)
        self.logs['role'].append(role)
        self.logs['current_time'].append(current_time)
        
        return test_copy

##### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    def get_embedding(self, frame):
        y1, y2, x1, x2 = 0, 0, 0, 0
        # get result from face analysis
        results = app.get(frame, max_num=1) # max_num: maximum number of faces to detect
        embeddings= None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 2: thickness
            cv2.putText(frame, 'samples ='+' '+str(self.sample), (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            # get facial features
            embeddings = res['embedding']
        return frame, embeddings

    def save_data_in_redis_db(self, name, role):
        if name is not None and role is not None:
            if name.strip() != '':
                key = role + '@' + name
            else:
                return False
        else:
            return 'Name and Role cannot be empty'

        if 'face_embedding.txt' not in os.listdir():
            return False

        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # save to redis
        r.hset('academy:register', key=key, value=x_mean_bytes)

        os.remove('face_embedding.txt')

        self.reset()
        return True

