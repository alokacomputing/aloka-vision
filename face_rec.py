import numpy as np
import pandas as pd
import cv2
import redis
import os

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to Redis
hostname = os.getenv("REDIS_HOST")
port = os.getenv("REDIS_PORT")
password = os.getenv("REDIS_PASSWORD")

r = redis.Redis(host=hostname, port=port, password=password)

# configure face analysis
app = FaceAnalysis(name="buffalo_l", root="playground", providers=["CUDAExecutionProvider"])
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
        name, role, similarity = data_filter.loc[argmax][
            ["name", "role", "cosine_similarity"]
        ]
        # print('The most similar person is ', name, 'with similarity ', similarity)
    else:
        name, role, similarity = "Unknown", "Unknown", None
        # print('No person found')
    return name, role, similarity

def face_prediction(test_image, dataframe, feature_column, thresh=0.5):
    # 1. take test image and apply face detection using insightface
    results = app.get(test_image)
    test_copy = test_image.copy()

    # 2. use for loop and extract face embedding from each face detected and pass it to ml_search_algorithm function
    for i, res in enumerate(results):
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embedding = res['embedding']
        name, role, similarity = ml_search_algorithm(dataframe, feature_column, test_vector=embedding, thresh=thresh)
        # print(name, role, similarity)

    if name == 'Unknown':
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(test_copy, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(test_copy, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return test_copy
