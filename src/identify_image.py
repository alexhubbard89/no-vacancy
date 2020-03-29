import cv2
from skimage.metrics import structural_similarity
import pandas as pd
import numpy as np
import pickle
import psycopg2
import urllib.parse as urlparse
import os
## fast writes
from sqlalchemy import event, create_engine
import psycopg2.extras
import psycopg2.errorcodes

def open_connection():
    connection = psycopg2.connect(
        database=os.environ['RACK_DATABASE'],
        user=os.environ['RACK_USERNAME'],
        password=os.environ['RACK_PASSWORD'],
        host=os.environ['RACK_HOST'],
        port=os.environ['RACK_PORT']
        )
    return connection

import boto3
import os

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_gray(faces, image, gray):
    image_gray = None
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image_gray = gray[y:y+h, x:x+w]
    return image_gray

def get_single_face(gray):
    for i in range(10, 25):
        faces = haar_cascade_face.detectMultiScale(gray, 1.1, i)
        if len(faces) == 1:
            return faces, True
        elif len(faces) == 0:
            return [], False
    return [], False

def get_gray(faces, image, gray):
    image_gray = None
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image_gray = gray[y:y+h, x:x+w]
    return image_gray

def compare_images(hotel_image_path, directory, id_, file, return_image=False):
    ## format image to compare
    s3 = boto3.resource('s3', region_name='us-east-1')
    bucket = s3.Bucket('racksaruploads')
    bucket.download_file(hotel_image_path, 'tmp.jpg')
    image = cv2.imread('tmp.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade_face.detectMultiScale(gray, 1.1, 2)
    gray_cap = get_gray(faces, image, gray)
    H, W = gray_cap.shape
    gray_cap_area = H*W


    ## get victim/perp image
    s3 = boto3.resource('s3', region_name='us-east-1')
    bucket = s3.Bucket(directory)
    bucket.download_file('{}/{}'.format(id_, file), 'tmp.jpg')
    image = cv2.imread('tmp.jpg')

    ## format image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, found = get_single_face(gray)
    if found:
        gray_v = get_gray(faces, image, gray)

        ## prep for comparison
        H, W = gray_v.shape
        image_area = H*W


        if gray_cap_area > image_area:
            gray_v = cv2.resize(gray_v, gray_cap.shape)
        elif gray_cap_area < image_area:
            gray_cap = cv2.resize(gray_cap, gray_v.shape)

        if return_image == False:
            return structural_similarity(gray_cap, gray_v)
        else:
            return structural_similarity(gray_cap, gray_v), image
    if return_image == False:
        return None
    else:
        return None, None

def score_new_image():
    """
    This is not completed but has the framwork to score
    and to push to s3.
    """
    tested_id = list(pd.read_sql_query('select distinct input_prefix from match_results', open_connection())['input_prefix'])
    sql_ = """
    SELECT bucket AS input_bucket,
           prefix AS input_prefix,
           filename AS input_filename
    FROM suspicious_activity_reports
    WHERE bucket = 'racksaruploads'
    AND prefix not in ({})
    """.format("'" + "', '".join(tested_id) + "'")
    df_suspicious = pd.read_sql_query(sql_, open_connection())
    df_suspicious['hotel_image_path'] = df_suspicious['input_prefix'] +'/' + df_suspicious['input_filename']

    photo_compare_df = pd.read_sql_query("select * from victim_photo", open_connection())
    photo_compare_df['face_vector'] = photo_compare_df['face_vector'].apply(lambda row: pickle.loads(eval(row)))
    photo_compare_df = photo_compare_df.loc[photo_compare_df['face_vector'].notnull()]

    for i in photo_compare_df.index:
        id_ = photo_compare_df.loc[i, 'id']
        file = photo_compare_df.loc[i, 'url'].split('/')[-1]
        df_suspicious['similarity'] = df_suspicious['hotel_image_path'].apply(lambda row: compare_images(hotel_image_path=row, directory='victims', id_=id_, file=file))

        found_df = df_suspicious.loc[df_suspicious['similarity'] > .8]
