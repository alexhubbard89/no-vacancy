import requests
import io
from bs4 import BeautifulSoup
from unidecode import unidecode
import pandas as pd
import numpy as np

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

engine = create_engine('postgresql://{}:{}@{}/{}'.format(os.environ['RACK_USERNAME'],
                                                            os.environ['RACK_PASSWORD'],
                                                            os.environ['RACK_HOST'],
                                                            os.environ['RACK_DATABASE']
                                                           ))

"""
This file is used to see the database with missing children.
The source for this collectiong is: http://www.pollyklaas.org/missing.
For future collection the plan is to collect missing and wanted.
Websites to scrape:
- https://www.namus.gov/MissingPersons/Search#/results
- https://www.fbi.gov/wanted/fugitives
- https://www.interpol.int/en/How-we-work/Notices/View-Red-Notices#
- https://www.missingkids.org/search


To get us started with identifying missing and wanted I collected
from pollyklass.org, and seeded images of jeffrey epstein for
wanted persons.
"""

POLLY_KLASS_BASE = 'http://www.pollyklaas.org/missing/index.html?state=&year=&search_name=&page={}'
COLS = ['Abductor', 'DOB', 'Date Missing', 'Missing From',
        'CADOB', 'Age at Disappearance', 'Sex', 'Race', 'Height',
        'Weight', 'Eyes', 'Hair', 'Other', 'Circumstances']

def get_pk_urls():
    missing_children_url = []
    for i in range(1, 15):
        url = POLLY_KLASS_BASE.format(i)
        print(url)
        r = requests.get(url)
        page = (BeautifulSoup(r.content)
                .find('div', {'class': 'missing-child-resize'}))
        missing_children_url.extend([x.find('a').get('href') for x in page.findAll('tbody')])
    return missing_children_url

def get_missing_deets(url):
    print(url)
    r = requests.get(url)
    page = BeautifulSoup(r.content)

    missing_dict = {}
    for body in page.findAll('tbody'):
        if None != body.find('span').find('p'):
            if 'Missing' in body.find('span').find('p').text:
                ## prep to extract
                num = len(body.findAll('p'))
                if num > 1:
                    txt = body.findAll('p')[1].text
                    name = body.find('span').find('p').text.split('Missing Child: ')[1]
                else:
                    txt = body.text
                    name = page.find('title').text
                txt = txt.replace('\n', '')
                for i in range(10):
                    txt = txt.replace('  ', ' ').strip(' ')

                for col in COLS:
                    txt = txt.replace(col, ' :{}'.format(col))
                txt = txt.strip(' ')

                ## get info about child
                tmp_dict = {'abductor': False}
                keys = txt.split(':')[1:][::2]
                values = [unidecode(x).strip(' ') for x in txt.split(':')[1:][1::2]]
                for i in range(len(keys)):
                    tmp_dict.update({keys[i].replace(' ', '_').lower(): values[i]})

                ## update overall dict
                missing_dict.update({name: tmp_dict})
        elif 'Abductor' in body.text:
            ## clean
            txt = unidecode(body.text.replace('\n', ''))
            for i in range(10):
                txt = txt.replace('  ', ' ').strip(' ')
            for col in COLS:
                txt = txt.replace(col, ' :{}'.format(col))
            txt = txt.strip(' ')

            ## get info about abuductor
            tmp_dict = {'abductor': True}
            keys = txt.split(':')[1:][::2]
            values = [unidecode(x).strip(' ') for x in txt.split(':')[1:][1::2]]
            key_len = np.array([len(keys), len(values)]).min()
            for i in range(len(keys)):
                if keys[i] == 'Abductor':
                    name = values[i]
                else:
                    tmp_dict.update({keys[i].replace(' ', '_').lower(): values[i]})
            missing_dict.update({name: tmp_dict})


    df = pd.DataFrame.from_dict(missing_dict).transpose().reset_index().rename(columns={'index': 'name'})
    images = [x.get('src') for x in page.findAll('img', src=True) if 'missing/kids/' in x.get('src')]
    df['image'] = images[:len(df)]
    return df

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def get_gray(faces, image, gray):
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image_gray = gray[y:y+h, x:x+w]
    return image_gray

def get_single_face(gray):
    for i in range(25):
        faces = haar_cascade_face.detectMultiScale(gray, 1.1, i)
        if len(faces) == 1:
            return faces, True
        elif len(faces) == 0:
            return [], False
    return [], False

## save image vector for fast photo comparisons
def get_vector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, found = get_single_face(gray)
    if found:
        return get_gray(faces, image, gray)
    else:
        return None


if __name__ == "__main__":
    cols = ['name', 'dob', 'date_missing', 'eyes', 'hair', 'height',
            'weight', 'race', 'sex', 'missing_from', 'age_at_disappearance',
            'circumstances', 'other'
           ]

    overall_df = pd.DataFrame()
    for url in pk_missing:
        try:
            tmp = get_missing_deets(url)
            overall_df = overall_df.append(tmp).reset_index(drop=True)
        except:
            pass

    victim_df = overall_df.loc[overall_df['abductor'] == False, cols].copy()
    victim_df['dob'] = pd.to_datetime(victim_df['dob'])
    victim_df['date_missing'] = pd.to_datetime(victim_df['date_missing'])
    victim_df['weight'] = victim_df['weight'].apply(lambda row: int(row.split(' ')[0].split('-')[0]))
    victim_df['city'] = victim_df['missing_from'].apply(lambda row: row.split(', ')[0])
    victim_df['state'] = victim_df['missing_from'].apply(lambda row: row.split(', ')[1])
    victim_df = victim_df.drop(['missing_from'], 1)
    victim_df['country'] = 'USA'

    ## edge case
    victim_df.loc[23, 'age_at_disappearance'] = 1

    for col in victim_df.columns:
        victim_df.loc[victim_df[col] == '', col] = None
    victim_df.to_sql('victims', con=engine, if_exists='append', index=False)

    ## create photo database
    victim_with_id = pd.read_sql_query("select name, id from victims", open_connection())
    victim_with_id = (victim_with_id
                      .merge(overall_df[['name', 'image']], how='left', on='name')
                      .rename(columns={'image': 'url'}).drop('name', 1))


    for i in victim_with_id.index:
        url = victim_with_id.loc[i, 'url']
        file_name = url.split('/')[-1].split('.')[0]
        id_ = victim_with_id.loc[i, 'id']

        local_path = 'tmp.jpg'
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        PIL.Image.open(image_file).convert('RGB')
        image = PIL.Image.open(image_file).convert('RGB')
        with open(local_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)

        ## write to s3
        path_write = '{}/{}.jpg'.format(id_, file_name)
        bucket.upload_file(local_path, path_write)

        victim_with_id.loc[i, 'face_vector'] = get_vector(image)

    ## save photos info
    victim_with_id.to_sql('victim_photo', con=engine, if_exists='append', index=False)
