from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
from joblib import load
import string
import pickle
import gensim
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

### Don't forget to fill in your API key below:
def getUserLatLon(location):
    requestString = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={'YourAPIkey'}"
    r = requests.get(requestString)
    results = r.json().get('results', False)
    return results

def haversine(lat1, lat2, lon1, lon2):
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    delta_lat = lat2_rad-lat1_rad
    delta_lon = lon2_rad-lon2_rad
    a = np.sqrt((np.sin(delta_lat/2))**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(delta_lon/2))**2 )
    d_meters = 2 * 6371000 * np.arcsin(a)
    d = d_meters/1609.344
    return d

def cleanList(string):
    str_list = string.replace("'", "")[1:-1]
    old_list = list(map(lambda x: x.strip(), str_list.split(',')))
    newList = [tag if tag != 'dogs on leash' else 'dog friendly' for tag in old_list]
    return newList

def get_filtered_df(df, timing, location):
    maxTime = int(timing)
    minTime = maxTime - 0.75
    df_dur1 = df[df['duration'] <= maxTime]
    df_dur = df_dur1[df_dur1['duration'] >= minTime]

    result = getUserLatLon(location)
    latU = result[0]['geometry']['lat']
    lonU = result[0]['geometry']['lng']
    df_dur['userDistance']= haversine(latU, df_dur['latitude'], lonU, df_dur['longitude'])
    maxUserDistance = 10
    df_filt = df_dur[df_dur['userDistance'] <= maxUserDistance]
    return (df_filt)

def get_user_tags (indices):
    userTags = list(set(np.concatenate(df_photos['tags'].values[indices])))
    return userTags

def get_avg_word_vector(taglist):
    vmat = [vectorlookup[tag] for tag in taglist]
    length = len(vmat)
    summed = np.sum(vmat, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tags'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                                generate_missing=generate_missing))
    return list(embeddings)

def get_tag_vectors(userTags, vectorlookup):
       return [vectorlookup[tag] for tag in userTags]

def recommend(df, timing, location, indices, vectorlookup):
    df_filt = get_filtered_df(df, timing, location)
    userTags = get_user_tags(indices)
    userVector = get_average_word2vec(userTags, vectorlookup)
    df_user = df_filt[['name']]
    df_user['url'] = df_filt[['url']]
    df_user['similarity'] = df_filt['embeddings'].apply(
        lambda x: cosine_similarity(userVector.reshape(1,-1), x.reshape(1,-1))[0,0])
    userRank = df_user.sort_values(by= 'similarity', ascending = False)
    return userRank

### Load and clean data:
read_df = '/Users/jacqui/Dropbox/InsightFellowship/MainProject/web_app/myAppTesting/static/data/embeddings.csv'
read_photo = '/Users/jacqui/Dropbox/InsightFellowship/MainProject/web_app/myAppTesting/static/photos/photos.csv'
vectorlookup = pickle.load( open('/Users/jacqui/Dropbox/InsightFellowship/MainProject/web_app/myAppTesting/static/data/vectorlookup.pkl', 'rb') )
df = pd.read_csv(read_df)
df_photos= pd.read_csv(read_photo)
df['embeddings'] = df['embeddings'].apply(lambda x: np.array(list(map(lambda y: float(y.strip()), np.array(x[1:-1].split(','))))))
df['tags'] = df.tags.apply(lambda x: cleanList(x))
df_photos['tags'] = df_photos.tags.apply(lambda x: cleanList(x))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods = ['GET', 'POST'])
def data():

    ### Gets the input data and saves them to these values to be used in the filter below
    location = request.args.get('location')
    timing = request.args.get('timing')
    photos = [float(request.args.get(f'photos{i}', 0)) for i in range(1,11)]
    print(np.array(photos))
    indices = np.where(np.array(photos)==1)
    print(indices)

    #indices = [6,3]
    #clicks = request.args.get('clicks')
    #print([item for item in photos])
    userRec = recommend(df, timing, location, indices, vectorlookup)

    output_tags = get_user_tags(indices)
    output_urls = userRec.sort_values('similarity', ascending=False).iloc[:3]['url'].tolist()
    output_name = userRec.sort_values('similarity', ascending=False).iloc[:3]['name'].tolist()


    return render_template('results.html', photos = photos, output_tags = output_tags, output_name = output_name, output_urls = output_urls, location= location, timing = timing)

if __name__ == "__main__":

    app.run()
