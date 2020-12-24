import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

## STEP 1: READ CSV FILE
df = pd.read_csv("movie_dataset.csv")
    #print(df.head())
    #print(df.columns)          #To show the features

## STEP 2: SELECT FEATURES
features = ['keywords','cast','genres','director']

## STEP 3: CREATE A COLUMN IN DF WHICH COMBINES ALL SELECTED FEATURES
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+ row["genres"]+" "+row["director"] 
    except:
        print("Error: ", row)
    
df["combined_features"] = df.apply(combine_features,axis=1)

print ("Combined Features: ", df["combined_features"].head())

## STEP 4: CREATE COUNT MATRIX FROM THIS NEW COMBINED COLUMN
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

## STEP 5: COMPUTE THE COSINE SIMILARITY BASED ON THE COUNT_MATRIX
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

## STEP 6: GET INDEX OF THIS MOVIE FROM ITS TITLE
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

## STEP 7: GET A LIST OF SIMILAR MOVIES IN DESCENDING ORDER OF SIMILARITY SCORE
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1], reverse=True)

## STEP 8: PRINT TITLES OF FIRST 50 MOVIES
i=0
for movie in sorted_similar_movies:
    print (get_title_from_index(movie[0]))
    i+=1
    if i>50:
        break