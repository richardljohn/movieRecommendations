import pandas as pd

movies = pd.read_csv("ml-25m/movies.csv")
# movies

import re

def clean_title(title): 
    return re.sub("[^a-zA-z0-9 ]", "", title)

def title_splitter(title):
    return title[:-5]

def year_splitter(title):
    return title[-4:]

movies["clean_title"] = movies["title"].apply(clean_title)
movies["movie_title"] = movies["clean_title"].apply(title_splitter)
movies["year_released"] = movies["clean_title"].apply(year_splitter)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

def search(title):
    title = clean_title(title)
    qry_vector = vectorizer.transform([title])
    similarity = cosine_similarity(qry_vector, tfidf).flatten()
#     return similarity
    found = np.argpartition(similarity, -5)[-5:]
    found_movies = movies.iloc[found][::-1]
    return found_movies


import ipywidgets as widgets 
from IPython.display import display

movie_input = widgets.Text(
    value = "Jurassic Park",
    description = "Movie Title:",
    disabled=False
)

movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names="value")

display(movie_input, movie_list)

ratings = pd.read_csv("ml-25m/ratings.csv")

movie_id = 1

similar_users = ratings[(ratings["movieId"] == movie_id & (ratings["rating"] > 4))]["userId"].unique()
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
similar_user_recs_percentage = similar_user_recs.value_counts() / len(similar_users)
similar_user_recs_top_percentage = similar_user_recs_percentage[similar_user_recs_percentage > .101]

all_users = ratings[(ratings["movieId"].isin(similar_user_recs_top_percentage.index)) & (ratings["rating"] > 4)]
all_users_recs_percentages = all_users["movieId"].value_counts() / len(all_users["userId"].unique())


rec_percentages = pd.concat([similar_user_recs_top_percentage, all_users_recs_percentages], axis = 1)
rec_percentages.columns = ["similar", "all"]
rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
rec_percentages = rec_percentages.sort_values("score", ascending=False)


top_ten_recommendations = rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id & (ratings["rating"] > 4))]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs_percentage = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs_top_percentage = similar_user_recs_percentage[similar_user_recs_percentage > .101]

    all_users = ratings[(ratings["movieId"].isin(similar_user_recs_top_percentage.index)) & (ratings["rating"] > 4)]
    all_users_recs_percentages = all_users["movieId"].value_counts() / len(all_users["userId"].unique())


    rec_percentages = pd.concat([similar_user_recs_top_percentage, all_users_recs_percentages], axis = 1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)


    top_ten_recommendations = rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")
    return top_ten_recommendations