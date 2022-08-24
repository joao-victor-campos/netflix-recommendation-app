from array import array

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def movie_similarity(self, chosen_movie: array, sim_movies: array) -> array:
        """Calculate the cosine similarity between two vectors.

        Args:
            chosen_movie (array): Array with all information about the movie
            chosen by the user.
            sim_movies (array): n dimensions array with all movies.
        Returns:
            array: Returns the cosine similarity between chosen_movie and
            sim_array.
        """
        chosen_movie = chosen_movie.reshape(1, -1)
        return cosine_similarity(chosen_movie, sim_movies, dense_output=True)

    def recommend(self, movie_id: str, n_rec: int) -> pd.DataFrame:
        """Return nlargest similarity movies based on movie_id.

        Args:
            movie_id (str): Name of the movie to be compared.
            n_rec (int): Number of movies the user wants.
        Returns:
            pd.DataFrame: Dataframe with the n_rec recommendations.
        """
        movie_info = self.df.loc[movie_id].values
        sim_array = self.movie_similarity(movie_info, self.df.values)

        sim_list = sim_array.tolist()[0]
        self.df["similarity"] = sim_list

        return self.df.nlargest(columns="similarity", n=n_rec + 1)
