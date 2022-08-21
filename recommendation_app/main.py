import gradio as gr
import pandas as pd
from core.data_handler.data_handler import DataHandler
from core.model import Model

PATH = "../netflix-recommendation-app/data/output/df_titles.csv"
df2 = pd.read_csv(PATH)
movie_names = df2["title"].tolist()


def gradio(movie_name, n_rec):
    if __name__ == "__main__":
        PATH = "../netflix-recommendation-app/data/output/df_titles.csv"
        features = [
            "type",
            "release_year",
            "age_certification",
            "runtime",
            "seasons",
            "imdb_score",
            "tmdb_popularity",
            "tmdb_score",
            "genres_transformed",
            "production_countries_transformed",
        ]
        df = pd.read_csv(PATH)
        df_model = df.copy()
        df_model = df_model[features]
        x = DataHandler(df_model)
        numeric_features = [
            "release_year",
            "runtime",
            "seasons",
            "imdb_score",
            "tmdb_popularity",
            "tmdb_score",
        ]
        x.normalize(numeric_features)
        categorical_features = [
            "age_certification",
            "type",
            "genres_transformed",
            "production_countries_transformed",
        ]
        x.one_hot_encode(categorical_features)
        # print(x.one_hot_encode(categorical_features))
        # print(x.df)
        mdl = Model(x.df)
        n_rec = int(n_rec)
        movie_name = str(movie_name)
        movie_id = df.index[df["title"] == movie_name].tolist()
        print(movie_id)
        recommendations = mdl.recommend(movie_id, n_rec)
        top_index = list(recommendations.index)[1:]
        print(df[["title", "description"]].loc[top_index])
        return df[["title", "description"]].loc[top_index]


app = gr.Interface(
    fn=gradio,
    inputs=[gr.Dropdown(choices=movie_names), gr.inputs.Number()],
    outputs=[gr.outputs.Dataframe()],
)
app.launch()
