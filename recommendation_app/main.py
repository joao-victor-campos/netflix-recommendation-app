import gradio as gr
import pandas as pd
from core.data_handler.data_handler import DataHandler
from core.model import Model

PATH = "../netflix-recommendation-app/data/output/df_titles.csv"
df_recommendation = pd.read_csv(PATH)
movie_names = df_recommendation["title"].tolist()


def gradio(movie_name, n_rec):
    if __name__ == "__main__":
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
        df_model = df_recommendation.copy()
        df_model = df_model[features]
        handled_feature = DataHandler(df_model)
        numeric_features = [
            "release_year",
            "runtime",
            "seasons",
            "imdb_score",
            "tmdb_popularity",
            "tmdb_score",
        ]
        handled_feature.normalize(numeric_features)
        categorical_features = [
            "age_certification",
            "type",
            "genres_transformed",
            "production_countries_transformed",
        ]
        handled_feature.one_hot_encode(categorical_features)

        rec_model = Model(handled_feature.df)
        n_rec = int(n_rec)
        movie_name = str(movie_name)
        movie_id = df_recommendation.index[
            df_recommendation["title"] == movie_name
        ].tolist()
        recommendations = rec_model.recommend(movie_id, n_rec)
        top_index = list(recommendations.index)[1:]
        return df_recommendation[["title", "description"]].loc[top_index]


app = gr.Interface(
    title="Netflix Recommendation App",
    description="This app recommends movies based on other movies you liked. To use the app, first, select the **Movie Name** and then select **how many** recommendations you want.",
    article="""The Model was trained with [kaggle dataset](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies) using cosine similarity based on its features.  
    The complete project is in our [GitHub repository](https://github.com/joao-victor-campos/netflix-recommendation-app)""",
    allow_flagging="never",
    fn=gradio,
    inputs=[
        gr.Dropdown(choices=movie_names, label="Movie Name", elem_id=True),
        gr.inputs.Number(label="Number of Recommendations"),
    ],
    outputs=[
        gr.outputs.Dataframe(label="Recommendations", headers=["Title", "Description"])
    ],
)
app.launch()
