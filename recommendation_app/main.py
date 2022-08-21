import pandas as pd
from core.data_handler.data_handler import DataHandler
from core.model import Model

if __name__ == "__main__":
    # constants
    # PATH = '../data/output/df_titles.csv'

    # import data
    # df = pd.read_csv(PATH)

    df3 = pd.DataFrame(
        [["c", 3, 10, "cat"], ["d", 4, 50, "dog"]],
        columns=["letter", "number", "number2", "animal"],
    )
    x = DataHandler(df3)
    x.normalize(["number", "number2"])
    print(x.one_hot_encode(["letter", "animal"]))
    print(x.df)
    # ran on a sample as an example
    mdl = Model(x.df)
    print(mdl.recommend(movie_id=0, n_rec=2))
