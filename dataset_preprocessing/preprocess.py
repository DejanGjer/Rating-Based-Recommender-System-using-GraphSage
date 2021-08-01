import re
import pandas as pd

dataset_dir = "../datasets/ml-25m"

movies_df = pd.read_csv(dataset_dir + "/movies.csv")
movies_df = movies_df.set_index("movieId")
ratings_df = pd.read_csv(dataset_dir + "/ratings.csv")
tags_df = pd.read_csv(dataset_dir + "/tags.csv")

tag_counts = tags_df["tag"].value_counts().rename_axis('tag').reset_index(name='counts')
tag_counts["tag"] = tag_counts["tag"].map(lambda t: t.lower())
tagovi = tag_counts.groupby("tag").counts.agg([sum])
tag_counts = tag_counts[tag_counts["counts"] >= 20]


