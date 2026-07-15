from pathlib import Path
import json
import pandas as pd

#Resolve project folders
# preprocess.py is inside: project_root/ml/preprocess.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "saved_model"
RATINGS_FILE = DATA_DIR / "rating.csv"
MOVIES_FILE = DATA_DIR / "movie.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Check whether required files exist
if not RATINGS_FILE.exists():
    raise FileNotFoundError(
        f"ratings.csv was not found at:\n{RATINGS_FILE}"
    )

if not MOVIES_FILE.exists():
    raise FileNotFoundError(
        f"movies.csv was not found at:\n{MOVIES_FILE}"
    )

#Load MovieLens data
print("Loading MovieLens 20M dataset...")
ratings = pd.read_csv(
    RATINGS_FILE,
    usecols=["userId", "movieId", "rating", "timestamp"],
    dtype={
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
    },
    parse_dates=["timestamp"],
)
movies = pd.read_csv(
    MOVIES_FILE,
    dtype={
        "movieId": "int32",
        "title": "string",
        "genres": "string",
    },
)

print(f"Original ratings: {len(ratings):,}")
print(f"Original movies: {len(movies):,}")
print(f"Original users: {ratings['userId'].nunique():,}")

#Remove invalid or duplicate rows
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

ratings = ratings.drop_duplicates(
    subset=["userId", "movieId"],
    keep="last",
)

movies = movies.drop_duplicates(
    subset=["movieId"],
    keep="first",
)

#Convert explicit ratings into positive interactions
#Ratings of 4.0 and above are treated as liked movies.
POSITIVE_RATING_THRESHOLD = 4.0

positive_ratings = ratings[
    ratings["rating"] >= POSITIVE_RATING_THRESHOLD
].copy()

print(
    f"Positive interactions with rating >= "
    f"{POSITIVE_RATING_THRESHOLD}: "
    f"{len(positive_ratings):,}"
)

#Filter very inactive users and movies

MIN_USER_INTERACTIONS = 20
MIN_MOVIE_INTERACTIONS = 50

#We repeat filtering because removing movies can make some
#users fall below the threshold, and vice versa.
for iteration in range(10):
    previous_size = len(positive_ratings)

    user_counts = positive_ratings["userId"].value_counts()
    valid_users = user_counts[
        user_counts >= MIN_USER_INTERACTIONS
    ].index

    positive_ratings = positive_ratings[
        positive_ratings["userId"].isin(valid_users)
    ]

    movie_counts = positive_ratings["movieId"].value_counts()
    valid_movies = movie_counts[
        movie_counts >= MIN_MOVIE_INTERACTIONS
    ].index

    positive_ratings = positive_ratings[
        positive_ratings["movieId"].isin(valid_movies)
    ]

    current_size = len(positive_ratings)

    print(
        f"Filtering iteration {iteration + 1}: "
        f"{current_size:,} interactions"
    )

    if current_size == previous_size:
        break


if positive_ratings.empty:
    raise RuntimeError(
        "No ratings remained after filtering. "
        "Reduce MIN_USER_INTERACTIONS or "
        "MIN_MOVIE_INTERACTIONS."
    )

#Sort interactions
positive_ratings = positive_ratings.sort_values(
    by=["userId", "timestamp"]
).reset_index(drop=True)

#Create continuous graph indices

#MovieLens IDs are not guaranteed to be continuous.
#Graph neural networks need indices like:
#users  -> 0, 1, 2, 3, ...
#movies -> 0, 1, 2, 3, ...

unique_user_ids = sorted(
    positive_ratings["userId"].unique().tolist()
)

unique_movie_ids = sorted(
    positive_ratings["movieId"].unique().tolist()
)

user_id_to_index = {
    int(user_id): index
    for index, user_id in enumerate(unique_user_ids)
}

movie_id_to_index = {
    int(movie_id): index
    for index, movie_id in enumerate(unique_movie_ids)
}

positive_ratings["user_idx"] = (
    positive_ratings["userId"]
    .map(user_id_to_index)
    .astype("int32")
)

positive_ratings["movie_idx"] = (
    positive_ratings["movieId"]
    .map(movie_id_to_index)
    .astype("int32")
)

#Keep metadata only for movies present in the graph

filtered_movies = movies[
    movies["movieId"].isin(unique_movie_ids)
].copy()

filtered_movies["movie_idx"] = (
    filtered_movies["movieId"]
    .map(movie_id_to_index)
    .astype("int32")
)

filtered_movies = filtered_movies.sort_values(
    "movie_idx"
).reset_index(drop=True)

#Save processed datasets

ratings_output = OUTPUT_DIR / "ratings_processed.parquet"
movies_output = OUTPUT_DIR / "movies_processed.parquet"

positive_ratings.to_parquet(
    ratings_output,
    index=False,
)

filtered_movies.to_parquet(
    movies_output,
    index=False,
)

#Save ID mappings

user_map_output = OUTPUT_DIR / "user_map.json"
movie_map_output = OUTPUT_DIR / "movie_map.json"

with open(user_map_output, "w", encoding="utf-8") as file:
    json.dump(user_id_to_index, file)

with open(movie_map_output, "w", encoding="utf-8") as file:
    json.dump(movie_id_to_index, file)


# Reverse mappings are useful when converting predictions
# back into original MovieLens IDs.

index_to_user_id = {
    str(index): int(user_id)
    for user_id, index in user_id_to_index.items()
}

index_to_movie_id = {
    str(index): int(movie_id)
    for movie_id, index in movie_id_to_index.items()
}

with open(
    OUTPUT_DIR / "index_to_user.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(index_to_user_id, file)

with open(
    OUTPUT_DIR / "index_to_movie.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(index_to_movie_id, file)


#Save preprocessing information
statistics = {
    "positive_rating_threshold": POSITIVE_RATING_THRESHOLD,
    "minimum_user_interactions": MIN_USER_INTERACTIONS,
    "minimum_movie_interactions": MIN_MOVIE_INTERACTIONS,
    "number_of_users": len(user_id_to_index),
    "number_of_movies": len(movie_id_to_index),
    "number_of_positive_interactions": len(positive_ratings),
}

with open(
    OUTPUT_DIR / "preprocessing_stats.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(statistics, file, indent=4)

#Print final results

print("\nPreprocessing completed successfully.")
print("-------------------------------------")
print(f"Users: {len(user_id_to_index):,}")
print(f"Movies: {len(movie_id_to_index):,}")
print(f"Positive edges: {len(positive_ratings):,}")

print("\nCreated files:")

for file_path in [
    ratings_output,
    movies_output,
    user_map_output,
    movie_map_output,
    OUTPUT_DIR / "index_to_user.json",
    OUTPUT_DIR / "index_to_movie.json",
    OUTPUT_DIR / "preprocessing_stats.json",
]:
    print(f"- {file_path}")
