from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn.functional as F

from model import GraphSAGERecommender

#File paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"

MODEL_FILE = SAVED_MODEL_DIR / "graphsage_model.pt"
GRAPH_FILE = SAVED_MODEL_DIR / "movie_graph.pt"
MOVIES_FILE = SAVED_MODEL_DIR / "movies_processed.parquet"
MOVIE_MAP_FILE = SAVED_MODEL_DIR / "movie_map.json"

USER_EMBEDDINGS_FILE = (
    SAVED_MODEL_DIR / "final_user_embeddings.pt"
)

MOVIE_EMBEDDINGS_FILE = (
    SAVED_MODEL_DIR / "final_movie_embeddings.pt"
)


for required_file in [
    MODEL_FILE,
    GRAPH_FILE,
    MOVIES_FILE,
    MOVIE_MAP_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{required_file}"
        )

#Select device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

#Load graph and metadata

print("\nLoading complete graph...")

graph = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

graph.validate(raise_on_error=True)

print("Complete graph loaded.")


print("\nLoading movie metadata...")

movies = pd.read_parquet(MOVIES_FILE)

with open(
    MOVIE_MAP_FILE,
    "r",
    encoding="utf-8",
) as file:
    raw_movie_map = json.load(file)

movie_id_to_index = {
    int(movie_id): int(movie_index)
    for movie_id, movie_index in raw_movie_map.items()
}

index_to_movie_id = {
    movie_index: movie_id
    for movie_id, movie_index
    in movie_id_to_index.items()
}

print(f"Movies available: {len(movie_id_to_index):,}")

#Load checkpoint

print("\nLoading trained checkpoint...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False,
)

print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Training loss: "
    f"{checkpoint.get('training_loss', 'unknown')}"
)

#Recreate model architecture

model = GraphSAGERecommender(
    num_users=int(checkpoint["num_users"]),
    num_movies=int(checkpoint["num_movies"]),
    metadata=graph.metadata(),
    embedding_dim=int(checkpoint["embedding_dim"]),
    hidden_channels=int(checkpoint["hidden_channels"]),
    output_channels=int(checkpoint["output_channels"]),
    dropout=float(checkpoint["dropout"]),
)

model = model.to(device)


#Initialize lazy GraphSAGE layers

print("\nInitializing model architecture...")

forward_edge_type = ("user", "likes", "movie")

small_edge_count = min(
    10_000,
    graph[forward_edge_type].edge_index.size(1),
)

small_forward_edges = (
    graph[forward_edge_type]
    .edge_index[:, :small_edge_count]
    .to(device)
)

small_edge_index_dict = {
    ("user", "likes", "movie"): small_forward_edges,
    ("movie", "liked_by", "user"): (
        small_forward_edges.flip(0)
    ),
}

with torch.no_grad():
    temporary_x_dict = {
        "user": model.user_embedding.weight,
        "movie": model.movie_embedding.weight,
    }

    _ = model.encoder(
        temporary_x_dict,
        small_edge_index_dict,
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Trained model loaded successfully.")

#Generate full GraphSAGE embeddings

def generate_final_embeddings():
    """
    Run GraphSAGE message passing on the complete graph.

    This may take several minutes on CPU.
    """

    print(
        "\nGenerating final GraphSAGE embeddings "
        "for all users and movies..."
    )

    graph_on_device = graph.to(device)

    with torch.no_grad():
        x_dict = {
            "user": model.user_embedding.weight,
            "movie": model.movie_embedding.weight,
        }

        z_dict = model.encoder(
            x_dict,
            graph_on_device.edge_index_dict,
        )

        user_embeddings = (
            z_dict["user"]
            .detach()
            .cpu()
        )

        movie_embeddings = (
            z_dict["movie"]
            .detach()
            .cpu()
        )

    torch.save(
        user_embeddings,
        USER_EMBEDDINGS_FILE,
    )

    torch.save(
        movie_embeddings,
        MOVIE_EMBEDDINGS_FILE,
    )

    print("Final embeddings generated.")
    print(
        f"User embeddings shape: "
        f"{tuple(user_embeddings.shape)}"
    )
    print(
        f"Movie embeddings shape: "
        f"{tuple(movie_embeddings.shape)}"
    )

    return user_embeddings, movie_embeddings


#Load saved embeddings or generate them

if (
    USER_EMBEDDINGS_FILE.exists()
    and MOVIE_EMBEDDINGS_FILE.exists()
):
    print("\nLoading previously generated embeddings...")

    user_embeddings = torch.load(
        USER_EMBEDDINGS_FILE,
        map_location="cpu",
        weights_only=True,
    )

    movie_embeddings = torch.load(
        MOVIE_EMBEDDINGS_FILE,
        map_location="cpu",
        weights_only=True,
    )

    print("Saved embeddings loaded.")

else:
    user_embeddings, movie_embeddings = (
        generate_final_embeddings()
    )


#Normalize movie embeddings

normalized_movie_embeddings = F.normalize(
    movie_embeddings,
    p=2,
    dim=1,
)

#Search for movie

def search_movies(
    query: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Search movie titles using case-insensitive matching.
    """

    query = query.strip()

    if not query:
        return pd.DataFrame()

    result = movies[
        movies["title"].str.contains(
            query,
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    return result[
        [
            "movieId",
            "movie_idx",
            "title",
            "genres",
        ]
    ].head(limit)


#Create a new-user profile

def create_user_profile(
    liked_movie_ids: list[int],
) -> torch.Tensor:
    """
    A new application user does not exist in MovieLens.

    We represent the new user by averaging the final
    GraphSAGE embeddings of movies they liked.
    """

    liked_indices = []

    for movie_id in liked_movie_ids:
        if movie_id in movie_id_to_index:
            liked_indices.append(
                movie_id_to_index[movie_id]
            )
        else:
            print(
                f"Warning: movieId {movie_id} "
                f"is not present in the graph."
            )

    if not liked_indices:
        raise ValueError(
            "None of the provided movies exist "
            "inside the processed graph."
        )

    liked_tensor = torch.tensor(
        liked_indices,
        dtype=torch.long,
    )

    selected_movie_embeddings = (
        normalized_movie_embeddings[liked_tensor]
    )

    user_profile = selected_movie_embeddings.mean(
        dim=0
    )

    user_profile = F.normalize(
        user_profile.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)

    return user_profile


#Recommend movies for a new user

def recommend_from_liked_movies(
    liked_movie_ids: list[int],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Generate recommendations based on movies selected
    by a new application user.
    """

    user_profile = create_user_profile(
        liked_movie_ids
    )

    scores = torch.matmul(
        normalized_movie_embeddings,
        user_profile,
    )

    #Do not recommend movies the user already selected.
    for movie_id in liked_movie_ids:
        movie_index = movie_id_to_index.get(movie_id)

        if movie_index is not None:
            scores[movie_index] = float("-inf")

    number_of_candidates = min(
        top_k,
        len(scores),
    )

    top_scores, top_indices = torch.topk(
        scores,
        k=number_of_candidates,
    )

    recommendation_rows = []

    for rank, (
        movie_index,
        score,
    ) in enumerate(
        zip(
            top_indices.tolist(),
            top_scores.tolist(),
        ),
        start=1,
    ):
        movie_id = index_to_movie_id[movie_index]

        metadata_row = movies[
            movies["movieId"] == movie_id
        ]

        if metadata_row.empty:
            continue

        movie_row = metadata_row.iloc[0]

        recommendation_rows.append(
            {
                "rank": rank,
                "movieId": int(movie_id),
                "title": str(movie_row["title"]),
                "genres": str(movie_row["genres"]),
                "similarity_score": round(
                    float(score),
                    4,
                ),
            }
        )

    return pd.DataFrame(recommendation_rows)

#Interactive terminal application

def run_terminal_application():
    print("\n" + "=" * 70)
    print("GraphSAGE Movie Recommendation System")
    print("=" * 70)

    print(
        "\nSearch for movies you like. "
        "Enter at least three movie IDs."
    )

    selected_movie_ids = []

    while True:
        print(
            "\nType a movie title to search."
        )
        print(
            "Type 'done' when you have selected enough movies."
        )
        print(
            "Type 'quit' to close the program."
        )

        query = input("\nMovie search: ").strip()

        if query.lower() == "quit":
            print("Application closed.")
            return

        if query.lower() == "done":
            if len(selected_movie_ids) < 3:
                print(
                    "Please select at least three movies."
                )
                continue

            break

        search_result = search_movies(query)

        if search_result.empty:
            print("No matching movies were found.")
            continue

        print("\nSearch results:")
        print(
            search_result[
                [
                    "movieId",
                    "title",
                    "genres",
                ]
            ].to_string(index=False)
        )

        movie_id_input = input(
            "\nEnter one movieId to add, "
            "or press Enter to search again: "
        ).strip()

        if not movie_id_input:
            continue

        try:
            selected_movie_id = int(
                movie_id_input
            )
        except ValueError:
            print(
                "Movie ID must be a whole number."
            )
            continue

        if selected_movie_id not in movie_id_to_index:
            print(
                "That movie ID is not available "
                "inside the processed graph."
            )
            continue

        if selected_movie_id in selected_movie_ids:
            print(
                "That movie has already been selected."
            )
            continue

        selected_movie_ids.append(
            selected_movie_id
        )

        selected_title = movies[
            movies["movieId"] == selected_movie_id
        ]["title"]

        if not selected_title.empty:
            print(
                f"Added: {selected_title.iloc[0]}"
            )

        print(
            f"Selected movies: "
            f"{len(selected_movie_ids)}"
        )

    print("\nGenerating recommendations...")

    recommendations = recommend_from_liked_movies(
        liked_movie_ids=selected_movie_ids,
        top_k=10,
    )

    print("\nYour recommendations:")
    print("=" * 70)

    print(
        recommendations.to_string(
            index=False
        )
    )


#Program entry point

if __name__ == "__main__":
    run_terminal_application()