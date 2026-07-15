from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import HeteroData

#Locate project folders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "saved_model"

RATINGS_FILE = PROCESSED_DIR / "ratings_processed.parquet"
MOVIES_FILE = PROCESSED_DIR / "movies_processed.parquet"
GRAPH_FILE = PROCESSED_DIR / "movie_graph.pt"

#Confirm processed files exist

if not RATINGS_FILE.exists():
    raise FileNotFoundError(
        f"Processed ratings file not found:\n{RATINGS_FILE}\n"
        "Run preprocess.py first."
    )

if not MOVIES_FILE.exists():
    raise FileNotFoundError(
        f"Processed movies file not found:\n{MOVIES_FILE}\n"
        "Run preprocess.py first."
    )

#Load processed data

print("Loading processed ratings...")

ratings = pd.read_parquet(
    RATINGS_FILE,
    columns=["user_idx", "movie_idx"],
)

movies = pd.read_parquet(
    MOVIES_FILE,
    columns=["movie_idx", "movieId", "title", "genres"],
)

num_users = int(ratings["user_idx"].max()) + 1
num_movies = int(ratings["movie_idx"].max()) + 1

print(f"Users: {num_users:,}")
print(f"Movies: {num_movies:,}")
print(f"Positive interactions: {len(ratings):,}")

#Convert interaction table into edge tensor

user_indices = torch.tensor(
    ratings["user_idx"].to_numpy(),
    dtype=torch.long,
)

movie_indices = torch.tensor(
    ratings["movie_idx"].to_numpy(),
    dtype=torch.long,
)

#Shape:
#[2, number_of_edges]
#
#First row: user graph indices
#Second row: movie graph indices

edge_index = torch.stack(
    [user_indices, movie_indices],
    dim=0,
)

print(f"Forward edge tensor shape: {edge_index.shape}")

#Build heterogeneous user-movie graph

graph = HeteroData()

#Two different kinds of nodes:
# - user nodes
# - movie nodes

graph["user"].num_nodes = num_users
graph["movie"].num_nodes = num_movies

# Forward relationship:
# user --likes--> movie

graph["user", "likes", "movie"].edge_index = edge_index

# Reverse relationship:
# movie --liked_by--> user
#
# GraphSAGE needs information to flow in both directions.

reverse_edge_index = edge_index.flip(0)

graph["movie", "liked_by", "user"].edge_index = reverse_edge_index


#Validate the graph

graph.validate(raise_on_error=True)

print("\nGraph metadata:")
print(graph.metadata())

print("\nGraph structure:")
print(graph)

#Save graph

torch.save(graph, GRAPH_FILE)

print("\nGraph created successfully.")
print(f"Saved at: {GRAPH_FILE}")