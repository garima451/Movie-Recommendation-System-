from pathlib import Path

import torch
from model import GraphSAGERecommender

#Locate the saved training graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"

TRAIN_GRAPH_FILE = SAVED_MODEL_DIR / "train_graph.pt"


if not TRAIN_GRAPH_FILE.exists():
    raise FileNotFoundError(
        f"Training graph was not found:\n"
        f"{TRAIN_GRAPH_FILE}\n\n"
        "Run this first:\n"
        "python ml/split_graph.py"
    )

#Load training graph
print("Loading training graph...")

train_graph = torch.load(
    TRAIN_GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

print("Training graph loaded.")

#Read graph dimensions
num_users = train_graph["user"].num_nodes
num_movies = train_graph["movie"].num_nodes
metadata = train_graph.metadata()

print(f"Number of users: {num_users:,}")
print(f"Number of movies: {num_movies:,}")
print(f"Metadata: {metadata}")

#Create GraphSAGE model
model = GraphSAGERecommender(
    num_users=num_users,
    num_movies=num_movies,
    metadata=metadata,
    embedding_dim=64,
    hidden_channels=128,
    output_channels=64,
    dropout=0.2,
)

print("\nModel created successfully:")
print(model)

#Select a tiny set of user-movie pairs

edge_type = ("user", "likes", "movie")

all_edge_label_index = (
    train_graph[edge_type].edge_label_index
)

# Only use five pairs for this test.
test_edge_label_index = all_edge_label_index[:, :5]

#Run one forward pass

model.eval()

with torch.no_grad():
    scores = model(
        edge_index_dict=train_graph.edge_index_dict,
        edge_label_index=test_edge_label_index,
    )

print("\nTest edge-label index:")
print(test_edge_label_index)

print("\nPredicted scores:")
print(scores)

print("\nPredicted score shape:")
print(scores.shape)

print("\nModel forward pass completed successfully.")