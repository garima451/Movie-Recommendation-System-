from pathlib import Path
import random
import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit

#Reproducibility
#Using fixed seeds makes the split repeatable.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#Locate project files

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"

GRAPH_FILE = SAVED_MODEL_DIR / "movie_graph.pt"

TRAIN_FILE = SAVED_MODEL_DIR / "train_graph.pt"
VALIDATION_FILE = SAVED_MODEL_DIR / "validation_graph.pt"
TEST_FILE = SAVED_MODEL_DIR / "test_graph.pt"

#Check that the graph exists

if not GRAPH_FILE.exists():
    raise FileNotFoundError(
        f"Graph file was not found:\n{GRAPH_FILE}\n\n"
        "Run this command first:\n"
        "python ml/build_graph.py"
    )

#Load the graph

print("Loading complete user-movie graph...")

# weights_only=False is required because movie_graph.pt stores
# a HeteroData Python object, not only tensor weights.
graph = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

print("Graph loaded successfully.")
print(graph)

#Define graph relationships

FORWARD_EDGE_TYPE = ("user", "likes", "movie")
REVERSE_EDGE_TYPE = ("movie", "liked_by", "user")

#Count original edges

original_edge_count = graph[FORWARD_EDGE_TYPE].edge_index.size(1)

print(f"\nOriginal positive edges: {original_edge_count:,}")


#Configure edge splitting

split_transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.05,

    #We have two different node types, so this is not a
    #conventional homogeneous undirected graph.
    is_undirected=False,

    #Negative examples will be created later by the
    #mini-batch loader during training.
    neg_sampling_ratio=0.0,
    add_negative_train_samples=False,

    #Prevent the same user-movie pair from appearing more
    #than once inside a split.
    disjoint_train_ratio=0.1,

    edge_types=[FORWARD_EDGE_TYPE],
    rev_edge_types=[REVERSE_EDGE_TYPE],
)

#Produce train, validation and test graphs

print("\nSplitting graph...")

train_graph, validation_graph, test_graph = split_transform(graph)

print("Graph splitting completed.")

#Validate all three graphs

train_graph.validate(raise_on_error=True)
validation_graph.validate(raise_on_error=True)
test_graph.validate(raise_on_error=True)

print("All split graphs passed validation.")


#Inspect supervision edges

def print_split_information(name, split_data):
    """
    Print the number of message-passing edges and the number
    of labelled edges used as prediction targets.
    """

    edge_store = split_data[FORWARD_EDGE_TYPE]

    message_passing_edges = edge_store.edge_index.size(1)

    if hasattr(edge_store, "edge_label_index"):
        supervised_edges = edge_store.edge_label_index.size(1)
    else:
        supervised_edges = 0

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Message-passing edges: {message_passing_edges:,}")
    print(f"Supervision edges: {supervised_edges:,}")

    if hasattr(edge_store, "edge_label"):
        labels = edge_store.edge_label

        positive_labels = int((labels == 1).sum())
        negative_labels = int((labels == 0).sum())

        print(f"Positive labels: {positive_labels:,}")
        print(f"Negative labels: {negative_labels:,}")


print_split_information("Training graph", train_graph)
print_split_information("Validation graph", validation_graph)
print_split_information("Test graph", test_graph)


#save graph splits

print("\nSaving graph splits...")

torch.save(train_graph, TRAIN_FILE)
torch.save(validation_graph, VALIDATION_FILE)
torch.save(test_graph, TEST_FILE)

print("\nGraph splits saved successfully.")
print(f"Training graph:   {TRAIN_FILE}")
print(f"Validation graph: {VALIDATION_FILE}")
print(f"Test graph:       {TEST_FILE}")