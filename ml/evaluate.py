from __future__ import annotations
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)

from model import GraphSAGERecommender

#Configuration

SEED = 42

BATCH_SIZE = 1024
NUM_NEIGHBORS = [15, 10]
NEGATIVE_SAMPLING_RATIO = 1.0
NUM_WORKERS = 0

#Keep this as None to evaluate every validation/test edge.
#
#For a quick test, temporarily use:
#MAX_EVALUATION_BATCHES = 20
MAX_EVALUATION_BATCHES = None

#Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#Locate project files

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"

MODEL_FILE = SAVED_MODEL_DIR / "graphsage_model.pt"
VALIDATION_GRAPH_FILE = SAVED_MODEL_DIR / "validation_graph.pt"
TEST_GRAPH_FILE = SAVED_MODEL_DIR / "test_graph.pt"

RESULTS_FILE = SAVED_MODEL_DIR / "evaluation_results.json"


for required_file in [
    MODEL_FILE,
    VALIDATION_GRAPH_FILE,
    TEST_GRAPH_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{required_file}"
        )


#Select CPU or GPU

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Evaluation will run on the CPU.")


#Define graph edge types

FORWARD_EDGE_TYPE = ("user", "likes", "movie")
REVERSE_EDGE_TYPE = ("movie", "liked_by", "user")

#Load graph splits

print("\nLoading validation graph...")

validation_graph = torch.load(
    VALIDATION_GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

print("Loading test graph...")

test_graph = torch.load(
    TEST_GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

validation_graph.validate(raise_on_error=True)
test_graph.validate(raise_on_error=True)

print("Validation and test graphs loaded successfully.")

#Load the trained checkpoint

print("\nLoading trained GraphSAGE checkpoint...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False,
)

required_checkpoint_keys = [
    "model_state_dict",
    "num_users",
    "num_movies",
    "embedding_dim",
    "hidden_channels",
    "output_channels",
    "dropout",
]

for key in required_checkpoint_keys:
    if key not in checkpoint:
        raise KeyError(
            f"Checkpoint is missing required key: {key}"
        )

print(
    f"Loaded checkpoint from epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Checkpoint training loss: "
    f"{checkpoint.get('training_loss', 'unknown')}"
)


#Function for creating an evaluation loader

def create_evaluation_loader(graph):
    edge_store = graph[FORWARD_EDGE_TYPE]

    if not hasattr(edge_store, "edge_label_index"):
        raise RuntimeError(
            "The graph has no edge_label_index. "
            "Rerun split_graph.py."
        )

    if not hasattr(edge_store, "edge_label"):
        raise RuntimeError(
            "The graph has no edge_label. "
            "Rerun split_graph.py."
        )

    loader = LinkNeighborLoader(
        data=graph,

        num_neighbors=NUM_NEIGHBORS,

        edge_label_index=(
            FORWARD_EDGE_TYPE,
            edge_store.edge_label_index,
        ),

        edge_label=edge_store.edge_label.float(),

        #Add one unseen negative pair for each positive edge.
        neg_sampling_ratio=NEGATIVE_SAMPLING_RATIO,

        batch_size=BATCH_SIZE,

        #Evaluation order does not need to be shuffled.
        shuffle=False,

        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return loader


validation_loader = create_evaluation_loader(
    validation_graph
)

test_loader = create_evaluation_loader(
    test_graph
)

print("\nEvaluation loaders created.")
print(
    f"Validation batches: {len(validation_loader):,}"
)
print(
    f"Test batches: {len(test_loader):,}"
)


#Create the same model architecture

model = GraphSAGERecommender(
    num_users=int(checkpoint["num_users"]),
    num_movies=int(checkpoint["num_movies"]),
    metadata=validation_graph.metadata(),
    embedding_dim=int(checkpoint["embedding_dim"]),
    hidden_channels=int(checkpoint["hidden_channels"]),
    output_channels=int(checkpoint["output_channels"]),
    dropout=float(checkpoint["dropout"]),
)

model = model.to(device)

#Initialize lazy GraphSAGE layers

#The SAGEConv layers were created using (-1, -1).
#Therefore, they need one real sampled batch to determine
#their input dimensions before the checkpoint is loaded.

print("\nInitializing model architecture...")

initial_batch = next(iter(validation_loader))
initial_batch = initial_batch.to(device)

initial_x_dict = {
    "user": model.user_embedding(
        initial_batch["user"].n_id
    ),
    "movie": model.movie_embedding(
        initial_batch["movie"].n_id
    ),
}

with torch.no_grad():
    initial_z_dict = model.encoder(
        initial_x_dict,
        initial_batch.edge_index_dict,
    )

    _ = model.decoder(
        user_embeddings=initial_z_dict["user"],
        movie_embeddings=initial_z_dict["movie"],
        edge_label_index=(
            initial_batch[FORWARD_EDGE_TYPE]
            .edge_label_index
        ),
    )

print("Model architecture initialized.")

#Load trained parameters

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Trained model parameters loaded successfully.")

#Loss function

loss_function = nn.BCEWithLogitsLoss()


#Evaluate one graph split

def evaluate_split(
    split_name: str,
    loader: LinkNeighborLoader,
) -> dict:
    print(f"\nEvaluating {split_name} set...")
    print("-" * 60)

    all_labels = []
    all_probabilities = []

    total_loss = 0.0
    total_examples = 0

    start_time = time.time()

    with torch.no_grad():
        for batch_number, batch in enumerate(
            loader,
            start=1,
        ):
            if (
                MAX_EVALUATION_BATCHES is not None
                and batch_number
                > MAX_EVALUATION_BATCHES
            ):
                break

            batch = batch.to(device)

            #n_id maps sampled local nodes back to their
            #original node indices in the complete graph.

            x_dict = {
                "user": model.user_embedding(
                    batch["user"].n_id
                ),
                "movie": model.movie_embedding(
                    batch["movie"].n_id
                ),
            }

            z_dict = model.encoder(
                x_dict,
                batch.edge_index_dict,
            )

            edge_store = batch[FORWARD_EDGE_TYPE]

            logits = model.decoder(
                user_embeddings=z_dict["user"],
                movie_embeddings=z_dict["movie"],
                edge_label_index=(
                    edge_store.edge_label_index
                ),
            )

            labels = edge_store.edge_label.float()

            loss = loss_function(
                logits,
                labels,
            )

            probabilities = torch.sigmoid(logits)

            batch_size = labels.numel()

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            all_labels.append(
                labels.detach().cpu()
            )

            all_probabilities.append(
                probabilities.detach().cpu()
            )

            if batch_number % 50 == 0:
                elapsed = time.time() - start_time

                print(
                    f"{split_name} | "
                    f"Batch {batch_number:,}/"
                    f"{len(loader):,} | "
                    f"Loss {loss.item():.4f} | "
                    f"Elapsed {elapsed:.1f}s"
                )

    if not all_labels:
        raise RuntimeError(
            f"No examples were evaluated for {split_name}."
        )

    labels_tensor = torch.cat(
        all_labels
    )

    probabilities_tensor = torch.cat(
        all_probabilities
    )

    labels_numpy = labels_tensor.numpy()
    probabilities_numpy = (
        probabilities_tensor.numpy()
    )

    predicted_labels = (
        probabilities_numpy >= 0.5
    ).astype(np.int32)

    unique_labels = np.unique(labels_numpy)

    if len(unique_labels) < 2:
        raise RuntimeError(
            f"{split_name} contains only one label class. "
            "ROC-AUC cannot be calculated."
        )

    average_loss = (
        total_loss / max(total_examples, 1)
    )

    roc_auc = roc_auc_score(
        labels_numpy,
        probabilities_numpy,
    )

    average_precision = average_precision_score(
        labels_numpy,
        probabilities_numpy,
    )

    accuracy = accuracy_score(
        labels_numpy,
        predicted_labels,
    )

    positive_examples = int(
        (labels_tensor == 1).sum().item()
    )

    negative_examples = int(
        (labels_tensor == 0).sum().item()
    )

    duration = time.time() - start_time

    results = {
        "split": split_name,
        "examples": int(total_examples),
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "binary_cross_entropy_loss": float(
            average_loss
        ),
        "roc_auc": float(roc_auc),
        "average_precision": float(
            average_precision
        ),
        "accuracy_at_0.5": float(accuracy),
        "duration_seconds": float(duration),
    }

    print(f"\n{split_name} results")
    print("=" * 60)
    print(f"Examples:          {total_examples:,}")
    print(f"Positive examples: {positive_examples:,}")
    print(f"Negative examples: {negative_examples:,}")
    print(f"Loss:              {average_loss:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(
        f"Average precision: "
        f"{average_precision:.4f}"
    )
    print(f"Accuracy @ 0.5:    {accuracy:.4f}")
    print(f"Time:              {duration:.1f}s")

    return results


#Evaluate validation and test sets

validation_results = evaluate_split(
    split_name="validation",
    loader=validation_loader,
)

test_results = evaluate_split(
    split_name="test",
    loader=test_loader,
)

#Save evaluation results

final_results = {
    "model_file": str(MODEL_FILE),
    "checkpoint_epoch": checkpoint.get(
        "epoch"
    ),
    "checkpoint_training_loss": checkpoint.get(
        "training_loss"
    ),
    "configuration": {
        "batch_size": BATCH_SIZE,
        "num_neighbors": NUM_NEIGHBORS,
        "negative_sampling_ratio": (
            NEGATIVE_SAMPLING_RATIO
        ),
        "maximum_evaluation_batches": (
            MAX_EVALUATION_BATCHES
        ),
    },
    "validation": validation_results,
    "test": test_results,
}

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_results,
        file,
        indent=4,
    )

print("\nEvaluation completed successfully.")
print(f"Results saved at:\n{RESULTS_FILE}")