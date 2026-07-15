from __future__ import annotations
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import LinkNeighborLoader
from model import GraphSAGERecommender

#Configuration
SEED = 42
EMBEDDING_DIM = 64
HIDDEN_CHANNELS = 128
OUTPUT_CHANNELS = 64
DROPOUT = 0.2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024

#Two entries because our GraphSAGE model has two layers:
#layer 1 samples up to 15 neighbors
#layer 2 samples up to 10 neighbors
NUM_NEIGHBORS = [15, 10]
NEGATIVE_SAMPLING_RATIO = 1.0
EPOCHS = 5
NUM_WORKERS = 0

#Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"

TRAIN_GRAPH_FILE = SAVED_MODEL_DIR / "train_graph.pt"

MODEL_FILE = SAVED_MODEL_DIR / "graphsage_model.pt"
CHECKPOINT_FILE = SAVED_MODEL_DIR / "graphsage_checkpoint.pt"


if not TRAIN_GRAPH_FILE.exists():
    raise FileNotFoundError(
        f"Training graph was not found:\n{TRAIN_GRAPH_FILE}\n\n"
        "Run this command first:\n"
        "python ml/split_graph.py"
    )

#Choose CPU or GPU

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(
        "CUDA GPU was not detected. Training will use the CPU."
    )


#Load training graph

print("\nLoading training graph...")

train_graph = torch.load(
    TRAIN_GRAPH_FILE,
    map_location="cpu",
    weights_only=False,
)

train_graph.validate(raise_on_error=True)

print("Training graph loaded successfully.")
print(train_graph)


#Define edge types

FORWARD_EDGE_TYPE = ("user", "likes", "movie")
REVERSE_EDGE_TYPE = ("movie", "liked_by", "user")

edge_store = train_graph[FORWARD_EDGE_TYPE]


if not hasattr(edge_store, "edge_label_index"):
    raise RuntimeError(
        "The training graph has no edge_label_index. "
        "Rerun split_graph.py."
    )

if not hasattr(edge_store, "edge_label"):
    raise RuntimeError(
        "The training graph has no edge_label. "
        "Rerun split_graph.py."
    )


print(
    f"\nTraining supervision edges: "
    f"{edge_store.edge_label_index.size(1):,}"
)

print(
    f"Training message-passing edges: "
    f"{edge_store.edge_index.size(1):,}"
)

#Create mini-batch neighbor loader

print("\nCreating mini-batch neighbor loader...")

train_loader = LinkNeighborLoader(
    data=train_graph,

    #number of neighbors sampled for the two GraphSAGE layers.
    num_neighbors=NUM_NEIGHBORS,

    #These are the positive edges the model must learn to predict.
    edge_label_index=(
        FORWARD_EDGE_TYPE,
        edge_store.edge_label_index,
    ),

    edge_label=edge_store.edge_label.float(),

    #For every positive user-movie edge, sample one negative pair.
    neg_sampling_ratio=NEGATIVE_SAMPLING_RATIO,

    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,

    #Keeps all tensors in normal pageable CPU memory.
    #We can enable it later for GPU optimization.
    pin_memory=False,
)

print("Training loader created.")
print(f"Approximate batches per epoch: {len(train_loader):,}")

#Create GraphSAGE model

num_users = int(train_graph["user"].num_nodes)
num_movies = int(train_graph["movie"].num_nodes)

model = GraphSAGERecommender(
    num_users=num_users,
    num_movies=num_movies,
    metadata=train_graph.metadata(),
    embedding_dim=EMBEDDING_DIM,
    hidden_channels=HIDDEN_CHANNELS,
    output_channels=OUTPUT_CHANNELS,
    dropout=DROPOUT,
)

model = model.to(device)

print("\nModel created:")
print(f"Users: {num_users:,}")
print(f"Movies: {num_movies:,}")

#Initialize lazy GraphSAGE layers
#SAGEConv was created with (-1, -1), so some dimensions are
#initialized only after seeing a real batch.

print("\nInitializing model with one sampled batch...")

initial_batch = next(iter(train_loader))
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
        initial_z_dict["user"],
        initial_z_dict["movie"],
        initial_batch[FORWARD_EDGE_TYPE].edge_label_index,
    )

print("Model initialization completed.")

#Optimizer and loss

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

#Model output is a raw score called a logit.
#BCEWithLogitsLoss internally applies sigmoid.
loss_function = nn.BCEWithLogitsLoss()

#One training epoch

def train_one_epoch(
    epoch_number: int,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    epoch_start = time.time()

    for batch_number, batch in enumerate(
        train_loader,
        start=1,
    ):
        batch = batch.to(device)

        optimizer.zero_grad()

        #batch["user"].n_id contains the original/global user
        #indices represented inside this sampled mini-batch.
        #
        #The sampled graph uses local indices, so we first
        #select the corresponding global embedding rows.

        x_dict = {
            "user": model.user_embedding(
                batch["user"].n_id
            ),
            "movie": model.movie_embedding(
                batch["movie"].n_id
            ),
        }

        #Run GraphSAGE only on this sampled neighborhood.
        z_dict = model.encoder(
            x_dict,
            batch.edge_index_dict,
        )

        batch_edge_store = batch[FORWARD_EDGE_TYPE]

        predictions = model.decoder(
            user_embeddings=z_dict["user"],
            movie_embeddings=z_dict["movie"],
            edge_label_index=(
                batch_edge_store.edge_label_index
            ),
        )

        labels = batch_edge_store.edge_label.float()

        loss = loss_function(
            predictions,
            labels,
        )

        loss.backward()

        #Prevent unusually large gradient updates.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = labels.numel()

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if batch_number % 100 == 0:
            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch_number:02d} | "
                f"Batch {batch_number:,}/{len(train_loader):,} | "
                f"Loss {loss.item():.4f} | "
                f"Elapsed {elapsed:.1f}s"
            )

    average_loss = total_loss / max(total_examples, 1)

    return average_loss

#Main training loop

print("\nStarting GraphSAGE training...")
print("=" * 60)

best_training_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    average_loss = train_one_epoch(epoch)

    epoch_duration = time.time() - epoch_start

    print(
        f"\nEpoch {epoch:02d} completed | "
        f"Average loss: {average_loss:.4f} | "
        f"Time: {epoch_duration:.1f}s\n"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss": average_loss,
        "num_users": num_users,
        "num_movies": num_movies,
        "metadata": train_graph.metadata(),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_channels": HIDDEN_CHANNELS,
        "output_channels": OUTPUT_CHANNELS,
        "dropout": DROPOUT,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE,
    )

    if average_loss < best_training_loss:
        best_training_loss = average_loss

        torch.save(
            checkpoint,
            MODEL_FILE,
        )

        print(
            f"Saved new best model with loss "
            f"{best_training_loss:.4f}"
        )


print("=" * 60)
print("Training completed successfully.")
print(f"Best training loss: {best_training_loss:.4f}")
print(f"Best model saved at: {MODEL_FILE}")
print(f"Latest checkpoint saved at: {CHECKPOINT_FILE}")