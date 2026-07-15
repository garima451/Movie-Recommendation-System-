from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


class GraphSAGEEncoder(nn.Module):
    """
    Homogeneous GraphSAGE encoder.

    PyTorch Geometric will convert this into a heterogeneous
    encoder for user and movie nodes using to_hetero().
    """

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # (-1, -1) allows the layer to infer the source and
        # destination feature dimensions during initialization.
        self.conv1 = SAGEConv(
            (-1, -1),
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            (-1, -1),
            output_channels,
        )

        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


class DotProductDecoder(nn.Module):
    """
    Produces one recommendation score for every user-movie pair.

    High score:
        the model believes the user is likely to like the movie.

    Low score:
        the model believes the user is unlikely to like the movie.
    """

    def forward(
        self,
        user_embeddings: Tensor,
        movie_embeddings: Tensor,
        edge_label_index: Tensor,
    ) -> Tensor:
        user_indices = edge_label_index[0]
        movie_indices = edge_label_index[1]

        selected_users = user_embeddings[user_indices]
        selected_movies = movie_embeddings[movie_indices]

        # Element-wise multiplication followed by summation
        # gives the dot product for every user-movie pair.
        scores = (
            selected_users
            * selected_movies
        ).sum(dim=-1)

        return scores


class GraphSAGERecommender(nn.Module):
    """
    Complete recommendation model.

    Components:
    1. Trainable user ID embeddings
    2. Trainable movie ID embeddings
    3. Heterogeneous GraphSAGE encoder
    4. Dot-product link decoder
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        metadata,
        embedding_dim: int = 64,
        hidden_channels: int = 128,
        output_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        # Each user receives a trainable vector.
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        # Each movie receives a trainable vector.
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim,
        )

        base_encoder = GraphSAGEEncoder(
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            dropout=dropout,
        )

        # Convert the encoder into a heterogeneous model using:
        #
        # node types:
        #   user, movie
        #
        # edge types:
        #   user -> likes -> movie
        #   movie -> liked_by -> user
        self.encoder = to_hetero(
            base_encoder,
            metadata,
            aggr="sum",
        )

        self.decoder = DotProductDecoder()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize trainable user and movie vectors.
        """

        nn.init.xavier_uniform_(
            self.user_embedding.weight
        )

        nn.init.xavier_uniform_(
            self.movie_embedding.weight
        )

    def create_input_features(self) -> dict[str, Tensor]:
        """
        Return the trainable input feature dictionaries expected
        by a heterogeneous PyTorch Geometric model.
        """

        return {
            "user": self.user_embedding.weight,
            "movie": self.movie_embedding.weight,
        }

    def encode(
        self,
        edge_index_dict,
    ) -> dict[str, Tensor]:
        """
        Run GraphSAGE message passing and return final embeddings
        for every user and movie.
        """

        x_dict = self.create_input_features()

        z_dict = self.encoder(
            x_dict,
            edge_index_dict,
        )

        return z_dict

    def decode(
        self,
        z_dict: dict[str, Tensor],
        edge_label_index: Tensor,
    ) -> Tensor:
        """
        Score user-movie candidate pairs.
        """

        return self.decoder(
            user_embeddings=z_dict["user"],
            movie_embeddings=z_dict["movie"],
            edge_label_index=edge_label_index,
        )

    def forward(
        self,
        edge_index_dict,
        edge_label_index: Tensor,
    ) -> Tensor:
        """
        Complete forward pass:

        graph structure
            -> GraphSAGE embeddings
            -> user-movie scores
        """

        z_dict = self.encode(edge_index_dict)

        scores = self.decode(
            z_dict,
            edge_label_index,
        )

        return scores