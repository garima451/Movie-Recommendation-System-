# Movie-Recommendation-System-
# GraphSAGE Movie Recommendation System

An end-to-end graph-based movie recommendation system built using **GraphSAGE**, **PyTorch Geometric**, and the **MovieLens 20M dataset**.

The project represents users and movies as nodes in a heterogeneous bipartite graph. Positive user–movie interactions are represented as edges, and GraphSAGE learns user and movie embeddings through neighborhood aggregation. These embeddings are later used to rank and recommend movies for a user.

## Project Objective

The objective of this project is to build a scalable recommendation pipeline that can answer:

> Given a set of movies a user likes, which unseen movies should be recommended next?

The project covers the complete machine-learning workflow:

- Raw-data preprocessing
- Positive interaction extraction
- Sparse user and movie filtering
- Heterogeneous graph construction
- Train, validation, and test edge splitting
- GraphSAGE model development
- Mini-batch link-prediction training
- Model evaluation
- Cold-start movie recommendation generation

## Key Skills Demonstrated

- Graph Neural Networks
- GraphSAGE
- Recommendation Systems
- Heterogeneous Graphs
- Link Prediction
- Neighborhood Sampling
- Negative Sampling
- Large-Scale Data Preprocessing
- PyTorch and PyTorch Geometric
- Model Evaluation
- Node Embeddings
- Cold-Start Recommendation
- Reproducible Machine-Learning Pipelines

## Technology Stack

- Python
- Pandas
- NumPy
- PyTorch
- PyTorch Geometric
- Scikit-learn
- Parquet
- JSON
- Visual Studio Code

## Dataset

This project uses the **MovieLens 20M Dataset**, a widely used benchmark dataset for building and evaluating movie recommendation systems. The dataset is provided by the GroupLens Research Lab at the University of Minnesota.

The dataset contains approximately:

* 20 million user ratings
* 138,000+ users
* 27,000+ movies

The raw dataset includes the following files:

* `rating.csv`
* `movie.csv`
* `tag.csv`
* `link.csv`

For this project, the recommendation pipeline primarily uses **`rating.csv`** and **`movie.csv`**.

* **`rating.csv`** contains user IDs, movie IDs, ratings, and timestamps. It is used to build the user–movie interaction graph.
* **`movie.csv`** contains movie IDs, movie titles, and genres. It is used to display meaningful movie recommendations to the user.

The remaining files (`tag.csv` and `link.csv`) are not used in the current implementation but can be incorporated in future versions to build richer graph features and hybrid recommendation models.

## Development Environment

The complete project was designed, implemented, and executed locally using Visual Studio Code as the primary development environment. All stages of the machine learning pipeline- including data preprocessing, graph construction, model development, training, evaluation, and recommendation generation- were implemented as individual Python scripts and executed sequentially through the Visual Studio Code terminal.

A dedicated Python virtual environment (venv) was used to manage project dependencies and ensure a reproducible execution environment throughout development.

The source code for the entire pipeline is organized inside the ml/ directory, where each script is responsible for a specific stage of the recommendation system. The scripts are executed in sequence, with the output of one stage serving as the input for the next stage.

This GitHub repository serves as a portfolio representation of the project and contains the complete source code, execution screenshots, preprocessing statistics, evaluation results, and sample recommendation outputs.

---

## Preprocessing Results

### Original Dataset Statistics

| Metric | Count |
|--------|-------:|
| Original Ratings | 20,000,263 |
| Original Users | 138,493 |
| Original Movies | 27,278 |

### Dataset After Filtering

| Metric | Count |
|--------|-------:|
| Active Users | 101,456 |
| Remaining Movies | 7,287 |
| Positive User–Movie Interactions | 9,396,876 |

The preprocessing stage successfully reduced the dataset by removing inactive users and rarely rated movies, resulting in a cleaner and denser interaction graph suitable for GraphSAGE training.

---

## Generated Files

The preprocessing stage generates the following files inside the `saved_model/` directory.

| File | Description |
|------|-------------|
| `ratings_processed.parquet` | Stores the cleaned positive user–movie interactions after preprocessing. |
| `movies_processed.parquet` | Stores the filtered movie metadata corresponding to the remaining movies. |
| `user_map.json` | Maps original MovieLens user IDs to continuous graph node indices. |
| `movie_map.json` | Maps original MovieLens movie IDs to continuous graph node indices. |
| `index_to_user.json` | Reverse mapping from graph node indices back to the original user IDs. |
| `index_to_movie.json` | Reverse mapping from graph node indices back to the original movie IDs. |
| `preprocessing_stats.json` | Stores preprocessing configuration and final dataset statistics for reproducibility. |

---

## Preprocessing Output

![Preprocessing Output - Part 1](results%20of%20preprocessing.png)
![Preprocessing Output - Part 2](results%20of%20preprocessing(pt-2).png)

**Figure.** Successful execution of the preprocessing pipeline showing dataset loading, iterative filtering, final graph statistics, and the generated preprocessing artifacts.

---

## Source Code

The implementation for this stage can be found here:

**[📄 preprocess.py](ml/preprocess.py)**

# 2. Graph Construction

## Overview

The second stage of the project converts the processed MovieLens dataset into a heterogeneous graph representation that can be directly used by the GraphSAGE Graph Neural Network.

Unlike traditional recommendation systems that operate on tabular data, Graph Neural Networks require the dataset to be represented as interconnected nodes and edges. In this project, every user and every movie becomes a graph node, while every positive user–movie interaction becomes an edge connecting the two.

The graph is built using the PyTorch Geometric `HeteroData` object, which supports multiple node types and relationship types.

---

## Graph Construction Workflow

The graph construction pipeline performs the following operations:

1. Loads the processed datasets generated during preprocessing.
2. Reads the graph node indices (`user_idx` and `movie_idx`).
3. Converts the interaction table into PyTorch tensors.
4. Creates the forward edge tensor representing **User → Likes → Movie** relationships.
5. Generates reverse edges representing **Movie → Liked By → User** relationships.
6. Creates a heterogeneous graph using the `HeteroData` data structure.
7. Adds user nodes and movie nodes to the graph.
8. Validates the graph structure to ensure that all node indices and edge connections are correct.
9. Saves the complete heterogeneous graph for the training stage.

---

## Graph Representation

The graph contains two different node types:

- User
- Movie

It also contains two directed edge relationships:

```
User ───── likes ─────► Movie
Movie ── liked_by ───► User
```

The reverse edges are included to allow GraphSAGE to propagate information in both directions during message passing.

---

## Graph Statistics

After graph construction:

- User Nodes: **101,456**
- Movie Nodes: **7,287**
- Positive User–Movie Interactions: **9,396,876**
- Forward Edge Tensor Shape: **[2, 9,396,876]**

The heterogeneous graph contains:

- User node type
- Movie node type
- User → Likes → Movie edges
- Movie → Liked By → User edges

---

## Generated File

The graph construction stage generates the following file inside the `saved_model/` directory.

| File | Purpose |
|------|---------|
| movie_graph.pt | Stores the heterogeneous graph used during graph splitting and GraphSAGE training. |

---

## Graph Construction Output

### Terminal Execution


---

## Source Code

The complete implementation for this stage can be found here:

📄 [build_graph.py](ml/build_graph.py)










