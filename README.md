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








