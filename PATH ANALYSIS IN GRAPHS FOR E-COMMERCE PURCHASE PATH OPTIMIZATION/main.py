# GRAPH THEORY PROJECT: PATH ANALYSIS IN GRAPHS FOR E-COMMERCE PURCHASE PATH OPTIMIZATION
# GROUP MEMBERS: 22k-4148 SHAYAN | 22K-4159 RANIA | 22K-4320
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# FUNCTION FOR LOADING THE DATASET FROMTHE CSV FILE: "navigation_data.csv":
def load_dataset(file_path):
    """
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    dataset = pd.read_csv(file_path)
    return dataset

# FUNCTION FOR PREPROCESSING NAVIGATION DATA FOR CONSTRUCTION OF GRAPHS:
def preprocess_data(data):
    """
    Args:
        data (pd.DataFrame): Raw dataset.
    Returns:
        list: List of (source, destination) edges.
    """
    edges = list(zip(data['CurrentPage'], data['NextPage']))
    return edges

# FUNCTION FOR CONSTRUCTING DIRECTED GRAPH FROM EDGE LIST PROVIDED AS AN ARGUEMENT:
def build_graph(edges):
    """
    Args:
        edges (list): List of (source, destination) edges.
    Returns:
        nx.DiGraph: Directed graph object.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

# FFUNCTION FOR ANALYZING PATHS IN THE GRAPH:
def analyze_paths(G):
    """
    Args:
        G (nx.DiGraph): Directed graph.
    Returns:
        dict: Analysis results (e.g., shortest paths, most common paths).
    """
    shortest_paths = dict(nx.shortest_path_length(G))

    edge_list = list(G.edges()) # GET ALL EDGES AS A LIST OF COUNTING
    most_common_edges = Counter(edge_list).most_common(5)

    return {"shortest_paths": shortest_paths, "most_common_edges": most_common_edges}

# FUNCTION FOR EVALUATING THE GRAPH MODEL ON TEST DATA
def evaluate_graph(G, test_data):
    """
    Args:
        G (nx.DiGraph): Directed graph.
        test_data (list): Test edges for evaluation.
    Returns:
        dict: Precision, recall, F1 score.
    """
    true_edges = set(G.edges())
    predicted_edges = set(test_data)

    tp = len(true_edges & predicted_edges)  # TRUE POSITIVES
    fp = len(predicted_edges - true_edges)  # FALSE POSITIVE
    fn = len(true_edges - predicted_edges)  # FALSE NEGATIVES

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

# FUNCTION FOR VISULIZING THE GRAPH USING MATPLOTLIB LIBRARY:
def visualize_graph(G):
    """
    Args:
        G (nx.DiGraph): Directed graph.
    """
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_color="skyblue", node_size=2000, edge_color="gray", font_size=10)
    plt.title("E-commerce Navigation Graph")
    plt.show()

# MAIN FUNCTION
def main():
    # LOADING AND PREPROCESSING DATA:
    dataset_path = "navigation_data.csv"  # Replace with actual dataset path
    data = load_dataset(dataset_path)
    edges = preprocess_data(data)

    # GRAPH CONSTRUCTION:
    G = build_graph(edges)

    # PATH ANALYSIS:
    analysis_results = analyze_paths(G)
    print("Path Analysis Results:", analysis_results)

    # PERFORMANCE EVALUATION:
    test_edges = preprocess_data(data.sample(frac=0.2))  # Using a sample as test data
    evaluation_results = evaluate_graph(G, test_edges)
    print("Evaluation Results:", evaluation_results)

    # GRAPH VISUALIZATION:
    visualize_graph(G)

# MAIN FUNCTION EXECUTION:
if __name__ == "__main__":
    main()
