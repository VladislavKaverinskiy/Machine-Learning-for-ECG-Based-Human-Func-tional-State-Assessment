import os
import umap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import hdbscan
from scipy.stats import f_oneway
from scipy.optimize import minimize
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score
import itertools
from sklearn.metrics import confusion_matrix
from itertools import product
from typing import List

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*deprecation.*")


def taget(x, embedding):
    clusterization_results = clastrrize(embedding, metric="euclidean", min_cluster_size=int(x[0]),
                                        leaf_size=int(x[1]), p=2)
    clusterer = clusterization_results["clusterer"]
    n_clusters = clusterization_results["n_clusters"]
    print("n_clusters ", n_clusters)
    size = clusterer.labels_.size
    bool_arr = (clusterer.labels_ == -1)
    count = np.count_nonzero(bool_arr)
    print(100 * float(count) / float(size))
    return float(count) / float(size)

def plot_umap_with_labels(X, y, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Reduces the feature dimension of X using UMAP and visualizes the result with color coding by y-labels.

        Parameters:
            X: np.ndarray or pd.DataFrame
            Feature matrix.
            y: array-like
            Label array (e.g., classes).
            n_neighbors: int
            Number of neighbors for UMAP.
            min_dist: float
            Minimum distance between points in the projection.
            metric: str
            Distance metric (e.g., 'euclidean', 'manhattan', 'chebyshev').
            random_state: int
            Random seed for reproducibility.

        Returns:
            embedding: np.ndarray
            2D coordinates after UMAP projection.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        random_state=random_state)
    embedding = reducer.fit_transform(X)

    # Creation of scatter plot
    plt.figure(figsize=(8, 6))  # slightly smaller area

    palette = sns.color_palette("hls", len(set(y)))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=y,
        palette=palette,
        s=50,
        linewidth=0.5,
        edgecolor='k'
    )

    plt.title(f'UMAP projection (metric={metric}, neighbors={n_neighbors}, min_dist={min_dist})', fontsize=16)
    plt.xlabel('UMAP-1', fontsize=14)
    plt.ylabel('UMAP-2', fontsize=14)

    plt.legend(title='Class', title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    return embedding


def load_features_and_labels(csv_path='work_data/normalized..csv', class_col='class'):
    """
    Loads data from a CSV file, separating features and class labels.

        Parameters:
            csv_path (str): path to the CSV file.
            class_col (str): name of the column with class labels.

        Returns:
            x (pd.DataFrame): features.
            y (pd.Series): class labels.
    """
    df = pd.read_csv(csv_path)
    if class_col not in df.columns:
        raise ValueError(f"Column '{class_col}' is not found in file.")
    y = df[class_col]
    X = df.drop(columns=[class_col])
    return X, y


def plot_clusters(clusterer, embedding, n_clusters, file_name='clusters.png', show_plot=True, save_plot=False):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=clusterer.labels_, linewidth=0.5,
                cmap=plt.get_cmap('Spectral', n_clusters+1),
                edgecolors='black', alpha=0.25, s=50)
    ax.set_xticks(np.arange(round(min(embedding[:, 0]) - 1), round(max(embedding[:, 0]) + 1, 2)))
    ax.set_yticks(np.arange(round(min(embedding[:, 1]) - 1), round(max(embedding[:, 1]) + 1, 2)))
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.colorbar(boundaries=np.arange(clusterer.labels_.min(),
                                      clusterer.labels_.max()+2) - 0.5).set_ticks(np.arange(clusterer.labels_.min(),
                                                                                            clusterer.labels_.max()+1))
    plt.title('HDBSCAN clusters of projection', fontsize=24)
    if save_plot:
        plt.savefig(file_name)
    if show_plot:
        plt.show()
    plt.clf()


def preprocess_classes(X, y, exclude_classes=None, merge_classes=None):
    """
    Excludes and merges classes in a dataset.

        Parameters:
            - X: np.ndarray or pd.DataFrame - features
            - y: np.ndarray or pd.Series - class labels
            - exclude_classes: list[int] - list of classes to exclude
            - merge_classes: dict[int, list[int]] - dictionary,
              where the key is the new class and the value is the list of old classes to merge

        Returns:
            - X_new: np.ndarray - updated features
            - y_new: np.ndarray - updated class labels
    """
    X = np.array(X)
    y = np.array(y)

    mask = np.ones(len(y), dtype=bool)

    # Eliminating classes
    if exclude_classes:
        for cls in exclude_classes:
            mask &= (y != cls)

    X_new = X[mask]
    y_new = y[mask]

    # Combining classes
    if merge_classes:
        for new_cls, old_classes in merge_classes.items():
            for old_cls in old_classes:
                y_new[y_new == old_cls] = new_cls

    return X_new, y_new


def clastrrize(data, metric='euclidean', min_cluster_size=50, leaf_size=40, p=2, prediction_data=True):
    clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=min_cluster_size, leaf_size=leaf_size,
                                p=p, prediction_data=prediction_data)
    clusterer.fit(data)
    clusters_labels = clusterer.labels_
    n_clusters = clusterer.labels_.max() + 1
    probabilities = clusterer.probabilities_
    return {"clusterer": clusterer, "n_clusters": n_clusters, "clusters_labels": clusters_labels,
            "clusters_probabilities": probabilities}


def evaluate_clustering_quality(y_true, y_pred, noise_label=-1, alpha=1.0, beta=0.5):
    mask = y_pred != noise_label
    if np.sum(mask) < 0.5 * len(y_true):
        return -np.inf  # bad situation

    y_true_filtered = np.array(y_true)[mask]
    y_pred_filtered = np.array(y_pred)[mask]

    similarity = adjusted_rand_score(y_true_filtered, y_pred_filtered)
    noise_ratio = 1.0 - (len(y_pred_filtered) / len(y_pred))

    return alpha * similarity - beta * noise_ratio


def optimize_hdbscan_params(embedding, y_true,
                             metric_list=["euclidean", "manhattan", "chebyshev"],
                             min_cluster_size_list=range(5, 100, 1),
                             leaf_size_list=range(5, 100, 1),
                             noise_penalty_weight=0.7,
                             min_valid_ratio=0.3):
    """
    noise_penalty_weight 0 the penalty for noise.
    For example, if 0.5, then if 40 % of the samples are outclassed, the final score is reduced by 0.2.
    """
    best_score = -np.inf
    best_params = None
    best_clusterer = None

    for metric, min_cluster_size, leaf_size in product(metric_list, min_cluster_size_list, leaf_size_list):
        try:
            print(f"Testing: min_cluster_size={min_cluster_size}, leaf_size={leaf_size}, metric={metric}")
            clusterer = hdbscan.HDBSCAN(metric=metric,
                                        min_cluster_size=min_cluster_size,
                                        leaf_size=leaf_size)
            labels_pred = clusterer.fit_predict(embedding)

            # Mask of valid clusters (not -1)
            valid_mask = labels_pred != -1
            valid_ratio = np.count_nonzero(valid_mask) / len(labels_pred)

            if valid_ratio < min_valid_ratio:  # If there is very little clustering, we skip it.
                continue

            # Calculate ARI only for valid
            ari_score = adjusted_rand_score(y_true[valid_mask], labels_pred[valid_mask])

            # Penalty for declassified
            penalty = noise_penalty_weight * (1 - valid_ratio)
            adjusted_score = ari_score - penalty

            print(f"ARI: {ari_score:.4f}, penalty: {penalty:.4f}, adjusted: {adjusted_score:.4f}, best: {best_score:.4f}")

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_params = {
                    "metric": metric,
                    "min_cluster_size": min_cluster_size,
                    "leaf_size": leaf_size
                }
                best_clusterer = clusterer
                print(f"New best score: {adjusted_score:.4f}")
        except Exception as e:
            print(f"Error for {metric}, {min_cluster_size}, {leaf_size}: {e}")
            continue

    return best_params, best_score, best_clusterer


def greedy_optimize_hdbscan_params(embedding, y_true,
                                    metric_list=["euclidean", "manhattan", "chebyshev"],
                                    min_cluster_size_range=(5, 100),
                                    leaf_size_range=(5, 100),
                                    step_fraction=0.01,  # relative step
                                    max_iterations=50):
    best_score = -1
    best_params = None
    best_clusterer = None

    for metric in metric_list:
        print(f"Testing metric: {metric}")
        min_cs = (min_cluster_size_range[0] + min_cluster_size_range[1]) // 2
        leaf_sz = (leaf_size_range[0] + leaf_size_range[1]) // 2

        step_cs = max(1, int((min_cluster_size_range[1] - min_cluster_size_range[0]) * step_fraction))
        step_ls = max(1, int((leaf_size_range[1] - leaf_size_range[0]) * step_fraction))

        for _ in range(max_iterations):
            candidates = [
                (min_cs, leaf_sz),
                (min_cs + step_cs, leaf_sz),
                (min_cs - step_cs, leaf_sz),
                (min_cs, leaf_sz + step_ls),
                (min_cs, leaf_sz - step_ls),
            ]

            candidates = [(cs, ls) for cs, ls in candidates
                          if min_cluster_size_range[0] <= cs <= min_cluster_size_range[1]
                          and leaf_size_range[0] <= ls <= leaf_size_range[1]]

            local_best = (min_cs, leaf_sz)
            local_best_score = -1
            local_best_clusterer = None

            for cs, ls in candidates:
                try:
                    clusterer = hdbscan.HDBSCAN(metric=metric,
                                                min_cluster_size=cs,
                                                leaf_size=ls)
                    labels_pred = clusterer.fit_predict(embedding)
                    valid_mask = labels_pred != -1
                    if np.count_nonzero(valid_mask) < 0.7 * len(labels_pred):
                        continue
                    score = adjusted_rand_score(y_true[valid_mask], labels_pred[valid_mask])

                    if score > local_best_score:
                        local_best = (cs, ls)
                        local_best_score = score
                        local_best_clusterer = clusterer

                except Exception as e:
                    print(f"Error for {metric}, {cs}, {ls}: {e}")
                    continue

            if local_best_score > best_score:
                best_score = local_best_score
                best_params = {
                    "metric": metric,
                    "min_cluster_size": local_best[0],
                    "leaf_size": local_best[1]
                }
                best_clusterer = local_best_clusterer

            # Updating the current point
            if local_best == (min_cs, leaf_sz):
                break  # local maximum reached
            else:
                min_cs, leaf_sz = local_best

    return best_params, best_score, best_clusterer


def save_clustered_data_separately(original_data: pd.DataFrame, labels_pred, output_dir="clusters_output"):
    """
    Saves original_data rows into separate CSV files for predicted clusters.
    Noise (label == -1) is saved to a separate file, cluster_noise.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = set(labels_pred)

    for label in unique_labels:
        # Filter rows by the current label
        cluster_data = original_data[labels_pred == label]

        # Determine the file name
        if label == -1:
            filename = "cluster_noise.csv"
            filename_2 = "cluster_noise.xlsx"
        else:
            filename = f"cluster_{label}.csv"
            filename_2 = f"cluster_{label}.xlsx"

        # Save to file
        cluster_data.to_csv(os.path.join(output_dir, filename), index=False, encoding="utf-8-sig")
        cluster_data.to_excel(os.path.join(output_dir, filename_2), index=False, engine='openpyxl')

    print(f"Saved {len(unique_labels)} files to a folder '{output_dir}'.")


if __name__ == "__main__":
    X, y = load_features_and_labels()
    # embedding = plot_umap_with_labels(X, y, n_neighbors=15, min_dist=0.1, metric='manhattan')
    # print(embedding)

    X_new, y_new = preprocess_classes(X, y,
                                      exclude_classes=[5],
                                      merge_classes={10: [1, 2, 4], 11: [3, 6]})

    print(len(X_new))
    embedding = plot_umap_with_labels(X_new, y_new, n_neighbors=15, min_dist=0.1, metric='manhattan')

    clusterization_results = clastrrize(embedding, metric="manhattan", min_cluster_size=7, leaf_size=5,
                                        p=2)
    clusterer = clusterization_results["clusterer"]
    n_clusters = clusterization_results["n_clusters"]
    print("n_clusters ", n_clusters)
    print(clusterer.labels_)
    labels = list(clusterer.labels_)

    size = clusterer.labels_.size
    print(size)
    bool_arr = (clusterer.labels_ == -1)
    count = np.count_nonzero(bool_arr)
    print(100 * float(count) / float(size))

    score = adjusted_rand_score(y_new, clusterer.labels_)
    print(score)

    plot_clusters(clusterer, embedding, n_clusters=len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0))

    df = pd.read_csv("work_data/preprocessed.csv", encoding="utf-8-sig")

    X_original = df.copy()
    X_original = X_original[~X_original["class"].isin([5])].copy()
    save_clustered_data_separately(original_data=X_original, labels_pred=clusterer.labels_)
