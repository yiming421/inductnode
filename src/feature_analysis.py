"""
Feature Analysis and Visualization Module

This module contains functions for analyzing feature quality and visualizing
transformations like PCA, random projections, etc.

Functions for:
- Computing feature statistics (Fisher Score, Silhouette Score, Effective Rank, etc.)
- Printing feature statistics comparisons
- t-SNE visualization
"""

import torch
import numpy as np
import os
import time
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def compute_feature_statistics(features, labels, feature_type="Features"):
    """
    Compute statistical metrics for feature quality analysis.

    Args:
        features: Feature matrix [N, D] (numpy array or torch tensor)
        labels: Class labels [N] (numpy array or torch tensor)
        feature_type: Description string

    Returns:
        dict: Dictionary with various statistics
    """
    # Convert to numpy
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    n_samples, n_features = features.shape
    classes = np.unique(labels)
    n_classes = len(classes)

    stats = {}

    # 1. Fisher Score (class separability)
    # Higher is better - measures ratio of between-class to within-class variance
    class_means = np.array([features[labels == c].mean(axis=0) for c in classes])
    overall_mean = features.mean(axis=0)

    # Between-class scatter
    S_B = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        n_c = np.sum(labels == c)
        mean_diff = (class_means[i] - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

    # Within-class scatter
    S_W = np.zeros((n_features, n_features))
    for c in classes:
        class_features = features[labels == c]
        class_mean = class_features.mean(axis=0)
        S_W += (class_features - class_mean).T @ (class_features - class_mean)

    # Fisher score per feature
    fisher_scores = np.diagonal(S_B) / (np.diagonal(S_W) + 1e-10)
    stats['fisher_score_mean'] = float(np.mean(fisher_scores))
    stats['fisher_score_std'] = float(np.std(fisher_scores))
    stats['fisher_score_max'] = float(np.max(fisher_scores))

    # 2. Effective Rank
    # Lower means more redundancy/compression
    _, singular_values, _ = np.linalg.svd(features, full_matrices=False)
    singular_values = singular_values / np.sum(singular_values)
    entropy = -np.sum(singular_values * np.log(singular_values + 1e-10))
    effective_rank = np.exp(entropy)
    stats['effective_rank'] = float(effective_rank)
    stats['effective_rank_ratio'] = float(effective_rank / n_features)

    # 3. Silhouette Score (cluster quality)
    # Range [-1, 1], higher is better
    # Only compute if we have enough samples and not too many
    if n_samples > n_classes and n_samples < 10000:
        silhouette = silhouette_score(features, labels, metric='euclidean', sample_size=min(5000, n_samples))
        stats['silhouette_score'] = float(silhouette)
    else:
        stats['silhouette_score'] = None

    # 4. Variance statistics
    feature_variances = np.var(features, axis=0)
    stats['variance_mean'] = float(np.mean(feature_variances))
    stats['variance_std'] = float(np.std(feature_variances))
    stats['variance_total'] = float(np.sum(feature_variances))

    # 5. Intra-class vs Inter-class variance ratio
    intra_class_var = 0
    for c in classes:
        class_features = features[labels == c]
        class_mean = class_features.mean(axis=0)
        intra_class_var += np.sum((class_features - class_mean) ** 2)
    intra_class_var /= n_samples

    inter_class_var = 0
    for i, c in enumerate(classes):
        n_c = np.sum(labels == c)
        inter_class_var += n_c * np.sum((class_means[i] - overall_mean) ** 2)
    inter_class_var /= n_samples

    stats['intra_class_variance'] = float(intra_class_var)
    stats['inter_class_variance'] = float(inter_class_var)
    stats['variance_ratio'] = float(inter_class_var / (intra_class_var + 1e-10))

    return stats


def print_feature_statistics(stats_original, stats_processed, method_name, dataset_name):
    """
    Print a formatted comparison of feature statistics before and after transformation.

    Args:
        stats_original: Statistics dict from original features
        stats_processed: Statistics dict from processed features
        method_name: Name of the transformation method
        dataset_name: Name of the dataset
    """
    print("=" * 80)
    print(f"Feature Statistics Analysis: {dataset_name} - {method_name}")
    print("=" * 80)
    print()

    print(f"{'Metric':<35} {'Original':<20} {'After Transform':<20} {'Change'}")
    print("-" * 80)

    # Helper function to format values and compute change
    def format_stat(key, original, processed, higher_is_better=None):
        orig_val = original.get(key)
        proc_val = processed.get(key)

        if orig_val is None or proc_val is None:
            return f"{'N/A':<20} {'N/A':<20} N/A"

        # Format values
        if isinstance(orig_val, float):
            orig_str = f"{orig_val:.4f}"
            proc_str = f"{proc_val:.4f}"
        else:
            orig_str = f"{orig_val}"
            proc_str = f"{proc_val}"

        # Compute change
        if orig_val != 0:
            diff = proc_val - orig_val
            pct_change = (diff / abs(orig_val)) * 100
            if abs(diff) < 0.0001:
                change_str = "~0"
            else:
                change_str = f"↑ {abs(diff):.4f} ({pct_change:+.1f}%)"
                if diff < 0:
                    change_str = f"↓ {abs(diff):.4f} ({pct_change:.1f}%)"

                # Add GOOD/BAD indicator
                if higher_is_better is not None:
                    if (higher_is_better and diff > 0) or (not higher_is_better and diff < 0):
                        change_str += " GOOD"
                    else:
                        change_str += " BAD"
        else:
            change_str = "N/A"

        return f"{orig_str:<20} {proc_str:<20} {change_str}"

    # Print metrics
    print(f"{'Fisher Score (mean)':<35} {format_stat('fisher_score_mean', stats_original, stats_processed, True)}")
    print(f"{'Fisher Score (max)':<35} {format_stat('fisher_score_max', stats_original, stats_processed, True)}")
    print(f"{'Effective Rank':<35} {format_stat('effective_rank', stats_original, stats_processed, None)}")
    print(f"{'Effective Rank Ratio':<35} {format_stat('effective_rank_ratio', stats_original, stats_processed, None)}")
    print(f"{'Silhouette Score':<35} {format_stat('silhouette_score', stats_original, stats_processed, True)}")
    print(f"{'Variance (total)':<35} {format_stat('variance_total', stats_original, stats_processed, None)}")
    print(f"{'Inter/Intra Variance Ratio':<35} {format_stat('variance_ratio', stats_original, stats_processed, True)}")

    print("=" * 80)
    print()


def plot_tsne_features(original_features, processed_features, labels, dataset_name, method_name, save_dir='./tsne_plots', rank=0):
    """
    Plot t-SNE visualization of original and processed features.

    Args:
        original_features: Original feature matrix [N, D] (torch tensor or numpy)
        processed_features: Processed feature matrix [N, D'] (torch tensor or numpy)
        labels: Class labels [N] (torch tensor or numpy)
        dataset_name: Name of the dataset
        method_name: Name of the transformation method
        save_dir: Directory to save plots
        rank: Process rank (only rank 0 will plot)
    """
    if rank != 0:
        return

    # Convert to numpy
    if torch.is_tensor(original_features):
        original_features = original_features.cpu().numpy()
    if torch.is_tensor(processed_features):
        processed_features = processed_features.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"Creating t-SNE visualization for {dataset_name}")
    print("=" * 70)

    # Compute statistics
    stats_original = compute_feature_statistics(original_features, labels, "Original")
    stats_processed = compute_feature_statistics(processed_features, labels, "Processed")
    print("Computing feature statistics...")
    print()
    print_feature_statistics(stats_original, stats_processed, method_name, dataset_name)

    # Create t-SNE plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original features
    print(f"Applying t-SNE to original features ({original_features.shape[1]}D -> 2D)...")
    tsne_orig = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
    original_embedded = tsne_orig.fit_transform(original_features)

    scatter1 = ax1.scatter(original_embedded[:, 0], original_embedded[:, 1],
                          c=labels, cmap='tab10', s=10, alpha=0.6)
    ax1.set_title(f'Original Features\n{original_features.shape[1]}D', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter1, ax=ax1, label='Class')

    # Processed features
    print(f"Applying t-SNE to processed features ({processed_features.shape[1]}D -> 2D)...")
    tsne_proc = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
    processed_embedded = tsne_proc.fit_transform(processed_features)

    scatter2 = ax2.scatter(processed_embedded[:, 0], processed_embedded[:, 1],
                          c=labels, cmap='tab10', s=10, alpha=0.6)
    ax2.set_title(f'{method_name}\n{processed_features.shape[1]}D', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter2, ax=ax2, label='Class')

    # Add statistics as text
    stats_text = f"Fisher: {stats_processed['fisher_score_mean']:.4f}\n"
    if stats_processed['silhouette_score'] is not None:
        stats_text += f"Silhouette: {stats_processed['silhouette_score']:.4f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Dataset: {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    safe_method_name = method_name.replace(' ', '_').replace('/', '_')
    save_path = os.path.join(save_dir, f'{dataset_name}_{safe_method_name}_tsne.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ t-SNE plot saved to: {save_path}")
    plt.close()
