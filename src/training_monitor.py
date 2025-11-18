"""
Training monitoring utilities for diagnosing instability in graph classification.

This module provides comprehensive logging of:
- Gradient statistics (norms, distributions)
- Loss components breakdown
- Batch statistics
- Model output statistics (logits, probabilities)
- Embedding statistics
- Numerical health checks (NaN, Inf detection)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import warnings


class TrainingMonitor:
    """Monitor training dynamics to diagnose instability."""

    def __init__(self, log_interval=1, detailed=False):
        """
        Args:
            log_interval: Log every N batches (1 = every batch)
            detailed: If True, log detailed statistics (slower but more informative)
        """
        self.log_interval = log_interval
        self.detailed = detailed
        self.batch_count = 0
        self.epoch_stats = defaultdict(list)

    def reset_epoch_stats(self):
        """Reset statistics at the start of each epoch."""
        self.epoch_stats = defaultdict(list)
        self.batch_count = 0

    def check_gradients(self, model, predictor, identity_projection=None, prefix=""):
        """
        Comprehensive gradient monitoring.

        Returns:
            dict: Gradient statistics
        """
        stats = {}

        # Collect all parameters with gradients
        all_params = []
        param_groups = {
            'model': list(model.parameters()),
            'predictor': list(predictor.parameters())
        }
        if identity_projection is not None:
            param_groups['identity_projection'] = list(identity_projection.parameters())

        # Compute norms per module
        for name, params in param_groups.items():
            param_norms = []
            grad_norms = []
            grad_values = []

            for p in params:
                if p.requires_grad:
                    # Parameter norm
                    param_norms.append(p.data.norm(2).item())

                    # Gradient statistics
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2).item()
                        grad_norms.append(grad_norm)

                        if self.detailed:
                            grad_values.extend(p.grad.data.cpu().flatten().numpy())

                        # Check for NaN/Inf
                        if torch.isnan(p.grad).any():
                            warnings.warn(f"NaN detected in {name} gradients!")
                            stats[f'{name}_has_nan'] = True
                        if torch.isinf(p.grad).any():
                            warnings.warn(f"Inf detected in {name} gradients!")
                            stats[f'{name}_has_inf'] = True

            if grad_norms:
                stats[f'{name}_grad_norm_mean'] = np.mean(grad_norms)
                stats[f'{name}_grad_norm_max'] = np.max(grad_norms)
                stats[f'{name}_grad_norm_min'] = np.min(grad_norms)
                stats[f'{name}_grad_norm_std'] = np.std(grad_norms)
                stats[f'{name}_param_norm_mean'] = np.mean(param_norms)

                # Gradient-to-parameter ratio (important for stability)
                grad_to_param = [g/p for g, p in zip(grad_norms, param_norms) if p > 0]
                if grad_to_param:
                    stats[f'{name}_grad_to_param_ratio'] = np.mean(grad_to_param)

                if self.detailed and grad_values:
                    stats[f'{name}_grad_percentile_1'] = np.percentile(grad_values, 1)
                    stats[f'{name}_grad_percentile_99'] = np.percentile(grad_values, 99)

        # Total gradient norm (across all parameters)
        total_norm = 0.0
        for params in param_groups.values():
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        stats['total_grad_norm'] = total_norm

        return stats

    def check_loss_components(self, nll_loss, auxiliary_loss, orthogonal_loss, total_loss):
        """
        Monitor loss component magnitudes.

        Returns:
            dict: Loss component statistics
        """
        stats = {
            'nll_loss': nll_loss.item() if isinstance(nll_loss, torch.Tensor) else nll_loss,
            'auxiliary_loss': auxiliary_loss.item() if isinstance(auxiliary_loss, torch.Tensor) else auxiliary_loss,
            'orthogonal_loss': orthogonal_loss.item() if isinstance(orthogonal_loss, torch.Tensor) else orthogonal_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        }

        # Check for NaN/Inf in losses
        for name, value in stats.items():
            if np.isnan(value):
                warnings.warn(f"NaN detected in {name}!")
                stats[f'{name}_is_nan'] = True
            if np.isinf(value):
                warnings.warn(f"Inf detected in {name}!")
                stats[f'{name}_is_inf'] = True

        # Component ratios
        if stats['total_loss'] > 0:
            stats['nll_loss_ratio'] = stats['nll_loss'] / stats['total_loss']
            if stats['auxiliary_loss'] > 0:
                stats['auxiliary_loss_ratio'] = stats['auxiliary_loss'] / stats['total_loss']
            if stats['orthogonal_loss'] > 0:
                stats['orthogonal_loss_ratio'] = stats['orthogonal_loss'] / stats['total_loss']

        return stats

    def check_batch_statistics(self, batch_data, valid_mask=None, labels=None):
        """
        Monitor batch composition and label distribution.

        Returns:
            dict: Batch statistics
        """
        stats = {
            'batch_size': batch_data.num_graphs if hasattr(batch_data, 'num_graphs') else len(batch_data),
            'num_nodes': batch_data.num_nodes if hasattr(batch_data, 'num_nodes') else batch_data.x.size(0),
            'num_edges': batch_data.edge_index.size(1) if hasattr(batch_data, 'edge_index') else 0,
        }

        # Valid sample statistics
        if valid_mask is not None:
            stats['num_valid_samples'] = valid_mask.sum().item()
            stats['valid_sample_ratio'] = stats['num_valid_samples'] / len(valid_mask)

        # Label distribution
        if labels is not None:
            if valid_mask is not None:
                valid_labels = labels[valid_mask]
            else:
                valid_labels = labels

            if len(valid_labels) > 0:
                unique_labels, counts = torch.unique(valid_labels, return_counts=True)
                stats['num_classes_in_batch'] = len(unique_labels)
                stats['label_imbalance_ratio'] = counts.max().item() / counts.min().item() if len(counts) > 1 else 1.0

                # Most common and rarest classes
                stats['most_common_class'] = unique_labels[counts.argmax()].item()
                stats['most_common_class_count'] = counts.max().item()
                stats['rarest_class'] = unique_labels[counts.argmin()].item()
                stats['rarest_class_count'] = counts.min().item()

        return stats

    def check_predictor_outputs(self, scores, labels=None, valid_mask=None):
        """
        Monitor predictor output statistics (logits, probabilities).

        Returns:
            dict: Predictor output statistics
        """
        stats = {}

        # Logit statistics
        stats['logit_mean'] = scores.mean().item()
        stats['logit_std'] = scores.std().item()
        stats['logit_max'] = scores.max().item()
        stats['logit_min'] = scores.min().item()
        stats['logit_range'] = stats['logit_max'] - stats['logit_min']

        # Check for numerical issues
        if torch.isnan(scores).any():
            warnings.warn("NaN detected in predictor scores!")
            stats['scores_has_nan'] = True
        if torch.isinf(scores).any():
            warnings.warn("Inf detected in predictor scores!")
            stats['scores_has_inf'] = True

        # Softmax probabilities
        with torch.no_grad():
            probs = F.softmax(scores, dim=1)
            stats['prob_mean'] = probs.mean().item()
            stats['prob_std'] = probs.std().item()
            stats['prob_max'] = probs.max().item()
            stats['prob_min'] = probs.min().item()

            # Prediction confidence (max probability per sample)
            max_probs = probs.max(dim=1)[0]
            stats['confidence_mean'] = max_probs.mean().item()
            stats['confidence_std'] = max_probs.std().item()

            # Entropy (measure of prediction uncertainty)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            stats['entropy_mean'] = entropy.mean().item()
            stats['entropy_std'] = entropy.std().item()

        # Accuracy if labels provided
        if labels is not None:
            if valid_mask is not None:
                valid_scores = scores[valid_mask]
                valid_labels = labels[valid_mask]
            else:
                valid_scores = scores
                valid_labels = labels

            if len(valid_labels) > 0:
                predictions = valid_scores.argmax(dim=1)
                accuracy = (predictions == valid_labels).float().mean().item()
                stats['batch_accuracy'] = accuracy

        return stats

    def check_embeddings(self, embeddings, name="embeddings"):
        """
        Monitor embedding statistics.

        Returns:
            dict: Embedding statistics
        """
        stats = {}

        stats[f'{name}_mean'] = embeddings.mean().item()
        stats[f'{name}_std'] = embeddings.std().item()
        stats[f'{name}_norm_mean'] = embeddings.norm(dim=-1).mean().item()
        stats[f'{name}_norm_std'] = embeddings.norm(dim=-1).std().item()

        # Check for numerical issues
        if torch.isnan(embeddings).any():
            warnings.warn(f"NaN detected in {name}!")
            stats[f'{name}_has_nan'] = True
        if torch.isinf(embeddings).any():
            warnings.warn(f"Inf detected in {name}!")
            stats[f'{name}_has_inf'] = True

        # Embedding diversity (cosine similarity statistics)
        if self.detailed and embeddings.size(0) > 1:
            normalized = F.normalize(embeddings, p=2, dim=-1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
            off_diagonal = similarity_matrix[mask]

            stats[f'{name}_cosine_sim_mean'] = off_diagonal.mean().item()
            stats[f'{name}_cosine_sim_std'] = off_diagonal.std().item()
            stats[f'{name}_cosine_sim_max'] = off_diagonal.max().item()
            stats[f'{name}_cosine_sim_min'] = off_diagonal.min().item()

        return stats

    def log_batch_stats(self, stats_dict, task_idx=None, batch_idx=None):
        """
        Log batch statistics (called every batch or every N batches).

        Args:
            stats_dict: Dictionary of statistics to log
            task_idx: Current task index (for multi-task)
            batch_idx: Current batch index
        """
        self.batch_count += 1

        # Only log every N batches
        if self.batch_count % self.log_interval != 0:
            return

        # Store for epoch summary
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.epoch_stats[key].append(value)

        # Format log message
        prefix = ""
        if task_idx is not None:
            prefix += f"Task {task_idx} "
        if batch_idx is not None:
            prefix += f"Batch {batch_idx} "

        # Key metrics to always show
        key_metrics = ['total_loss', 'nll_loss', 'total_grad_norm', 'batch_accuracy', 'batch_size', 'num_valid_samples']
        log_parts = [prefix.strip()]

        for metric in key_metrics:
            if metric in stats_dict:
                value = stats_dict[metric]
                if isinstance(value, float):
                    log_parts.append(f"{metric}={value:.4f}")
                else:
                    log_parts.append(f"{metric}={value}")

        print(" | ".join(log_parts))

        # Warnings for critical issues
        if stats_dict.get('total_grad_norm', 0) > 100:
            print(f"  ⚠️  WARNING: Large gradient norm detected: {stats_dict['total_grad_norm']:.2f}")

        if stats_dict.get('total_grad_norm', 1) < 1e-6:
            print(f"  ⚠️  WARNING: Very small gradient norm: {stats_dict['total_grad_norm']:.2e}")

        if stats_dict.get('logit_range', 0) > 50:
            print(f"  ⚠️  WARNING: Large logit range: {stats_dict['logit_range']:.2f}")

        if stats_dict.get('confidence_mean', 1.0) < 0.4:
            print(f"  ⚠️  WARNING: Low prediction confidence: {stats_dict['confidence_mean']:.3f}")

    def print_epoch_summary(self, epoch):
        """
        Print summary statistics for the epoch.

        Args:
            epoch: Current epoch number
        """
        if not self.epoch_stats:
            return

        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*80}")

        # Group statistics by category
        categories = {
            'Loss': ['total_loss', 'nll_loss', 'auxiliary_loss', 'orthogonal_loss'],
            'Gradients': ['total_grad_norm', 'model_grad_norm_mean', 'predictor_grad_norm_mean'],
            'Predictions': ['batch_accuracy', 'confidence_mean', 'entropy_mean'],
            'Logits': ['logit_mean', 'logit_std', 'logit_range'],
            'Batch': ['batch_size', 'num_valid_samples', 'valid_sample_ratio'],
        }

        for category, metrics in categories.items():
            category_stats = {}
            for metric in metrics:
                if metric in self.epoch_stats:
                    values = self.epoch_stats[metric]
                    category_stats[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            if category_stats:
                print(f"\n{category}:")
                for metric, stats in category_stats.items():
                    print(f"  {metric:30s}: mean={stats['mean']:8.4f} ± {stats['std']:7.4f} "
                          f"[{stats['min']:7.4f}, {stats['max']:7.4f}]")

        print(f"{'='*80}\n")


def create_monitoring_wrapper(train_fn):
    """
    Decorator to add monitoring to training functions.

    Usage:
        @create_monitoring_wrapper
        def train_graph_classification_single_task(...):
            ...
    """
    def wrapper(*args, monitor=None, **kwargs):
        # If monitor provided, use it; otherwise create default
        if monitor is None:
            monitor = TrainingMonitor(log_interval=10, detailed=False)

        # Call original function with monitoring
        return train_fn(*args, monitor=monitor, **kwargs)

    return wrapper
