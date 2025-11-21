"""
Mini-batch data loader for large node classification datasets.
Uses NeighborLoader for datasets with large training sets.
"""
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_sparse import SparseTensor
import time
import psutil
from collections import defaultdict


class MiniBatchNCLoader:
    """
    Mini-batch loader for node classification.

    Decides whether to use mini-batch sampling based on training set size.
    Tracks subgraph size, time, and memory usage.
    """

    def __init__(self, data, split_idx, args, device='cuda'):
        """
        Args:
            data: PyG Data object
            split_idx: Dict with 'train', 'valid', 'test' indices
            args: Arguments with mini-batch sampling config
            device: Device to load data to
        """
        self.data = data
        self.split_idx = split_idx
        self.args = args
        self.device = device

        # Initialize tracking (only essential metrics)
        self.stats = {
            'subgraph_sizes': [],  # {'seed_nodes', 'total_nodes', 'total_edges'}
            'time_per_batch': [],  # Total time per batch
            'gpu_memory_mb': []    # GPU memory peak per batch
        }

        # Parse num_neighbors from string
        if isinstance(args.minibatch_num_neighbors, str):
            num_neighbors_list = [int(x.strip()) for x in args.minibatch_num_neighbors.split(',')]
        else:
            num_neighbors_list = args.minibatch_num_neighbors

        # Cap at 5 layers as designed
        num_neighbors_list = num_neighbors_list[:5]

        # Adjust to actual model depth
        num_layers = args.num_layers
        if num_layers > 5:
            # Cap at 5-layer sampling for deep models
            self.num_neighbors = num_neighbors_list[:5]
        else:
            # Use model's actual depth
            self.num_neighbors = num_neighbors_list[:num_layers]

        # Determine mode based on TRAINING SET SIZE (not full dataset)
        num_train = len(split_idx['train'])
        self.use_minibatch = (
            args.use_minibatch_sampling and
            num_train >= args.minibatch_node_threshold
        )

        if self.use_minibatch:
            self.mode = 'minibatch'
            self.batch_size = args.minibatch_batch_size
            self.batches_per_epoch = args.minibatch_batches_per_epoch

            # Create NeighborLoader
            self.loader = self._create_loader()

            print(f"    [MiniBatchLoader] Mini-batch mode:")
            print(f"      Dataset: {data.num_nodes} total nodes, {num_train} train nodes")
            print(f"      Sampling: {len(self.num_neighbors)}-layer {self.num_neighbors}")
            print(f"      Batch size: {self.batch_size}")
            print(f"      Batches per epoch: {self.batches_per_epoch}")

            # Estimate receptive field
            receptive = 1
            for n in self.num_neighbors:
                receptive *= n
            print(f"      Est. receptive field: ~{receptive * self.batch_size:,} nodes/batch")
        else:
            self.mode = 'fullbatch'
            print(f"    [MiniBatchLoader] Full-batch mode: {data.num_nodes} nodes, {num_train} train")

    def _create_loader(self):
        """Create NeighborLoader for mini-batch sampling."""
        loader = NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=self.split_idx['train'],
            shuffle=True,
            num_workers=self.args.minibatch_num_workers,
            persistent_workers=False,  # Disable to prevent memory leaks across runs
        )
        return loader

    def get_batches(self):
        """
        Get batches for training.

        Returns:
            List of batches (for minibatch mode) or [full data] (for fullbatch mode)
        """
        if self.mode == 'minibatch':
            # DEBUG: Check loader memory before sampling
            import gc
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated(self.device) / 1024**2
                print(f"    [DEBUG Loader] Before sampling: GPU {mem_before:.0f} MB, loader id: {id(self.loader)}")
                # Check if loader has internal cache
                if hasattr(self.loader, 'data'):
                    print(f"    [DEBUG Loader] Loader holds data: {self.loader.data.num_nodes} nodes")

            batches = []
            for batch_idx, batch in enumerate(self.loader):
                if batch_idx >= self.batches_per_epoch:
                    break

                # Track subgraph size
                self.stats['subgraph_sizes'].append({
                    'seed_nodes': batch.batch_size,
                    'total_nodes': batch.num_nodes,
                    'total_edges': batch.num_edges
                })

                batches.append(batch.to(self.device))

            # DEBUG: Check loader memory after sampling
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated(self.device) / 1024**2
                print(f"    [DEBUG Loader] After sampling: GPU {mem_after:.0f} MB (+{mem_after - mem_before:.0f} MB)")

            return batches
        else:
            # Full-batch mode: return data as-is
            return [self.data]

    def is_minibatch(self):
        """Check if using mini-batch mode."""
        return self.mode == 'minibatch'

    def reset_stats(self):
        """Reset statistics for new epoch (prevents memory accumulation)."""
        self.stats = {
            'subgraph_sizes': [],
            'time_per_batch': [],
            'gpu_memory_mb': []
        }

    def cleanup(self):
        """Cleanup resources to prevent memory leaks across runs."""
        if self.mode == 'minibatch' and hasattr(self, 'loader'):
            # Recreate loader to release worker memory
            del self.loader
            self.loader = self._create_loader()

    def get_stats_summary(self):
        """Get summary statistics for the loader."""
        if not self.stats['subgraph_sizes']:
            return None

        import numpy as np

        summary = {
            'num_batches': len(self.stats['subgraph_sizes']),
            'subgraph': {
                'avg_seed_nodes': np.mean([s['seed_nodes'] for s in self.stats['subgraph_sizes']]),
                'avg_total_nodes': np.mean([s['total_nodes'] for s in self.stats['subgraph_sizes']]),
                'avg_total_edges': np.mean([s['total_edges'] for s in self.stats['subgraph_sizes']]),
                'avg_expansion': np.mean([s['total_nodes'] / s['seed_nodes'] for s in self.stats['subgraph_sizes']]),
            }
        }

        if self.stats['time_per_batch']:
            summary['time'] = {
                'total_sec': sum(self.stats['time_per_batch']),
                'avg_ms': np.mean(self.stats['time_per_batch']) * 1000
            }

        if self.stats['gpu_memory_mb']:
            summary['gpu_memory_mb'] = {
                'avg': np.mean(self.stats['gpu_memory_mb']),
                'max': max(self.stats['gpu_memory_mb'])
            }

        return summary

    def print_stats(self, dataset_name=''):
        """Print concise statistics summary."""
        summary = self.get_stats_summary()
        if summary is None:
            return

        print(f"\n  [Mini-Batch Stats] {dataset_name}:")
        print(f"    Batches: {summary['num_batches']}")
        print(f"    Subgraph: {summary['subgraph']['avg_seed_nodes']:.0f} seed → "
              f"{summary['subgraph']['avg_total_nodes']:.0f} nodes "
              f"({summary['subgraph']['avg_expansion']:.1f}x), "
              f"{summary['subgraph']['avg_total_edges']:.0f} edges")

        if 'time' in summary:
            print(f"    Time: {summary['time']['avg_ms']:.1f}ms/batch "
                  f"({summary['time']['total_sec']:.2f}s total)")

        if 'gpu_memory_mb' in summary:
            print(f"    GPU Memory: {summary['gpu_memory_mb']['avg']:.0f}MB avg, "
                  f"{summary['gpu_memory_mb']['max']:.0f}MB peak")

    def reset_stats(self):
        """Reset all statistics."""
        for key in self.stats:
            self.stats[key] = []


def compute_nc_loss_with_loader(data_loader, split_idx, model, predictor, args, device,
                                  identity_projection=None, projector=None, external_embeddings=None, optimizer=None):
    """
    Compute node classification loss using MiniBatchNCLoader.
    Performs forward pass, backward pass, and optimizer step for each batch.
    Tracks time and memory for forward/backward passes.

    Args:
        data_loader: MiniBatchNCLoader instance
        split_idx: Train/val/test split indices
        model: GNN model
        predictor: Classification head
        args: Arguments
        device: Device
        identity_projection: Optional identity projection layer
        external_embeddings: Optional external embeddings
        optimizer: Optimizer (for backward pass)

    Returns:
        avg_loss: Average loss across batches (as scalar)
    """
    batches = data_loader.get_batches()
    total_loss = 0.0
    total_nll_loss = 0.0
    total_de_loss = 0.0
    count = 0

    # Track memory at start
    if torch.cuda.is_available():
        mem_start = torch.cuda.memory_allocated(device) / 1024**2
        print(f"    [Memory] Start of training: {mem_start:.0f} MB allocated")

    for batch_idx, batch in enumerate(batches):
        # Start timing and memory tracking
        batch_start = time.time()
        if torch.cuda.is_available() and data_loader.is_minibatch():
            torch.cuda.reset_peak_memory_stats(device)

        if data_loader.is_minibatch():
            # Mini-batch: batch is a sampled subgraph
            if torch.cuda.is_available():
                mem_before_batch = torch.cuda.memory_allocated(device) / 1024**2

            print(f"    Batch {batch_idx+1}/{data_loader.batches_per_epoch}: {batch.batch_size} seed nodes → {batch.num_nodes} total nodes, {batch.num_edges} edges", end="")
            if torch.cuda.is_available():
                print(f" [Mem: {mem_before_batch:.0f} MB]")
            else:
                print()

            x = batch.x
            edge_index = batch.edge_index
            batch_attr = batch.batch if hasattr(batch, 'batch') else None

            # Convert edge_index to SparseTensor (model expects adj_t)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(batch.num_nodes, batch.num_nodes)
            )

            # Apply identity projection if needed
            if identity_projection is not None:
                x = identity_projection(x)

            # Forward through GNN
            h = model(x, adj_t, batch_attr)

            # Only predict on seed nodes (first batch_size nodes in the batch)
            h_seed = h[:batch.batch_size]
            labels = batch.y[:batch.batch_size]

            # Stratified context sampling: k nodes per class from the sampled subgraph
            # Sample from ALL nodes in the batch (not just seed nodes)
            num_context = args.context_num if hasattr(args, 'context_num') else 5
            classes = batch.y.unique()
            context_samples = []

            for c in classes:
                class_mask = (batch.y == c)
                class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze()

                if class_indices.numel() == 0:
                    continue
                if class_indices.dim() == 0:
                    class_indices = class_indices.unsqueeze(0)

                num_to_select = min(num_context, class_indices.size(0))
                selected = torch.randperm(class_indices.size(0), device=device)[:num_to_select]
                context_samples.append(class_indices[selected])

            if context_samples:
                context_indices = torch.cat(context_samples)
            else:
                # Fallback: use first few nodes if no valid context samples
                context_indices = torch.arange(min(5, batch.num_nodes), device=device)
                print(f"Warning: No valid context samples found in batch, using first {len(context_indices)} nodes as fallback")

            # Add context_sample to batch (required by predictor)
            batch.context_sample = context_indices

            context_h = h[context_indices]
            context_y = batch.y[context_indices]

            # Compute class embeddings using scatter_reduce (same as process_node_features)
            # class_h shape: [max_label + 1, hidden_dim]
            num_classes = int(batch.y.max().item() + 1)
            class_h = torch.zeros(num_classes, context_h.size(1), device=device, dtype=context_h.dtype)
            class_h = torch.scatter_reduce(
                class_h, 0, context_y.view(-1, 1).expand(-1, context_h.size(1)), context_h,
                reduce='mean', include_self=False
            )

            # Call predictor (no MoE)
            score, _ = predictor(batch, context_h, h_seed, context_y, class_h)
            score = F.log_softmax(score, dim=1)
            nll_loss = F.nll_loss(score, labels)

            # Extract DE loss if model has DE
            de_loss = 0.0
            if hasattr(model, 'get_de_loss'):
                de_loss_tensor = model.get_de_loss()
                de_loss = de_loss_tensor.item() if isinstance(de_loss_tensor, torch.Tensor) else de_loss_tensor

            loss = nll_loss + de_loss

        else:
            # Full-batch: delegate to original train() function
            from src.engine_nc import train
            data = batch  # batch is actually full data
            train_idx = split_idx['train']

            loss = train(
                model, data, train_idx, optimizer, predictor,
                batch_size=args.nc_batch_size if hasattr(args, 'nc_batch_size') else 1024,
                degree=False,
                att=None, mlp=None,
                orthogonal_push=args.orthogonal_push if hasattr(args, 'orthogonal_push') else 0.0,
                normalize_class_h=args.normalize_class_h if hasattr(args, 'normalize_class_h') else False,
                clip_grad=args.clip_grad if hasattr(args, 'clip_grad') else 1.0,
                projector=projector,
                rank=0,
                epoch=0,  # epoch doesn't matter for training
                identity_projection=identity_projection,
                lambda_=args.lambda_nc if hasattr(args, 'lambda_nc') else 1.0,
                args=args,
                external_embeddings=external_embeddings
            )

            # Handle dict return from train()
            if isinstance(loss, dict):
                total_loss += loss['total']
                total_nll_loss += loss['nll']
                total_de_loss += loss['de']
            else:
                total_loss += loss
                total_nll_loss += loss
                total_de_loss += 0.0
            count += 1
            continue  # Skip the common backward/optimizer code below (already done in train())

        # Backward pass and optimizer step (only for mini-batch)
        if optimizer is not None:
            optimizer.zero_grad()
            (loss * args.lambda_nc).backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad)
                if identity_projection is not None:
                    torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), args.clip_grad)
            optimizer.step()

        # Track time and memory (only for mini-batch)
        if data_loader.is_minibatch():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            data_loader.stats['time_per_batch'].append(batch_time)

            if torch.cuda.is_available():
                mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
                data_loader.stats['gpu_memory_mb'].append(mem_peak)

        total_loss += loss.item()
        total_nll_loss += nll_loss.item()
        total_de_loss += de_loss  # Already a scalar
        count += 1

        # Explicitly delete batch to free memory
        del batch

        # Track memory after batch processing
        if data_loader.is_minibatch() and torch.cuda.is_available():
            mem_after_batch = torch.cuda.memory_allocated(device) / 1024**2
            mem_increase = mem_after_batch - mem_before_batch
            if mem_increase > 100:  # Only print if significant increase
                print(f"      ⚠ Batch {batch_idx+1} leaked {mem_increase:.0f} MB!")

    # Clear cache once after all batches (not per batch)
    if data_loader.is_minibatch() and torch.cuda.is_available():
        mem_before_clear = torch.cuda.memory_allocated(device) / 1024**2
        print(f"    [DEBUG] About to delete batches list (id: {id(batches)}, len: {len(batches)})")

        # Track what's in local scope before deletion
        import sys
        import gc
        local_vars = list(locals().keys())
        print(f"    [DEBUG] Local variables before cleanup: {local_vars}")

        del batches  # Explicitly delete the list
        gc.collect()  # Force garbage collection
        mem_after_del = torch.cuda.memory_allocated(device) / 1024**2
        torch.cuda.empty_cache()
        mem_after_clear = torch.cuda.memory_allocated(device) / 1024**2
        print(f"    [Memory] After training: {mem_before_clear:.0f} MB → {mem_after_del:.0f} MB (after del) → {mem_after_clear:.0f} MB (after cache clear)")
        print(f"    [Memory] Net increase: {mem_after_clear - mem_start:.0f} MB")

        # Track GPU memory right before function returns
        print(f"    [DEBUG] Memory right before function returns: {torch.cuda.memory_allocated(device) / 1024**2:.0f} MB")

    avg_loss = total_loss / count if count > 0 else 0.0
    avg_nll_loss = total_nll_loss / count if count > 0 else 0.0
    avg_de_loss = total_de_loss / count if count > 0 else 0.0

    # Final memory check after return value is computed
    if data_loader.is_minibatch() and torch.cuda.is_available():
        print(f"    [DEBUG] Memory at function exit: {torch.cuda.memory_allocated(device) / 1024**2:.0f} MB")

    # Return dict with breakdown for DE loss tracking
    return {
        'total': avg_loss,
        'nll': avg_nll_loss,
        'de': avg_de_loss
    }


@torch.no_grad()
def evaluate_with_loader(data_loader, split_idx, model, predictor, args, device,
                         eval_split='valid', identity_projection=None, projector=None):
    """
    Evaluate using mini-batch sampling (Option 3: sample neighborhoods for eval nodes only).

    Reuses training sampling configs for consistency.

    Args:
        data_loader: MiniBatchNCLoader instance
        split_idx: Dictionary with 'train', 'valid', 'test' indices
        model: GNN model
        predictor: Predictor model
        args: Arguments
        device: Device
        eval_split: Which split to evaluate ('valid' or 'test')
        identity_projection: Optional identity projection layer

    Returns:
        accuracy: Accuracy on the eval split (as Python float)
    """
    import time

    model.eval()
    predictor.eval()

    eval_idx = split_idx[eval_split]

    if data_loader.is_minibatch():
        # Mini-batch evaluation: create loader for eval nodes only
        eval_start = time.time()

        from torch_geometric.loader import NeighborLoader

        eval_loader = NeighborLoader(
            data_loader.data,
            num_neighbors=data_loader.num_neighbors,
            batch_size=data_loader.batch_size,
            input_nodes=eval_idx,
            shuffle=False,
            num_workers=0  # Use 0 for evaluation
        )

        num_batches = len(eval_loader)

        # Apply max batches limit if specified
        max_batches = args.eval_max_batches if hasattr(args, 'eval_max_batches') else 0
        if max_batches > 0 and num_batches > max_batches:
            actual_batches = max_batches
            print(f"    [MiniBatch Eval] Evaluating {len(eval_idx)} {eval_split} nodes (using {actual_batches}/{num_batches} batches)")
        else:
            actual_batches = num_batches
            print(f"    [MiniBatch Eval] Evaluating {len(eval_idx)} {eval_split} nodes ({actual_batches} batches)")

        all_predictions = []
        all_labels = []

        for batch_idx, batch in enumerate(eval_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            # Show progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == actual_batches:
                print(f"      Progress: {batch_idx + 1}/{actual_batches} batches", flush=True)
            batch = batch.to(device)

            # Convert edge_index to SparseTensor
            adj_t = SparseTensor(
                row=batch.edge_index[0], col=batch.edge_index[1],
                sparse_sizes=(batch.num_nodes, batch.num_nodes)
            )

            x = batch.x

            # Apply projection strategies (same priority as engine_nc.py)
            # Priority: FUG embeddings > identity projection > raw features
            if hasattr(batch, 'uses_fug_embeddings') and batch.uses_fug_embeddings and projector is not None:
                x = projector(x)
            elif identity_projection is not None:
                x = identity_projection(x)

            # Forward pass on subgraph
            h = model(x, adj_t, None)

            # Get embeddings for eval nodes (first batch_size nodes)
            h_eval = h[:batch.batch_size]
            labels_eval = batch.y[:batch.batch_size]

            # Sample context from subgraph (same as training)
            num_context = args.context_num if hasattr(args, 'context_num') else 5
            classes = batch.y.unique()
            context_samples = []

            for c in classes:
                class_mask = (batch.y == c)
                class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze()

                if class_indices.numel() == 0:
                    continue
                if class_indices.dim() == 0:
                    class_indices = class_indices.unsqueeze(0)

                num_to_select = min(num_context, class_indices.size(0))
                selected = torch.randperm(class_indices.size(0), device=device)[:num_to_select]
                context_samples.append(class_indices[selected])

            if context_samples:
                context_indices = torch.cat(context_samples)
            else:
                context_indices = torch.arange(min(5, batch.num_nodes), device=device)

            batch.context_sample = context_indices
            context_h = h[context_indices]
            context_y = batch.y[context_indices]

            # Compute class embeddings
            num_classes = int(batch.y.max().item() + 1)
            class_h = torch.zeros(num_classes, context_h.size(1), device=device, dtype=context_h.dtype)
            class_h = torch.scatter_reduce(
                class_h, 0, context_y.view(-1, 1).expand(-1, context_h.size(1)), context_h,
                reduce='mean', include_self=False
            )

            # Predict
            pred_output = predictor(batch, context_h, h_eval, context_y, class_h)
            if len(pred_output) == 3:  # MoE case
                out, _, _ = pred_output
            else:
                out, _ = pred_output

            predictions = out.argmax(dim=1)

            # Collect results
            all_predictions.append(predictions.cpu())
            all_labels.append(labels_eval.cpu())

        # Compute accuracy
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        accuracy = (all_predictions == all_labels).float().mean().item()

        eval_time = time.time() - eval_start
        print(f"    [MiniBatch Eval] {eval_split} accuracy: {accuracy:.4f} ({eval_time:.2f}s)")
        return accuracy

    else:
        # Full-batch evaluation: fall back to original test() function
        # This will be handled by the caller
        raise NotImplementedError("Full-batch evaluation should use test() function directly")
