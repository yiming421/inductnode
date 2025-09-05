import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import psutil
import os
from .utils import process_node_features, acc, apply_final_pca

def train(model, data, train_idx, optimizer, pred, batch_size, degree=False, att=None, mlp=None, 
          orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None, rank=0, epoch=0, 
          identity_projection=None, lambda_=1.0):
    st = time.time()
    print(f"[RANK {rank}] Starting training epoch {epoch} on device cuda:{rank}", flush=True)
    
    # Show detailed memory breakdown for first epoch and every 10th epoch
    show_detailed = (epoch == 0 or epoch % 10 == 0)

    model.train()
    if att is not None:
        att.train()
    if mlp is not None:
        mlp.train()
    if projector is not None:
        projector.train()

    # Use distributed sampler for DDP
    if dist.is_initialized():
        indices = torch.arange(train_idx.size(0))
        sampler = DistributedSampler(indices, shuffle=True)
        sampler.set_epoch(epoch)  # Set epoch for proper shuffling
        dataloader = DataLoader(indices, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=True)

    total_loss = 0
    for batch_idx, perm in enumerate(dataloader):
        if isinstance(perm, torch.Tensor):
            perm = perm.tolist()
        train_perm_idx = train_idx[perm]
        
        # Apply different projection strategies
        if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
            x_input = identity_projection(data.x)
        elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
            projected_features = projector(data.x)
            # Apply final PCA to get features in proper PCA form
            if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
                x_input = apply_final_pca(projected_features, projected_features.size(1))
            else:
                x_input = projected_features
        else:
            x_input = data.x
        
        # Memory-optimized forward pass with gradient checkpointing and chunking
        if hasattr(data, 'use_gradient_checkpointing') and data.use_gradient_checkpointing:
            # Use gradient checkpointing to reduce memory
            h = torch.utils.checkpoint.checkpoint(model, x_input, data.adj_t)
        else:
            # Standard forward pass
            h = model(x_input, data.adj_t)

        # Memory optimization: Extract needed embeddings immediately and delete large tensor
        context_h = h[data.context_sample]
        context_y = data.y[data.context_sample]
        
        # Extract target embeddings for this batch
        target_h = h[train_perm_idx]
        
        # Immediately delete the large tensor to free memory
        del h
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Fix type safety by properly handling None values
        class_h = process_node_features(
            context_h, data, 
            degree_normalize=degree,
            attention_pool_module=att if att is not None else None, 
            mlp_module=mlp if mlp is not None else None, 
            normalize=normalize_class_h
        )

        target_y = data.y[train_perm_idx]
        score, class_h = pred(data, context_h, target_h, context_y, class_h)
        score = F.log_softmax(score, dim=1)
        label = data.y[train_perm_idx].squeeze()

        # Compute orthogonal loss with better numerical stability
        if orthogonal_push > 0:
            class_h_norm = F.normalize(class_h, p=2, dim=1)
            class_matrix = class_h_norm @ class_h_norm.T
            # Remove diagonal elements
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=label.device)
        
        nll_loss = F.nll_loss(score, label)
        loss = nll_loss + orthogonal_push * orthogonal_loss
        loss = loss * lambda_  # Apply lambda scaling

        # Only perform optimization if optimizer is provided (for joint training compatibility)
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        if clip_grad > 0:
            # Handle DDP wrapped models
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            nn.utils.clip_grad_norm_(pred.parameters(), clip_grad)
            if att is not None:
                nn.utils.clip_grad_norm_(att.parameters(), clip_grad)
            if mlp is not None:
                nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad)
            if projector is not None:
                nn.utils.clip_grad_norm_(projector.parameters(), clip_grad)
            if identity_projection is not None:
                nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)
        
        optimizer.step()
        
        # Memory cleanup after optimization step
        total_loss += loss.item()
        del context_h, target_h, class_h, score, label, loss, nll_loss, orthogonal_loss
        if 'x_input' in locals() and x_input is not data.x:
            del x_input
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    en = time.time()
    if rank == 0:
        print(f"Train time: {en-st:.2f}s", flush=True)
    
    loss_str = f"{total_loss / len(dataloader):.4f}" if optimizer is not None else "tensor"
    print(f"[RANK {rank}] Completed training epoch {epoch}, loss: {loss_str}", flush=True)
    
    return total_loss / len(dataloader)  # Return scalar for normal training

def train_all(model, data_list, split_idx_list, optimizer, pred, batch_size, degree=False, att=None,
              mlp=None, orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None, 
              rank=0, epoch=0, identity_projection=None, lambda_=1.0):
    tot_loss = 0
    for data, split_idx in zip(data_list, split_idx_list):
        train_idx = split_idx['train']
        loss = train(model, data, train_idx, optimizer, pred, batch_size, degree, att, mlp,
                     orthogonal_push, normalize_class_h, clip_grad, projector, rank, epoch, 
                     identity_projection, lambda_)
        if rank == 0:
            print(f"Dataset {data.name} Loss: {loss}", flush=True)
        tot_loss += loss
    return tot_loss / (len(data_list))

@torch.no_grad()
def test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False, 
         att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None):
    st = time.time()
    model.eval()
    predictor.eval()
    if projector is not None:
        projector.eval()
    if identity_projection is not None:
        identity_projection.eval()

    # Apply different projection strategies
    if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
        # Apply identity projection
        x_input = identity_projection(data.x)
    elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
        projected_features = projector(data.x)
        # Apply final PCA to get features in proper PCA form
        if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
            x_input = apply_final_pca(projected_features, projected_features.size(1))
        else:
            x_input = projected_features
    else:
        x_input = data.x

    h = model(x_input, data.adj_t)

    context_h = h[data.context_sample]
    context_y = data.y[data.context_sample]

    class_h = process_node_features(context_h, data, degree_normalize=degree, attention_pool_module=att, 
                                    mlp_module=mlp, normalize=normalize_class_h)

    # predict
    # break into mini-batches for large edge sets
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)

    valid_score = []
    for idx in valid_loader:
        target_h = h[valid_idx[idx]]
        out, _ = predictor(data, context_h, target_h, context_y, class_h)
        out = out.argmax(dim=1).flatten()
        valid_score.append(out)
    valid_score = torch.cat(valid_score, dim=0)

    train_score = []
    for idx in train_loader:
        target_h = h[train_idx[idx]]
        out, _ = predictor(data, context_h, target_h, context_y, class_h)
        out = out.argmax(dim=1).flatten()
        train_score.append(out)
    train_score = torch.cat(train_score, dim=0)

    test_score = []
    for idx in test_loader:
        target_h = h[test_idx[idx]]
        out, _ = predictor(data, context_h, target_h, context_y, class_h)
        out = out.argmax(dim=1).flatten()
        test_score.append(out)
    test_score = torch.cat(test_score, dim=0)

    # calculate valid metric
    valid_y = data.y[valid_idx]
    valid_results = acc(valid_y, valid_score)
    train_y = data.y[train_idx]
    train_results = acc(train_y, train_score)
    test_y = data.y[test_idx]
    test_results = acc(test_y, test_score)

    if rank == 0:
        print(f"Test time: {time.time()-st}", flush=True)
    return train_results, valid_results, test_results

def test_all(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None, 
             mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None):
    tot_train_metric, tot_valid_metric, tot_test_metric = 1, 1, 1
    for data, split_idx in zip(data_list, split_idx_list):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp,
             normalize_class_h, projector, rank, identity_projection)
        if rank == 0:
            print(f"Dataset {data.name}")
            print(f"Train {train_metric}, Valid {valid_metric}, Test {test_metric}", flush=True)
        tot_train_metric *= train_metric
        tot_valid_metric *= valid_metric
        tot_test_metric *= test_metric
    return tot_train_metric ** (1/(len(data_list))), tot_valid_metric ** (1/(len(data_list))), \
           tot_test_metric ** (1/(len(data_list)))

def test_all_induct(model, predictor, data_list, split_idx_list, batch_size, degree=False, 
                    att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None):
    train_metric_list, valid_metric_list, test_metric_list = [], [], []
    for data, split_idx in zip(data_list, split_idx_list):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp, 
             normalize_class_h, projector, rank, identity_projection)
        if rank == 0:
            print(f"Dataset {data.name}")
            print(f"Train {train_metric}, Valid {valid_metric}, Test {test_metric}", flush=True)
        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)
    return train_metric_list, valid_metric_list, test_metric_list 