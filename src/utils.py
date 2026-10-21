import torch
import torch.nn as nn

def process_node_features(context_h: torch.Tensor, data: object, degree_normalize: bool=False, attention_pool_module: nn.Module=None, mlp_module: nn.Module=None, normalize: bool=False) -> torch.Tensor:
    dev = context_h.device
    num_classes = int(data.y.max().item() + 1)
    context_y = data.y[data.context_sample]
    pooled_class_h = torch.zeros(num_classes, context_h.size(1), device=dev, dtype=context_h.dtype)
    pooled_class_h = torch.scatter_reduce(pooled_class_h, 0, context_y.view(-1, 1).expand(-1, context_h.size(1)), context_h, reduce='mean', include_self=False)
    final_class_h = pooled_class_h
    if normalize:
        final_class_h = nn.functional.normalize(final_class_h, p=2, dim=-1)
    return final_class_h

def acc(y_true, y_pred):
    import numpy as np
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    correct = y_true == y_pred
    return float(np.sum(correct)) / len(correct)
