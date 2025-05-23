import torch
data = torch.load('dataset/Products/ogbn-products_subset.pt', weights_only=False)
torch.save({
    'x': data.x,
    'adj_t': data.adj_t,
    'y': data.y,
    'train_mask': data.train_mask,
    'val_mask': data.val_mask,
    'test_mask': data.test_mask,
    # Add other relevant attributes like train_mask, etc.
}, 'ogbn-products_subset.pt')