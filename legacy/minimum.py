from model import PureGCN, Prodigy_Predictor
from data import load_data, load_ogbn_data
import argparse
import torch
from ogb.nodeproppred import Evaluator
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def acc(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    correct = y_true == y_pred
    return {'acc': float(np.sum(correct)) / len(correct)}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--degree', action='store_true', default=False)
    parser.add_argument('--norm', action='store_true', default=False)

    return parser.parse_args()

def visualize_embeddings(embeddings, graph_labels, sample_num=1000, name=''):

    # Convert to numpy for t-SNE
    sample_idx = np.random.choice(embeddings.size(0), sample_num, replace=False)
    embeddings = embeddings.cpu().numpy()
    graph_labels = graph_labels.squeeze().cpu().numpy()
    embeddings = embeddings[sample_idx]
    graph_labels = graph_labels[sample_idx]
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(graph_labels)
    for label in unique_labels:
        mask = (graph_labels == label)
        plt.scatter(x[mask], y[mask], label=label, alpha=0.7)

    # Add plot details
    plt.legend(title='Graph Label')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Embeddings')

    # Save the plot
    plt.savefig('{}_embeddings.png'.format(name))
    plt.close()

def visualize_class_embeddings(class_embeddings):
    """
    Visualize class embeddings using t-SNE.
    
    Parameters:
    - class_embeddings: PyTorch tensor of shape (N, D), where N is the number of classes (e.g., 40)
                        and D is the embedding dimension.
    
    The function reduces the embeddings to 2D using t-SNE, plots them with class indices as labels,
    and saves the plot as 'class_embeddings_tsne.png'.
    """
    # Convert PyTorch tensor to NumPy array
    embeddings_np = class_embeddings.t().cpu().numpy()
    
    # Get the number of classes
    N = embeddings_np.shape[0]
    
    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Extract x and y coordinates
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    
    # Add text labels for each point (class index)
    for i in range(N):
        plt.text(x[i], y[i], str(i), fontsize=9)
    
    # Set title and axis labels
    plt.title('t-SNE Visualization of Class Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('class_embeddings_tsne.png')
    
    # Close the plot to free resources
    plt.close()

def run():
    args = parse()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if args.dataset.startswith('ogbn-'):
        data, split_idx = load_ogbn_data(args.dataset)
    else:
        data, split_idx = load_data(args.dataset)

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    data.x = data.x.to(device)
    data.adj_t = data.adj_t.to(device)
    data.y = data.y.to(device)
    
    # visualize_embeddings(data.x, data.y, 1000, 'raw')

    predictor = Prodigy_Predictor(args.norm)
    
    model = PureGCN(args.num_layers)
    model = model.to(device)
    predictor = predictor.to(device)

    st = time.time()
    h = model(data.x, data.adj_t)

    # visualize_embeddings(h, data.y, 1000, 'gcn')

    c = data.y.max().item() + 1

    class_h = torch.zeros(c, h.size(1)).to(device)

    if args.degree:
        degree = data.adj_t.sum(dim=-1).to(device)
        h_degree = h / degree.view(-1, 1)
    else:
        h_degree = h
    
    h_degree = h_degree[train_idx]
    train_y = data.y[train_idx]

    class_h = torch.scatter_reduce(
        class_h, 0, train_y.view(-1, 1).expand(-1, h_degree.size(1)), h_degree, reduce='mean'
    )
    # class_h = torch.randn(c, h.size(1)).to(device) 
    # # debug
    visualize_class_embeddings(class_h)

    h_train = h[train_idx]
    h_valid = h[valid_idx]
    h_test = h[test_idx]
    train_pred = predictor(h_train, class_h)
    valid_pred = predictor(h_valid, class_h)
    test_pred = predictor(h_test, class_h)

    print(f"Time: {time.time()-st}", flush=True)

    train_pred = train_pred.argmax(dim=-1)
    valid_pred = valid_pred.argmax(dim=-1)
    test_pred = test_pred.argmax(dim=-1)

    print(f"Time: {time.time()-st}", flush=True)

    train_y = data.y[train_idx]
    valid_y = data.y[valid_idx]
    test_y = data.y[test_idx]
    
    train_metric = acc(train_y, train_pred)
    valid_metric = acc(valid_y, valid_pred)
    test_metric = acc(test_y, test_pred)
    print('Train Metric')
    print(train_metric)
    print('Valid Metric')
    print(valid_metric)
    print('Test Metric')
    print(test_metric)
    return train_metric, valid_metric, test_metric

if __name__ == '__main__':
    run()
