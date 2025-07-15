from model import PureGCN_v1, PureGCN, GCN, LightGCN, MLPPredictor, Prodigy_Predictor_mlp
from data import load_data, load_ogbn_data
import argparse
import torch
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import numpy as np

def acc(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    correct = y_true == y_pred
    return float(np.sum(correct)) / len(correct)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=131072)
    parser.add_argument('--test_batch_size', type=int, default=131072)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument("--hidden", default=512, type=int)
    parser.add_argument("--dp", default=0.5, type=float)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--scale', type=bool, default=False)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--model', type=str, default='PureGCN_v1')
    parser.add_argument('--predictor', type=str, default='MLP')
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_res', type=bool, default=False)
    parser.add_argument('--emb', type=bool, default=False)
    parser.add_argument('--pca', type=bool, default=False)
    parser.add_argument('--relu', type=bool, default=False)
    parser.add_argument('--res', type=bool, default=False)
    parser.add_argument('--multilayer', type=bool, default=False)
    parser.add_argument('--use_gin', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seperate', type=bool, default=False)
    parser.add_argument('--degree', type=bool, default=False)

    return parser.parse_args()

def train(model, data, train_idx, optimizer, pred, batch_size, prodigy=False, degree=False):
    st = time.time()

    model.train()
    pred.train()

    dataloader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=True)
    total_loss = 0

    for perm in dataloader:
        train_perm_idx = train_idx[perm]
        h = model(data.x, data.adj_t)
        if prodigy:
            device = h.device
            c = data.y.max().item() + 1
            class_h = torch.zeros(c, h.size(1)).to(device)
            if degree:
                degree = data.adj_t.sum(dim=-1).to(device)
                h_degree = h / degree.view(-1, 1)
            else:
                h_degree = h
            
            h_degree = h_degree[train_idx]
            train_y = data.y[train_idx]

            class_h = torch.scatter_reduce(
                class_h, 0, train_y.view(-1, 1).expand(-1, h_degree.size(1)), h_degree, 
                reduce='mean', include_self=False
            )

            score = pred(h[train_perm_idx], class_h)
        else:
            h_train = h[train_perm_idx]
            score = pred(h_train)
        score = F.log_softmax(score, dim=1)
        label = data.y[train_perm_idx].squeeze()
        loss = F.nll_loss(score, label)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    en = time.time()
    print(f"Train time: {en-st}", flush=True)

    return total_loss / len(dataloader)

@torch.no_grad()
def test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, prodigy=False, 
         degree=False):
    st = time.time()
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    if prodigy:
        device = h.device
        c = data.y.max().item() + 1
        class_h = torch.zeros(c, h.size(1)).to(device)
        
        if degree:
            degree = data.adj_t.sum(dim=-1).to(device)
            h_degree = h / degree.view(-1, 1)
        else:
            h_degree = h
        
        h_degree = h_degree[train_idx]
        train_y = data.y[train_idx]

        class_h = torch.scatter_reduce(
            class_h, 0, train_y.view(-1, 1).expand(-1, h_degree.size(1)), h_degree, reduce='mean'
        )

    # predict
    # break into mini-batches for large edge sets
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)

    valid_score = []
    for idx in valid_loader:
        out = predictor(h[valid_idx[idx]]) if not prodigy else predictor(h[valid_idx[idx]], class_h)
        out = out.argmax(dim=1).flatten()
        valid_score.append(out)
    valid_score = torch.cat(valid_score, dim=0)

    train_score = []
    for idx in train_loader:
        out = predictor(h[train_idx[idx]]) if not prodigy else predictor(h[train_idx[idx]], class_h)
        out = out.argmax(dim=1).flatten()
        train_score.append(out)
    train_score = torch.cat(train_score, dim=0)

    test_score = []
    for idx in test_loader:
        out = predictor(h[test_idx[idx]]) if not prodigy else predictor(h[test_idx[idx]], class_h)
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

    print(f"Test time: {time.time()-st}", flush=True)
    return train_results, valid_results, test_results


def run(args):
    wandb.init(project='inductnode', config=args)

    print(args, flush=True)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.dataset.startswith('ogbn-'):
        data, split_idx = load_ogbn_data(args.dataset)
    else:
        data, split_idx = load_data(args.dataset)

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    if args.emb:
        emb = nn.Embedding(data.num_nodes, args.hidden)
        nn.init.xavier_uniform__(emb.weight)
        data.x = emb.weight

    data.x = data.x.to(device)
    data.adj_t = data.adj_t.to(device)
    data.y = data.y.to(device)

    input_dim = data.x.size(1)

    if args.model == 'PureGCN_v1':
        model = PureGCN_v1(input_dim, args.num_layers, args.hidden, args.dp, args.norm, args.res, 
                           args.relu)
    elif args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    elif args.model == 'GCN':
        model = GCN(input_dim, args.hidden, args.norm, args.relu, args.num_layers, args.dp,
                    args.multilayer, args.use_gin, args.res)
    elif args.model == 'LightGCN':
        model = LightGCN(input_dim, args.hidden, args.num_layers, args.dp, args.relu, args.norm)
    else:
        raise NotImplementedError

    if args.pca:
        U, S, V = torch.pca_lowrank(data.x, q=args.hidden)
        data.x = torch.mm(U, torch.diag(S))

    c = data.y.max().item() + 1
    
    if args.predictor == 'MLP':
        predictor = MLPPredictor(args.hidden, c, args.dp, args.mlp_layers, args.mlp_res, 
                                 args.norm, args.scale)
    elif args.predictor == 'Prodigy':
        predictor = Prodigy_Predictor_mlp(args.hidden, args.dp, args.mlp_layers, args.norm, 
                                          args.scale)
    model = model.to(device)
    predictor = predictor.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), 
                                 lr=args.lr, weight_decay=args.weight_decay)

    st_all = time.time()
    best_valid = 0
    final_test = 0

    for epoch in range(args.epochs):
        st = time.time()
        loss = train(model, data, train_idx, optimizer, predictor, args.batch_size, 
                     args.predictor == 'Prodigy', args.degree)
        train_metric, valid_metric, test_metric = test(model, predictor, data, train_idx, valid_idx, 
                                                       test_idx, args.test_batch_size, 
                                                       args.predictor == 'Prodigy', args.degree)
        wandb.log({'train_loss': loss, 'train_metric': train_metric, 'valid_metric': valid_metric, 
                   'test_metric': test_metric})
        print(f"Epoch {epoch}, Train Metric: {train_metric}", flush=True)
        print(f"Epoch {epoch}, Valid Metric: {valid_metric}", flush=True)
        print(f"Epoch {epoch}, Test Metric: {test_metric}", flush=True)
        print(f"Epoch {epoch} Loss: {loss}", flush=True)
        en = time.time()
        print(f"Epoch time: {en-st}", flush=True)
        if valid_metric > best_valid:
            best_valid = valid_metric
            best_epoch = epoch
            final_test = test_metric

        if epoch - best_epoch >= 200:
            break

    print(f"Total time: {time.time()-st_all}", flush=True)
    wandb.log({'final_test': final_test})
    return final_test

def main():
    args = parse()
    avg_test = 0
    for _ in range(args.runs):
        final_test = run(args)
        avg_test += final_test
    avg_test /= args.runs
    print(f"Average test: {avg_test}")
    wandb.init(project='inductlink')
    wandb.log({'avg_test': avg_test})

if __name__ == '__main__':
    main()
