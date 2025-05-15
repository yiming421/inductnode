from model import PureGCN_v1, PureGCN, PFNPredictorNodeCls, GCN, AttentionPool, MLP
from data import load_all_data
import argparse
import torch
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import copy
import numpy as np
from utils import process_node_features
from transformers import get_cosine_schedule_with_warmup

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def acc(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    correct = y_true == y_pred
    return float(np.sum(correct)) / len(correct)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--test_dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--test_batch_size', type=int, default=131072)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', type=str, default='none')
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--dp", default=0.5, type=float)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--model', type=str, default='PureGCN_v1')
    parser.add_argument('--predictor', type=str, default='PFN')
    parser.add_argument('--sweep', type=str2bool, default=False)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--gnn_norm_affine', type=str2bool, default=False)
    parser.add_argument('--mlp_norm_affine', type=str2bool, default=False)
    parser.add_argument('--relu', type=str2bool, default=False)
    parser.add_argument('--res', type=str2bool, default=False)
    parser.add_argument('--optimizer', 

    parser.add_argument('--transformer_layers', type=int, default=1)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--context_num', type=int, default=5)
    parser.add_argument('--seperate', type=str2bool, default=False)
    parser.add_argument('--degree', type=str2bool, default=False)
    parser.add_argument('--padding', type=str, default='zero')
    parser.add_argument('--sim', type=str, default='dot')
    parser.add_argument('--att_pool', type=str2bool, default=False)
    parser.add_argument('--mlp_pool', type=str2bool, default=False)
    parser.add_argument('--orthogonal_push', type=float, default=0.0)
    parser.add_argument('--normalize_class_h', type=str2bool, default=False)
    parser.add_argument('--sign_normalize', type=str2bool, default=False)

    parser.add_argument('--use_gin', type=str2bool, default=False)
    parser.add_argument('--multilayer', type=str2bool, default=False)

    return parser.parse_args()

def train(model, data, train_idx, optimizer, pred, batch_size, degree=False, att=None, mlp=None, 
          scheduler=None, orthogonal_push=0.0, normalize_class_h=False):
    st = time.time()

    model.train()
    pred.train()

    dataloader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=True)
    total_loss = 0

    for perm in dataloader:
        train_perm_idx = train_idx[perm]
        h = model(data.x, data.adj_t)

        context_h = h[data.context_sample]
        context_y = data.y[data.context_sample]
        
        class_h = process_node_features(context_h, data, degree_normalize=degree,attention_pool_module=att, 
                                        mlp_module=mlp, normalize=normalize_class_h)

        target_h = h[train_perm_idx]
        score = pred(data, context_h, target_h, context_y, class_h)
        score = F.log_softmax(score, dim=1)
        label = data.y[train_perm_idx].squeeze()

        class_h = data.final_class_h
        class_h = F.normalize(class_h, p=2, dim=1)
        class_matrix = class_h @ class_h.T
        class_matrix = class_matrix - torch.eye(class_matrix.size(0), device=class_matrix.device,
                                                dtype=class_matrix.dtype)
        orthogonal_loss = torch.sum(class_matrix**2)

        loss = F.nll_loss(score, label) + orthogonal_push * orthogonal_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        if att is not None:
            nn.utils.clip_grad_norm_(att.parameters(), 1.0)
        if mlp is not None:
            nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    en = time.time()
    print(f"Train time: {en-st}", flush=True)

    return total_loss / len(dataloader)

def train_all(model, data_list, split_idx_list, optimizer, pred, batch_size, degree=False, att=None,
              mlp=None, scheduler=None, orthogonal_push=0.0, normalize_class_h=False):
    tot_loss = 0
    for data, split_idx in zip(data_list, split_idx_list):
        train_idx = split_idx['train']
        loss = train(model, data, train_idx, optimizer, pred, batch_size, degree, att, mlp, scheduler,
                     orthogonal_push, normalize_class_h)
        print(f"Dataset {data.name} Loss: {loss}", flush=True)
        tot_loss += loss
    return tot_loss / (len(data_list))

@torch.no_grad()
def test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False, 
         att=None, mlp=None, normalize_class_h=False):
    st = time.time()
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    context_h = h[data.context_sample]
    context_y = data.y[data.context_sample]

    class_h = process_node_features(context_h, data, degree_normalize=degree, attention_pool_module=att, 
                                    mlp_module=mlp, normalize=normalize_class_h)

    # class_h = F.normalize(class_h, p=2, dim=1)

    # predict
    # break into mini-batches for large edge sets
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)

    valid_score = []
    for idx in valid_loader:
        target_h = h[valid_idx[idx]]
        out = predictor(data, context_h, target_h, context_y, class_h)
        out = out.argmax(dim=1).flatten()
        valid_score.append(out)
    valid_score = torch.cat(valid_score, dim=0)
    # print(valid_score[:100], flush=True)
    # print(data.y[valid_idx][:100], flush=True)

    train_score = []
    for idx in train_loader:
        target_h = h[train_idx[idx]]
        out = predictor(data, context_h, target_h, context_y, class_h)
        out = out.argmax(dim=1).flatten()
        train_score.append(out)
    train_score = torch.cat(train_score, dim=0)
    # print(train_score[:100], flush=True)
    # print(data.y[train_idx][:100], flush=True)

    test_score = []
    for idx in test_loader:
        target_h = h[test_idx[idx]]
        out = predictor(data, context_h, target_h, context_y, class_h)
        out = out.argmax(dim=1).flatten()
        test_score.append(out)
    test_score = torch.cat(test_score, dim=0)
    # print(test_score[:100], flush=True)
    # print(data.y[test_idx][:100], flush=True)

    # calculate valid metric
    valid_y = data.y[valid_idx]
    valid_results = acc(valid_y, valid_score)
    train_y = data.y[train_idx]
    train_results = acc(train_y, train_score)
    test_y = data.y[test_idx]
    test_results = acc(test_y, test_score)

    print(f"Test time: {time.time()-st}", flush=True)
    return train_results, valid_results, test_results


def test_all(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None, 
             mlp=None, normalize_class_h=False):
    tot_train_metric, tot_valid_metric, tot_test_metric = 1, 1, 1
    for data, split_idx in zip(data_list, split_idx_list):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp,
             normalize_class_h)
        print(f"Dataset {data.name}")
        print(f"Train {train_metric}, Valid {valid_metric}, Test {test_metric}", flush=True)
        tot_train_metric *= train_metric
        tot_valid_metric *= valid_metric
        tot_test_metric *= test_metric
    return tot_train_metric ** (1/(len(data_list))), tot_valid_metric ** (1/(len(data_list))), \
           tot_test_metric ** (1/(len(data_list)))

def test_all_induct(model, predictor, data_list, split_idx_list, batch_size, degree=False, 
                    att=None, mlp=None, normalize_class_h=False):
    train_metric_list, valid_metric_list, test_metric_list = [], [], []
    for data, split_idx in zip(data_list, split_idx_list):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp, 
             normalize_class_h)
        print(f"Dataset {data.name}")
        print(f"Train {train_metric}, Valid {valid_metric}, Test {test_metric}", flush=True)
        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)
    return train_metric_list, valid_metric_list, test_metric_list

def select_k_shot_context(data, k, index):
    # debug:
    classes = data.y.unique()
    context_samples = []
    mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=data.y.device)
    mask[index] = True
    for c in classes:
        class_mask = (data.y == c) & mask

        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze()
        selected_indices = torch.randperm(class_indices.size(0))[:k]
        selected_indices = class_indices[selected_indices]
        context_samples.append(selected_indices)
    context_samples = torch.cat(context_samples)
    return context_samples

def process_data(data, split_idx, hidden, context_num, prop_model, sign_normalize=False):
    device = data.x.device
    if data.x.size(1) < hidden:
        if hidden % data.x.size(1) != 0:
            num_repeats = hidden // data.x.size(1) + 1
        else:
            num_repeats = hidden // data.x.size(1)
        xs = [data.x]
        h = data.x
        for _ in range(num_repeats - 1):
            h = prop_model(h, data.adj_t).detach()
            xs.append(h)
        data.x = torch.cat(xs, dim=1).to(device)
    split_idx['train'] = split_idx['train'].to(device)
    split_idx['valid'] = split_idx['valid'].to(device)
    split_idx['test'] = split_idx['test'].to(device)
    data.context_sample = select_k_shot_context(data, context_num, split_idx['train'])
    data.context_sample = data.context_sample.to(device)

    st = time.time()
    #debug
    U, S, V = torch.svd(data.x)
    U = U[:, :hidden]
    S = S[:hidden]

    # normalize the eigenvectors direction
    if sign_normalize:
        for i in range(hidden):
            feature_vector = U[:, i] * S[i]
            max_idx = torch.argmax(torch.abs(feature_vector))
            if feature_vector[max_idx] < 0:
                U[:, i] = -U[:, i]

    data.x = torch.mm(U, torch.diag(S)).to(device)
    # print(f'name: {data.name}', flush=True)
    # print(f'pca_data_mean: {data.x.mean(dim=0)}', flush=True)
    print(f"PCA time: {time.time()-st}", flush=True)

def run(args):
    if args.sweep:
        wandb.init(project='inductnode')
        config = wandb.config
        for key in config.keys():
            setattr(args, key, config[key])
    else:
        wandb.init(project='inductnode', config=args)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    train_dataset = args.train_dataset.split(',')
    data_list, split_idx_list = load_all_data(train_dataset)
    prop_model = PureGCN(1).to(device)

    for data, split_idx in zip(data_list, split_idx_list):
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        data.y = data.y.to(device)
        process_data(data, split_idx, args.hidden, args.context_num, prop_model, args.sign_normalize)

    att, mlp = None, None

    if args.att_pool:
        att = AttentionPool(args.hidden, args.hidden // 4, args.nhead, args.dp)
        att = att.to(device)
    if args.mlp_pool:
        mlp = MLP(args.hidden, args.hidden, args.hidden, 2, args.dp, args.norm, False, args.mlp_norm_affine)
        mlp = mlp.to(device)

        
    if args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    elif args.model == 'PureGCN_v1':
        model = PureGCN_v1(args.hidden, args.num_layers, args.hidden, args.dp, args.norm, args.res,
                            args.relu, args.gnn_norm_affine)
    elif args.model == 'GCN':
        model = GCN(args.hidden, args.hidden, args.norm, args.relu, args.num_layers, args.dp,
                    args.multilayer, args.use_gin, args.res, args.gnn_norm_affine)
    else:
        raise NotImplementedError

    if args.predictor == 'PFN':
        predictor = PFNPredictorNodeCls(args.hidden, args.nhead, args.transformer_layers, 
                                        args.mlp_layers, args.dp, args.norm, args.seperate, 
                                        args.degree, att, mlp, args.sim, args.padding, 
                                        args.mlp_norm_affine, args.normalize_class_h)
    else:
        raise NotImplementedError

    model = model.to(device)
    predictor = predictor.to(device)

    parameters = list(model.parameters()) + list(predictor.parameters())
    if args.att_pool:
        parameters += list(att.parameters())
    if args.mlp_pool:
        parameters += list(mlp.parameters())

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif args.schedule == 'warmup':
        warmup_steps = args.epochs // 10
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.epochs)
    else:
        scheduler = None

    print(f'number of parameters: {sum(p.numel() for p in parameters)}', flush=True)

    st_all = time.time()
    best_valid = 0
    final_test = 0
    best_pred = None
    best_model = None
    best_epoch = 0

    for epoch in range(args.epochs):
        st = time.time()
        print(f"Epoch {epoch}", flush=True)
        loss = train_all(model, data_list, split_idx_list, optimizer, predictor, args.batch_size,
                         args.degree, att, mlp, scheduler, args.orthogonal_push, args.normalize_class_h)
        train_metric, valid_metric, test_metric = \
        test_all(model, predictor, data_list, split_idx_list, args.test_batch_size, args.degree, 
                 att, mlp, args.normalize_class_h)
        wandb.log({'train_loss': loss, 'train_metric': train_metric, 'valid_metric': valid_metric, 
                   'test_metric': test_metric})
        en = time.time()
        print(f"Epoch time: {en-st}", flush=True)
        if valid_metric >= best_valid:
            best_valid = valid_metric
            best_epoch = epoch
            final_test = test_metric
            best_pred = copy.deepcopy(predictor.state_dict())
            best_model = copy.deepcopy(model.state_dict())

        if epoch - best_epoch >= 200:
            break

    print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9} GB", flush=True)
    print(f"Total time: {time.time()-st_all}", flush=True)
    wandb.log({'final_test': final_test})
    print(f"Best epoch: {best_epoch}", flush=True)

    model.load_state_dict(best_model)
    predictor.load_state_dict(best_pred)
    test_dataset = args.test_dataset.split(',')
    data_list, split_idx_list = load_all_data(test_dataset)
    prop_model = PureGCN(1).to(device)

    for data, split_idx in zip(data_list, split_idx_list):
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.adj_t = data.adj_t.to(device)
        process_data(data, split_idx, args.hidden, args.context_num, prop_model)

    st_total = time.time()
    st = time.time()
    train_metric_list, valid_metric_list, test_metric_list = \
    test_all_induct(model, predictor, data_list, split_idx_list, args.test_batch_size, 
                    args.degree, att, mlp, args.normalize_class_h)
    print(f"Test time: {time.time()-st}", flush=True)
    for i, data in enumerate(data_list):
        print(f"Dataset {data.name}")
        print(f"Train {train_metric_list[i]}")
        print(f"Valid {valid_metric_list[i]}")
        print(f"Test {test_metric_list[i]}")
        wandb.log({f'{data.name}_train_metric': train_metric_list, 
                   f'{data.name}_valid_metric': valid_metric_list[i], 
                   f'{data.name}_test_metric': test_metric_list[i]})
    train_metric = sum(train_metric_list) / len(train_metric_list)
    valid_metric = sum(valid_metric_list) / len(valid_metric_list)
    test_metric = sum(test_metric_list) / len(test_metric_list)
    print(f'induct_train_metric: {train_metric}, induct_valid_metric: {valid_metric}, induct_test_metric: {test_metric}', flush=True)
    wandb.log({'induct_train_metric': train_metric, 'induct_valid_metric': valid_metric, 
               'induct_test_metric': test_metric})
    print(f"Total time: {time.time()-st_total}", flush=True)

    return train_metric, valid_metric, test_metric

def main():
    args = parse()
    avg_train_metric = 0
    avg_valid_metric = 0
    avg_test_metric = 0
    for _ in range(args.runs):
        train_metric, valid_metric, test_metric = run(args)
        avg_train_metric += train_metric
        avg_valid_metric += valid_metric
        avg_test_metric += test_metric
    avg_train_metric /= args.runs
    avg_valid_metric /= args.runs
    avg_test_metric /= args.runs
    print('Average Train Metric')
    print(avg_train_metric)
    print('Average Valid Metric')
    print(avg_valid_metric)
    print('Average Test Metric')
    print(avg_test_metric)
    wandb.init(project='inductlink')
    wandb.log({'avg_train_metric': avg_train_metric, 'avg_valid_metric': avg_valid_metric, 'avg_test_metric': avg_test_metric})

if __name__ == '__main__':
    main()