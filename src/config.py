import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='ogbn-arxiv,CS,Physics,Computers,Photo,Flickr,USA,Brazil,Europe,Wiki,BlogCatalog,DBLP,FacebookPagePage,Reddit')
    parser.add_argument('--test_dataset', type=str, default='Cora,Citeseer,Pubmed,WikiCS')
    parser.add_argument('--gpu', type=str, default='auto',
                        help='GPU specification: "auto" for all GPUs, single GPU ID (e.g., "0"), or comma-separated list (e.g., "0,1,2,3")')
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--test_batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.0000915)
    parser.add_argument('--weight_decay', type=float, default=0.0000087)
    parser.add_argument('--schedule', type=str, default='warmup')
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--dp", default=0.31874, type=float)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--model', type=str, default='PureGCN_v1')
    parser.add_argument('--predictor', type=str, default='PFN')
    parser.add_argument('--sweep', type=str2bool, default=False)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--gnn_norm_affine', type=str2bool, default=False)
    parser.add_argument('--mlp_norm_affine', type=str2bool, default=True)
    parser.add_argument('--relu', type=str2bool, default=False)
    parser.add_argument('--res', type=str2bool, default=False)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--port', type=int, default=12355)
    parser.add_argument('--single_gpu', type=str2bool, default=False,
                        help='Use single GPU mode instead of distributed training')

    parser.add_argument('--transformer_layers', type=int, default=3)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--context_num', type=int, default=20)
    parser.add_argument('--seperate', type=str2bool, default=True)
    parser.add_argument('--degree', type=str2bool, default=False)
    parser.add_argument('--padding', type=str, default='zero', 
                        choices=['zero', 'mlp'], 
                        help='Padding method for PFN predictor: zero padding or MLP padding')
    parser.add_argument('--sim', type=str, default='dot')
    parser.add_argument('--att_pool', type=str2bool, default=False)
    parser.add_argument('--mlp_pool', type=str2bool, default=False)
    parser.add_argument('--orthogonal_push', type=float, default=0.001)
    parser.add_argument('--normalize_class_h', type=str2bool, default=True)
    parser.add_argument('--sign_normalize', type=str2bool, default=False)
    parser.add_argument('--use_full_pca', type=str2bool, default=False)
    parser.add_argument('--normalize_data', type=str2bool, default=False)   

    parser.add_argument('--use_gin', type=str2bool, default=False)
    parser.add_argument('--multilayer', type=str2bool, default=True)

    # Learnable projector arguments
    parser.add_argument('--use_projector', type=str2bool, default=False)
    parser.add_argument('--min_pca_dim', type=int, default=64)
    parser.add_argument('--skip_datasets', type=str2bool, default=False)
    parser.add_argument('--padding_strategy', type=str, default='random', 
                        choices=['zero', 'random', 'repeat'], 
                        help='Padding strategy: zero, random, or repeat')
    parser.add_argument('--use_batchnorm', type=str2bool, default=True,
                        help='Use BatchNorm instead of LayerNorm for normalization')    
    parser.add_argument('--use_identity_projection', type=str2bool, default=True,
                        help='Use identity-preserving projection (small_dim -> large_dim)')
    parser.add_argument('--projection_small_dim', type=int, default=128,
                        help='Small dimension for identity projection (PCA target)')
    parser.add_argument('--projection_large_dim', type=int, default=512,
                        help='Large dimension for identity projection (final size)')
    
    # Checkpointing arguments
    parser.add_argument('--save_checkpoint', type=str2bool, default=False,
                        help='Save checkpoint after training')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Custom checkpoint name (default: auto-generated)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load (for resuming training)')

    # Safe learning rate arguments
    parser.add_argument('--safe_transformer_layers', type=int, default=5,
                        help='Maximum number of transformer layers for safe learning rate')
    parser.add_argument('--safe_lr', type=float, default=0.00014,
                        help='Safe learning rate for deep transformers')
    return parser

def parse():
    parser = get_main_parser()
    return parser.parse_args()

def parse_link_prediction_args():
    """Parse arguments for link prediction."""
    parser = argparse.ArgumentParser(description='Link Prediction using PFN')
    
    # Model loading / creation
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False,
                        help='Whether to load a pretrained model')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to a trained model checkpoint to load')

    # Model architecture (used if not loading a pretrained model)
    parser.add_argument('--model', type=str, default='PureGCN_v1', help='GNN model type (e.g., PureGCN_v1, GCN)')
    parser.add_argument("--hidden", default=128, type=int, help='Hidden dimension size')
    parser.add_argument("--num_layers", default=4, type=int, help='Number of GNN layers')
    parser.add_argument("--dp", default=0.2, type=float, help='Dropout rate')
    parser.add_argument('--norm', type=str2bool, default=True, help='Use normalization in GNN')
    parser.add_argument('--res', type=str2bool, default=False, help='Use residual connections in GNN')
    parser.add_argument('--relu', type=str2bool, default=False, help='Use ReLU activation in GNN')
    parser.add_argument('--gnn_norm_affine', type=str2bool, default=True, help='Learnable affine parameters in GNN norm')
    parser.add_argument('--multilayer', type=str2bool, default=True, help='For GCN model: use multilayer structure')
    parser.add_argument('--use_gin', type=str2bool, default=False, help='For GCN model: use GIN variant')

    # PFN-specific architecture for prototype generation and prediction
    parser.add_argument('--predictor', type=str, default='PFN', help='Predictor type (only PFN supported)')
    parser.add_argument('--att_pool', type=str2bool, default=False, help='Use attention pooling to create prototypes')
    parser.add_argument('--mlp_pool', type=str2bool, default=False, help='Use MLP on top of pooled prototypes')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of layers in MLP pool')
    parser.add_argument('--mlp_norm_affine', type=str2bool, default=True, help='Learnable affine parameters in MLP norm')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads for attention pooling and PFN predictor')
    parser.add_argument('--transformer_layers', type=int, default=1, help='Number of transformer layers in PFN predictor')
    parser.add_argument('--seperate', type=str2bool, default=True, help='For PFN predictor')
    parser.add_argument('--degree', type=str2bool, default=False, help='For PFN predictor (not typically used for links)')
    parser.add_argument('--padding', type=str, default='zero', 
                        choices=['zero', 'mlp'], 
                        help='Padding method for PFN predictor: zero/zeros padding or MLP padding')
    
    # Data arguments
    parser.add_argument('--train_dataset', type=str, default='ogbn-arxiv,CS,Physics,Computers,Photo,Flickr,USA,Brazil,Europe,Wiki,BlogCatalog,DBLP,FacebookPage',
                        help='Comma-separated list of datasets for training link prediction model.')
    parser.add_argument('--test_dataset', type=str, default='Cora,Citeseer,Pubmed,ogbl-collab,ogbl-citation2',
                        help='Comma-separated list of datasets for inductive testing.')
    parser.add_argument('--context_neg_ratio', type=int, default=1,
                        help='Ratio of negative to positive samples for context')
    parser.add_argument('--train_neg_ratio', type=int, default=1,
                        help='Ratio of negative to positive samples for training')
    parser.add_argument('--sign_normalize', type=str2bool, default=False, help='For PCA: normalize eigenvectors direction')
    parser.add_argument('--use_full_pca', type=str2bool, default=False, help='For PCA: use full SVD')
    parser.add_argument('--normalize_data', type=str2bool, default=False, help='Normalize data features after preprocessing')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=4096,
                        help='Test batch size')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('--schedule', type=str, default='none', choices=['none', 'cosine', 'step', 'warmup'], help='Learning rate schedule')
    parser.add_argument('--orthogonal_push', type=float, default=0.0, help='Orthogonal push loss weight for prototypes')
    parser.add_argument('--normalize_class_h', type=str2bool, default=True, help='Normalize prototype embeddings')
    
    # Link prediction specific
    parser.add_argument('--sim', type=str, default='dot',
                        choices=['dot', 'cos', 'mlp'],
                        help='Similarity function for PFN predictor')
    parser.add_argument('--context_k', type=int, default=32,
                        help='Number of positive/negative samples for link context')
    parser.add_argument('--remove_context_from_train', type=str2bool, default=False,
                        help='If True, context samples are removed from the training set.')
    parser.add_argument('--mask_target_edges', type=str2bool, default=False,
                        help='If True, target edges are masked from the graph during message passing.')
    
    # System arguments
    parser.add_argument('--gpu', type=str, default='auto',
                        help='GPU specification: "auto" for all GPUs, single GPU ID (e.g., "0"), or comma-separated list (e.g., "0,1,2,3")')
    parser.add_argument('--use_ddp', type=str2bool, default=True,
                        help='Use distributed training')
    parser.add_argument('--port', type=int, default=12356,
                        help='Port for DDP')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs')
    parser.add_argument('--sweep', type=str2bool, default=False,
                        help='Use sweep for link prediction')
    
    parser.add_argument('--use_projector', type=str2bool, default=False,
                        help='Use projector for link prediction')
    parser.add_argument('--min_pca_dim', type=int, default=32,
                        help='Minimum PCA dimension')
    parser.add_argument('--padding_strategy', type=str, default='zero',
                        help='Padding strategy for link prediction')
    parser.add_argument('--use_batchnorm', type=str2bool, default=False,
                        help='Use batch normalization in GNN')
    parser.add_argument('--use_identity_projection', type=str2bool, default=False,
                        help='Use identity projection for link prediction')
    parser.add_argument('--projection_small_dim', type=int, default=128,
                        help='Small dimension for identity projection')
    parser.add_argument('--projection_large_dim', type=int, default=512,
                        help='Large dimension for identity projection')
    
    # Checkpointing arguments
    parser.add_argument('--save_checkpoint', type=str2bool, default=False,
                        help='Save checkpoint after training')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Custom checkpoint name (default: auto-generated)')

    # Safe learning rate arguments
    parser.add_argument('--safe_transformer_layers', type=int, default=5,
                        help='Maximum number of transformer layers for safe learning rate')
    parser.add_argument('--safe_lr', type=float, default=0.0001,
                        help='Safe learning rate for deep transformers')
    
    return parser.parse_args()
