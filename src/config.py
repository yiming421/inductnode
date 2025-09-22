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
    parser.add_argument('--normalize_data', type=str2bool, default=True)   

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
    parser.add_argument('--safe_lr', type=float, default=0.000014,
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
    parser.add_argument('--test_dataset', type=str, default='Cora,Citeseer,Pubmed,ogbl-collab',
                        help='Comma-separated list of datasets for inductive testing.')
    parser.add_argument('--context_neg_ratio', type=int, default=1,
                        help='Ratio of negative to positive samples for context')
    parser.add_argument('--train_neg_ratio', type=int, default=1,
                        help='Ratio of negative to positive samples for training')
    parser.add_argument('--sign_normalize', type=str2bool, default=False, help='For PCA: normalize eigenvectors direction')
    parser.add_argument('--use_full_pca', type=str2bool, default=False, help='For PCA: use full SVD')
    parser.add_argument('--normalize_data', type=str2bool, default=False, help='Normalize data features after preprocessing')
    parser.add_argument('--use_test_split_for_pretraining', type=str2bool, default=False, help='Use test split for pretraining to maximize data utilization')
    
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
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU specification: "auto" for all GPUs, single GPU ID (e.g., "0"), or comma-separated list (e.g., "0,1,2,3")')
    parser.add_argument('--use_ddp', type=str2bool, default=False,
                        help='Use distributed training')
    parser.add_argument('--port', type=int, default=12356,
                        help='Port for DDP')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs')
    parser.add_argument('--sweep', type=str2bool, default=False,
                        help='Use sweep for link prediction')
    
    # Logging arguments
    parser.add_argument('--log_level', type=str, default='VERBOSE', 
                        choices=['QUIET', 'INFO', 'DEBUG', 'VERBOSE'],
                        help='Log level: QUIET (minimal), INFO (standard), DEBUG (detailed), VERBOSE (everything)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Interval for printing training progress (epochs)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Interval for validation evaluation (epochs)')
    parser.add_argument('--analysis_interval', type=int, default=1000,
                        help='Interval for detailed waste analysis (epochs)')
    
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
    parser.add_argument('--projection_large_dim', type=int, default=256,
                        help='Large dimension for identity projection')
    
    # Checkpointing arguments
    parser.add_argument('--save_checkpoint', type=str2bool, default=False,
                        help='Save checkpoint after training')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Custom checkpoint name (default: auto-generated)')
    
    return parser.parse_args()

def parse_joint_training_args():
    """
    Parse command line arguments for joint training.
    Combines arguments from both node classification and link prediction scripts.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Joint Training for Node Classification and Link Prediction')
    
    # === Basic Configuration ===
    parser.add_argument('--runs', type=int, default=3, help='Number of training runs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--gpu', type=str, default='0', help='GPU specification: "auto" for all GPUs, single GPU ID (e.g., "0"), or comma-separated list (e.g., "0,1,2,3")')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', type=int, default=12355, help='Port for DDP')
    parser.add_argument('--single_gpu', type=str2bool, default=True, help='Use single GPU mode instead of distributed training')
    
    # === Model Architecture ===
    parser.add_argument('--model', type=str, default='PureGCN_v1', choices=['PureGCN_v1', 'GCN', 'UnifiedGNN'])
    parser.add_argument('--predictor', type=str, default='PFN', choices=['PFN'])
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--transformer_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of MLP layers')
    parser.add_argument('--dp', type=float, default=0.08717780667044389, help='Dropout rate')
    parser.add_argument('--norm', type=str2bool, default=True, help='Use normalization')
    parser.add_argument('--res', type=str2bool, default=False, help='Use residual connections')
    parser.add_argument('--relu', type=str2bool, default=False, help='Use ReLU activation')
    parser.add_argument('--gnn_norm_affine', type=str2bool, default=True, help='Learnable affine parameters in GNN norm')
    parser.add_argument('--mlp_norm_affine', type=str2bool, default=True, help='Learnable affine parameters in MLP norm')
    parser.add_argument('--multilayer', type=str2bool, default=True, help='Use multilayer structure for GCN')
    parser.add_argument('--use_gin', type=str2bool, default=False, help='Use GIN variant for GCN')

    # === UnifiedGNN Specific Arguments ===
    parser.add_argument('--unified_model_type', type=str, default='gcn', choices=['gcn', 'lightgcn', 'puregcn'],
                       help='UnifiedGNN model type')
    parser.add_argument('--conv_type', type=str, default='GCN', choices=['GCN', 'SAGE', 'GAT', 'GIN'],
                       help='Convolution type for UnifiedGNN')
    parser.add_argument('--residual', type=float, default=1.0, help='Residual connection strength')
    parser.add_argument('--linear', type=str2bool, default=False, help='Apply linear transformation after conv')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for LightGCN')
    parser.add_argument('--exp', type=str2bool, default=False, help='Use exponential alpha weights')
    parser.add_argument('--gin_aggr', type=str, default='sum', choices=['sum', 'mean'], help='Aggregation for GIN')
    parser.add_argument('--supports_edge_weight', type=str2bool, default=False, help='Whether model supports edge weights')
    parser.add_argument('--no_parameters', type=str2bool, default=False, help='Use parameter-free convolutions')
    parser.add_argument('--input_norm', type=str2bool, default=False, help='Apply input normalization')
    
    # === Training Configuration ===
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.000001995432810684568, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00041043056600071503, help='Weight decay')
    parser.add_argument('--schedule', type=str, default='step', choices=['cosine', 'step', 'warmup', 'none'])
    parser.add_argument('--nc_batch_size', type=int, default=8192, help='Node classification batch size')
    parser.add_argument('--lp_batch_size', type=int, default=32768, help='Link prediction batch size')
    parser.add_argument('--test_batch_size', type=int, default=16384, help='Test batch size')
    parser.add_argument('--clip_grad', type=float, default=0.9893307302161596, help='Gradient clipping')
    
    # === Joint Training Specific ===
    parser.add_argument('--enable_nc', type=str2bool, default=True, help='Enable node classification task')
    parser.add_argument('--enable_lp', type=str2bool, default=True, help='Enable link prediction task')
    parser.add_argument('--enable_gc', type=str2bool, default=True, help='Enable graph classification task')
    parser.add_argument('--lambda_nc', type=float, default=0.5339754552414909, help='Weight for node classification loss')
    parser.add_argument('--lambda_lp', type=float, default=2.735303979230086, help='Weight for link prediction loss')
    
    # === Dataset Configuration ===
    parser.add_argument('--nc_train_dataset', type=str, default='ogbn-arxiv,CS,Physics,Computers,Photo,Flickr,USA,Brazil,Europe,Wiki,BlogCatalog,DBLP,FacebookPagePage', 
                       help='Node classification training datasets')
    parser.add_argument('--nc_test_dataset', type=str, default='Cora,Citeseer,Pubmed,WikiCS', 
                       help='Node classification test datasets')
    parser.add_argument('--lp_train_dataset', type=str, default='CS,Physics,Computers,Photo,Flickr,Wiki,BlogCatalog,FacebookPage', 
                       help='Link prediction training datasets')
    parser.add_argument('--lp_test_dataset', type=str, default='Cora,Citeseer,Pubmed,ogbl-collab', 
                       help='Link prediction test datasets')
    
    # === Model Components ===
    parser.add_argument('--use_identity_projection', type=str2bool, default=True, help='Use identity projection')
    
    # === PFN Predictor Configuration ===
    parser.add_argument('--context_num', type=int, default=20, help='Number of context nodes')
    parser.add_argument('--seperate', type=str2bool, default=True, help='Separate processing in PFN predictor')
    parser.add_argument('--padding', type=str, default='zero', choices=['zero', 'mlp'], help='Padding method for PFN predictor')
    parser.add_argument('--sim', type=str, default='dot', choices=['dot', 'cos', 'mlp'], help='Similarity function')
    parser.add_argument('--orthogonal_push', type=float, default=0, help='Orthogonal push regularization weight')
    parser.add_argument('--normalize_class_h', type=str2bool, default=True, help='Normalize class embeddings')
    
    # === Data Processing ===
    parser.add_argument('--use_full_pca', type=str2bool, default=False, help='Use full PCA decomposition')
    parser.add_argument('--pca_device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to perform PCA computation (cpu=Incremental PCA, gpu=torch.pca_lowrank)')
    parser.add_argument('--incremental_pca_batch_size', type=int, default=10000, help='Batch size for CPU Incremental PCA')
    parser.add_argument('--pca_sample_threshold', type=int, default=100000, help='Threshold for using sampled PCA on GPU')
    parser.add_argument('--use_pca_cache', type=str2bool, default=True, help='Enable PCA result caching for faster subsequent runs')
    parser.add_argument('--pca_cache_dir', type=str, default='./pca_cache', help='Directory to store PCA cache files')
    parser.add_argument('--normalize_data', type=str2bool, default=True, help='Normalize input data')
    parser.add_argument('--padding_strategy', type=str, default='random', choices=['zero', 'random', 'repeat'], help='Feature padding strategy') #
    parser.add_argument('--use_batchnorm', type=str2bool, default=True, help='Use BatchNorm instead of LayerNorm')
    parser.add_argument('--projection_small_dim', type=int, default=128, help='Small dimension for identity projection')
    parser.add_argument('--projection_large_dim', type=int, default=512, help='Large dimension for identity projection')

    # === Edge Dropout Augmentation ===
    parser.add_argument('--edge_dropout_rate', type=float, default=0.0,
                       help='Edge dropout rate for data augmentation (0.0-0.5). Set to 0.0 to disable.')
    parser.add_argument('--edge_dropout_enabled', type=str2bool, default=False,
                       help='Enable edge dropout augmentation during training')
    parser.add_argument('--verbose_edge_dropout', type=str2bool, default=False,
                       help='Print edge dropout timing and statistics')

    # === Feature Dropout Augmentation ===
    parser.add_argument('--feature_dropout_rate', type=float, default=0.0,
                       help='Feature dropout rate for data augmentation (0.0-0.8). Set to 0.0 to disable.')
    parser.add_argument('--feature_dropout_enabled', type=str2bool, default=False,
                       help='Enable feature dropout augmentation during training (applied after projection)')
    parser.add_argument('--feature_dropout_type', type=str, default='element_wise',
                       choices=['element_wise', 'channel_wise', 'gaussian_noise'],
                       help='Type of feature dropout: element_wise, channel_wise, or gaussian_noise')
    parser.add_argument('--verbose_feature_dropout', type=str2bool, default=False,
                       help='Print feature dropout timing and statistics')

    # === Link Prediction Specific ===
    parser.add_argument('--context_neg_ratio', type=int, default=3, help='Negative sampling ratio for context')
    parser.add_argument('--train_neg_ratio', type=int, default=3, help='Negative sampling ratio for training')
    parser.add_argument('--context_k', type=int, default=32, help='Number of context samples for link prediction')
    parser.add_argument('--remove_context_from_train', type=str2bool, default=True, help='Remove context from training set')
    parser.add_argument('--mask_target_edges', type=str2bool, default=False, help='Mask target edges during message passing')
    
    # === Graph Classification Specific ===
    parser.add_argument('--gc_train_dataset', type=str, default='bace,bbbp', help='Graph classification training datasets')
    parser.add_argument('--gc_test_dataset', type=str, default='hiv,pcba', help='Graph classification test datasets')
    parser.add_argument('--gc_batch_size', type=int, default=1024, help='Graph classification batch size')
    parser.add_argument('--gc_test_batch_size', type=int, default=4096, help='Graph classification test batch size')
    parser.add_argument('--graph_pooling', type=str, default='max', choices=['mean', 'max', 'sum'], help='Graph pooling method')
    parser.add_argument('--lambda_gc', type=float, default=0.41834063194474214, help='Weight for graph classification loss')
    parser.add_argument('--context_graph_num', type=int, default=8, help='Number of context graphs for graph classification')
    
    # === OGB FUG embeddings arguments (for graph classification) ===
    parser.add_argument('--use_ogb_fug', type=str2bool, default=True,
                        help='Use OGB datasets with FUG embeddings (replaces node features with FUG molecular embeddings)')
    parser.add_argument('--fug_root', type=str, default='./fug',
                        help='Root directory for FUG embeddings')
    parser.add_argument('--ogb_root', type=str, default='./dataset/ogb',
                        help='Root directory for OGB datasets')
    

    # === TSGFM datasets arguments (for graph classification) ===
    parser.add_argument('--use_tsgfm', type=str2bool, default=False,
                        help='Use TSGFM datasets for graph classification (batched format with text features)')
    parser.add_argument('--use_tag_dataset', type=str2bool, default=False,
                        help='Use TAGDataset format with TAGLAS Lite embeddings')
    parser.add_argument('--tag_dataset_root', type=str, default='./dataset/tag',
                        help='Root directory for TAGDataset datasets')
    parser.add_argument('--embedding_family', type=str, default='e5',
                        help='Embedding family to use for graph classification (e.g., e5, st)')
    
    # === Dataset-Specific Context Shot Selection ===
    parser.add_argument('--nc_context_overrides', type=str, default=None,
                       help='Per-dataset context overrides for NC: "dataset1:shots1,dataset2:shots2"')
    parser.add_argument('--lp_context_overrides', type=str, default=None, 
                       help='Per-dataset context overrides for LP: "dataset1:shots1,dataset2:shots2"')
    parser.add_argument('--gc_context_overrides', type=str, default=None,
                       help='Per-dataset context overrides for GC: "dataset1:shots1,dataset2:shots2"')
    
    # === Dynamic Context Refresh ===
    parser.add_argument('--context_refresh_interval', type=int, default=1,
                       help='Refresh contexts every N epochs (0 = never refresh)')
    parser.add_argument('--context_batch_refresh_interval', type=int, default=1,
                       help='Refresh contexts every N batches within each task (0 = disabled)')
    parser.add_argument('--context_sampling_plan', type=str, default='decay', choices=['ori', 'random', 'decay'],
                       help='Context sampling strategy: ori=original fixed, random=random sampling, decay=gradual decay')
    parser.add_argument('--context_bounds', type=str, default='(5,20)(8,32)(8,32)',
                       help='Context bounds for NC/LP/GC: (lower,upper)(lower,upper)(lower,upper). Used for random sampling range and decay start/end.')
    parser.add_argument('--use_kmedoids_sampling', type=str2bool, default=True,
                       help='Use K-Medoids clustering to guide context sampling for better representativeness')
    
    # === Profiling and Performance ===
    parser.add_argument('--enable_profiling', type=str2bool, default=False,
                       help='Enable PyTorch profiler during evaluation to track CPU/GPU usage')
    
    # === Joint Multi-Task Training (TSGFM-style) ===
# Removed joint multitask arguments - only using task-specific and full batch training
    
    # === Checkpointing ===
    parser.add_argument('--save_checkpoint', type=str2bool, default=False, help='Save model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='Custom checkpoint name')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False, help='Use pretrained model')
    
    # === Logging and Monitoring ===
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['QUIET', 'INFO', 'DEBUG', 'VERBOSE'])
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval (epochs)')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval (epochs)')
    # === Experiment Tracking ===
    parser.add_argument('--sweep', type=str2bool, default=False, help='Running hyperparameter sweep')
    
    args = parser.parse_args()
    return args

def parse_graph_classification_args():
    """Parse arguments for graph classification."""
    parser = argparse.ArgumentParser(description='Graph Classification using PFN')
    
    # Model loading / creation
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False,
                        help='Whether to load a pretrained model')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to a trained model checkpoint to load')

    # Model architecture (used if not loading a pretrained model)
    parser.add_argument('--model', type=str, default='GCN', help='GNN model type (e.g., PureGCN_v1, GCN)')
    parser.add_argument("--hidden", default=512, type=int, help='Hidden dimension size')
    parser.add_argument("--num_layers", default=4, type=int, help='Number of GNN layers')
    parser.add_argument("--dp", default=0.058, type=float, help='Dropout rate')
    parser.add_argument('--norm', type=str2bool, default=True, help='Use normalization in GNN')
    parser.add_argument('--res', type=str2bool, default=False, help='Use residual connections in GNN')
    parser.add_argument('--relu', type=str2bool, default=False, help='Use ReLU activation in GNN')
    parser.add_argument('--gnn_norm_affine', type=str2bool, default=True, help='Learnable affine parameters in GNN norm')
    parser.add_argument('--multilayer', type=str2bool, default=True, help='For GCN model: use multilayer structure')
    parser.add_argument('--use_gin', type=str2bool, default=False, help='For GCN model: use GIN variant')

    # PFN-specific architecture for graph classification
    parser.add_argument('--predictor', type=str, default='PFN', help='Predictor type (only PFN supported)')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of layers in MLP pool')
    parser.add_argument('--mlp_norm_affine', type=str2bool, default=True, help='Learnable affine parameters in MLP norm')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads for attention pooling and PFN predictor')
    parser.add_argument('--transformer_layers', type=int, default=1, help='Number of transformer layers in PFN predictor')
    parser.add_argument('--seperate', type=str2bool, default=True, help='For PFN predictor')
    parser.add_argument('--degree', type=str2bool, default=False, help='For PFN predictor (not typically used for graphs)')
    parser.add_argument('--padding', type=str, default='zero', 
                        choices=['zero', 'mlp'], 
                        help='Padding method for PFN predictor: zero/zeros padding or MLP padding')
    
    # Graph classification specific pooling
    parser.add_argument('--graph_pooling', type=str, default='max', 
                        choices=['mean', 'max', 'sum'],
                        help='Graph-level pooling method for aggregating node embeddings')
    
    # Data arguments
    parser.add_argument('--train_dataset', type=str, default='bace,bbbp,muv,tox21,toxcast',
                        help='Comma-separated list of datasets for training graph classification model.')
    parser.add_argument('--test_dataset', type=str, default='chemhiv,chempcba',
                        help='Comma-separated list of datasets for inductive testing.')
    parser.add_argument('--sign_normalize', type=str2bool, default=False, help='For PCA: normalize eigenvectors direction')
    parser.add_argument('--use_full_pca', type=str2bool, default=False, help='For PCA: use full SVD')
    parser.add_argument('--normalize_data', type=str2bool, default=True, help='Normalize data features after preprocessing')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0000038,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size (number of graphs per batch)')
    parser.add_argument('--test_batch_size', type=int, default=4096,
                        help='Test batch size (number of graphs per batch)')
    parser.add_argument('--weight_decay', type=float, default=0.0000025,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.32,
                        help='Gradient clipping')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('--schedule', type=str, default='warmup', choices=['none', 'cosine', 'step', 'warmup'], help='Learning rate schedule')
    parser.add_argument('--orthogonal_push', type=float, default=0.0078, help='Orthogonal push loss weight for prototypes')
    parser.add_argument('--normalize_class_h', type=str2bool, default=True, help='Normalize prototype embeddings')
    
    # Graph classification specific
    parser.add_argument('--sim', type=str, default='dot',
                        choices=['dot', 'cos', 'mlp'],
                        help='Similarity function for PFN predictor')
    parser.add_argument('--context_k', type=int, default=5,
                        help='Number of context graphs per class for prototype generation')
    
    # System arguments
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU specification: "auto" for all GPUs, single GPU ID (e.g., "0"), or comma-separated list (e.g., "0,1,2,3")')
    parser.add_argument('--single_gpu', type=str2bool, default=True,
                        help='Use single GPU mode (DDP not implemented for graph classification yet)')
    parser.add_argument('--port', type=int, default=12357,
                        help='Port for DDP (if implemented in future)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs')
    parser.add_argument('--sweep', type=str2bool, default=False,
                        help='Use sweep for graph classification')
    
    # Logging arguments
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['QUIET', 'INFO', 'DEBUG', 'VERBOSE'],
                        help='Log level: QUIET (minimal), INFO (standard), DEBUG (detailed), VERBOSE (everything)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Interval for printing training progress (epochs)')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Interval for validation evaluation (epochs)')

    # === Full Batch Training ===
    parser.add_argument('--full_batch_training', type=str2bool, default=False, help='Use full batch training: accumulate gradients from all 128 tasks per batch, update once per batch')
    
    # Data processing arguments
    parser.add_argument('--min_pca_dim', type=int, default=32,
                        help='Minimum PCA dimension')
    parser.add_argument('--padding_strategy', type=str, default='random',
                        help='Padding strategy for graph classification')
    parser.add_argument('--use_batchnorm', type=str2bool, default=True,
                        help='Use batch normalization')
    parser.add_argument('--use_identity_projection', type=str2bool, default=False,
                        help='Use identity projection for graph classification')
    parser.add_argument('--projection_small_dim', type=int, default=128,
                        help='Small dimension for identity projection')
    parser.add_argument('--projection_large_dim', type=int, default=256,
                        help='Large dimension for identity projection')
    
    # GPSE embeddings arguments
    parser.add_argument('--use_gpse', type=str2bool, default=False,
                        help='Use pre-computed GPSE embeddings instead of original node embeddings')
    parser.add_argument('--gpse_path', type=str, default='/home/maweishuo/GPSE/datasets',
                        help='Path to GPSE datasets directory')
    
    # OGB FUG embeddings arguments
    parser.add_argument('--use_ogb_fug', type=str2bool, default=False,
                        help='Use OGB datasets with FUG embeddings (replaces node features with FUG molecular embeddings)')
    parser.add_argument('--fug_root', type=str, default='./fug',
                        help='Root directory for FUG embeddings')
    parser.add_argument('--ogb_root', type=str, default='./dataset/ogb',
                        help='Root directory for OGB datasets')
    
    # Checkpointing arguments
    parser.add_argument('--save_checkpoint', type=str2bool, default=False,
                        help='Save checkpoint after training')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Custom checkpoint name (default: auto-generated)')

    parser.add_argument('--use_tag_dataset', type=str2bool, default=False,
                        help='Use tag dataset for graph classification')
    parser.add_argument('--embedding_family', type=str, default='e5',
                        help='Embedding family to use for graph classification (e.g., e5, st)')

    parser.add_argument('--pca_device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to perform PCA computation (cpu=Incremental PCA, gpu=torch.pca_lowrank)')
    parser.add_argument('--incremental_pca_batch_size', type=int, default=1000000, help='Batch size for CPU Incremental PCA')
    parser.add_argument('--pca_sample_threshold', type=int, default=300000, help='If node count exceeds this, sample this many nodes for PCA (default: 500000)')
    parser.add_argument('--pcba_context_only_pca', type=str2bool, default=False, help='PCBA-specific: Use only context + test graphs for PCA fitting (avoids data leakage)')

    parser.add_argument('--use_pca_cache', type=str2bool, default=True, help='Enable PCA result caching for faster subsequent runs')
    parser.add_argument('--pca_cache_dir', type=str, default='./pca_cache', help='Directory to store PCA cache files')
    
    return parser.parse_args()