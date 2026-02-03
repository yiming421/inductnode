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
    parser.add_argument('--claim_all_gpu_memory', type=str2bool, default=False, help='Claim all available GPU memory at startup to prevent competition')

    # === Model Architecture ===
    parser.add_argument('--model', type=str, default='PureGCN_v1', choices=['PureGCN_v1', 'GCN', 'UnifiedGNN'])
    parser.add_argument('--predictor', type=str, default='PFN', choices=['PFN', 'MPLP'])
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--transformer_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--transformer_norm_type', type=str, default='pre', choices=['pre', 'post'],
                        help='Transformer normalization type: pre-norm (default, more stable) or post-norm')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of MLP layers')
    parser.add_argument('--ffn_expansion_ratio', type=int, default=4,
                        choices=[1, 2, 4, 8],
                        help='FFN expansion ratio: hidden_dim * expansion_ratio')
    parser.add_argument('--dp', type=float, default=0, help='Dropout rate')
    parser.add_argument('--norm', type=str2bool, default=True, help='Use normalization')
    parser.add_argument('--res', type=str2bool, default=False, help='Use residual connections')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu', 'silu'],
                        help='Activation function: relu, gelu (Gaussian Error Linear Unit), or silu (Sigmoid Linear Unit)')
    parser.add_argument('--relu', type=str2bool, default=False, help='Use ReLU activation (deprecated, use --activation)')
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
    parser.add_argument('--linear', type=str2bool, default=True, help='Apply linear transformation after conv')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for LightGCN')
    parser.add_argument('--exp', type=str2bool, default=False, help='Use exponential alpha weights')
    parser.add_argument('--gin_aggr', type=str, default='sum', choices=['sum', 'mean'], help='Aggregation for GIN')
    parser.add_argument('--supports_edge_weight', type=str2bool, default=True, help='Whether model supports edge weights')
    parser.add_argument('--no_parameters', type=str2bool, default=False, help='Use parameter-free convolutions')
    parser.add_argument('--input_norm', type=str2bool, default=False, help='Apply input normalization')
    
    # === Training Configuration ===
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.000001995432810684568, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizer (term added to denominator for numerical stability)')
    parser.add_argument('--schedule', type=str, default='cosine', choices=['cosine', 'step', 'warmup', 'none'])
    parser.add_argument('--nc_batch_size', type=int, default=1024, help='Node classification batch size')
    parser.add_argument('--lp_batch_size', type=int, default=16384, help='Link prediction batch size')
    parser.add_argument('--test_batch_size', type=int, default=16384, help='Test batch size')
    parser.add_argument('--unseen_test_context_samples', type=int, default=3,
                        help='Average unseen NC test metrics over N random few-shot context resamples (>=1)')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    
    # === Joint Training Specific ===
    parser.add_argument('--enable_nc', type=str2bool, default=True, help='Enable node classification task')
    parser.add_argument('--enable_lp', type=str2bool, default=True, help='Enable link prediction task')
    parser.add_argument('--enable_gc', type=str2bool, default=True, help='Enable graph classification task')

    # Separate optimizers option
    parser.add_argument('--use_separate_optimizers', type=str2bool, default=True,
                        help='Use separate optimizers for each task with task-specific learning rates')

    # Task-specific learning rates (only used when use_separate_optimizers=True)
    parser.add_argument('--lr_nc', type=float, default=None, help='Learning rate for node classification (uses --lr if None)')
    parser.add_argument('--lr_lp', type=float, default=None, help='Learning rate for link prediction (uses --lr if None)')
    parser.add_argument('--lr_gc', type=float, default=None, help='Learning rate for graph classification (uses --lr if None)')
    parser.add_argument('--lr_graphcl', type=float, default=None, help='Learning rate for GraphCL (uses --lr if None)')

    # Legacy lambda weights (deprecated, kept for backward compatibility)
    parser.add_argument('--lambda_nc', type=float, default=0.5339754552414909, help='[DEPRECATED] Weight for node classification loss - use --lr_nc instead')
    parser.add_argument('--lambda_lp', type=float, default=2.735303979230086, help='[DEPRECATED] Weight for link prediction loss - use --lr_lp instead')

    # === Hierarchical Training ===
    parser.add_argument('--use_hierarchical_training', type=str2bool, default=False,
                        help='Enable hierarchical/phased multi-task training to reduce task conflict')
    parser.add_argument('--hierarchical_phases', type=str, default='lp,nc+lp,nc+lp+gc',
                        help='Task schedule per phase (comma-separated). Use + for multiple tasks. Tasks: nc, lp, gc. Phases split at epochs 15 and 30.')

    # === Virtual Node ===
    parser.add_argument('--use_virtual_node', type=str2bool, default=True,
                        help='Add a virtual node connected to all graph nodes for global information aggregation (uses main dropout rate and residual connections)')

    # === Dataset Configuration ===
    parser.add_argument('--nc_train_dataset', type=str, default='ogbn-arxiv,CS,Physics,Computers,Photo,Flickr,USA,Brazil,Europe,Wiki,BlogCatalog,DBLP,FacebookPagePage,Actor,DeezerEurope,LastFMAsia,Twitch-DE,Twitch-EN,Twitch-ES,Twitch-FR,Twitch-PT,Twitch-RU', 
                       help='Node classification training datasets')
    parser.add_argument('--nc_test_dataset', type=str, default='Cora,Citeseer,Pubmed,WikiCS', 
                       help='Node classification test datasets')
    parser.add_argument('--lp_train_dataset', type=str, default='CS,Physics,Computers,Photo,Flickr,Wiki,BlogCatalog,FacebookPage', 
                       help='Link prediction training datasets')
    parser.add_argument('--lp_test_dataset', type=str, default='Cora,Citeseer,Pubmed,ogbl-collab', 
                       help='Link prediction test datasets')
    
    # === Model Components ===
    parser.add_argument('--use_identity_projection', type=str2bool, default=False, help='Use identity projection')
    
    # === PFN Predictor Configuration ===
    parser.add_argument('--context_num', type=int, default=5, help='Number of context nodes')
    parser.add_argument('--seperate', type=str2bool, default=True, help='Separate processing in PFN predictor')
    parser.add_argument('--padding', type=str, default='zero', choices=['zero', 'mlp'], help='Padding method for PFN predictor')
    
    # Similarity and Ridge Regression Configuration
    parser.add_argument('--sim', type=str, default='dot', choices=['dot', 'cos', 'euclidean', 'mlp', 'ridge'], 
                        help='Default similarity function (legacy, use nc_sim/lp_sim for task-specific control)')
    parser.add_argument('--ridge_alpha', type=float, default=1.0, 
                        help='Default regularization strength for ridge regression (legacy, use nc_ridge_alpha/lp_ridge_alpha for task-specific control)')
    
    # Task-Specific Similarity and Ridge Configuration
    parser.add_argument('--nc_sim', type=str, default='dot', choices=['dot', 'cos', 'euclidean', 'mlp', 'ridge'],
                        help='Similarity function for node classification')
    parser.add_argument('--nc_ridge_alpha', type=float, default=1.0,
                        help='Ridge regression regularization strength for node classification')
    parser.add_argument('--lp_sim', type=str, default='dot', choices=['dot', 'cos', 'euclidean', 'mlp', 'ridge'],
                        help='Similarity function for link prediction')
    parser.add_argument('--lp_ridge_alpha', type=float, default=1.0,
                        help='Ridge regression regularization strength for link prediction')
    parser.add_argument('--gc_sim', type=str, default='dot', choices=['dot', 'cos', 'euclidean', 'mlp', 'ridge'],
                        help='Similarity function for graph classification')
    parser.add_argument('--gc_ridge_alpha', type=float, default=1.0,
                        help='Ridge regression regularization strength for graph classification')

    parser.add_argument('--head_num_layers', type=int, default=0, help='Number of MLP layers in task-specific heads')
    parser.add_argument('--nc_head_num_layers', type=int, default=None,
                        help='Override head_num_layers for node classification head (default: use --head_num_layers)')
    parser.add_argument('--lp_head_num_layers', type=int, default=None,
                        help='Override head_num_layers for link prediction head (default: use --head_num_layers)')
    parser.add_argument('--orthogonal_push', type=float, default=0, help='Orthogonal push regularization weight')
    parser.add_argument('--normalize_class_h', type=str2bool, default=True, help='Normalize class embeddings')

    # === Correct & Smooth (C&S) Post-processing ===
    parser.add_argument('--use_cs', type=str2bool, default=True,
                        help='Apply Correct & Smooth post-processing. Decision is validation-based: only uses C&S if it improves validation accuracy (no test leakage)')
    parser.add_argument('--cs_num_iters', type=int, default=50, help='Number of label propagation iterations for C&S')
    parser.add_argument('--cs_alpha', type=float, default=0.5, help='Blending factor for C&S (higher = more emphasis on previous iteration)')

    # Meta-Graph C&S for Graph Classification
    parser.add_argument('--use_graph_cs', type=str2bool, default=False,
                        help='Use graph-level Correct & Smooth (build meta-graph with anchors and apply label propagation)')
    parser.add_argument('--num_anchors', type=int, default=1000,
                        help='Number of anchor graphs for meta-graph construction')
    parser.add_argument('--cs_k_neighbors', type=int, default=10,
                        help='Number of neighbors to connect in meta-graph')
    parser.add_argument('--weight_sharpening', type=float, default=1.0,
                        help='Power to raise edge weights to (>1: emphasize strong connections)')
    parser.add_argument('--meta_graph_sim', type=str, default='cos', choices=['cos', 'tanimoto', 'dot'],
                        help='Similarity metric for meta-graph construction')

    # === Matching Network Configuration ===
    parser.add_argument('--use_matching_network', type=str2bool, default=False, help='Use matching network instead of prototype-based prediction')
    parser.add_argument('--matching_network_projection', type=str, default='linear', choices=['linear', 'mlp'], help='Projection type for Q/K in matching network')
    parser.add_argument('--matching_network_temperature', type=float, default=1.0, help='Temperature for matching network attention (attn = scores / temp, so lower = sharper)')
    parser.add_argument('--matching_network_learnable_temp', type=str2bool, default=True, help='Make temperature learnable')
    
    # === Data Processing ===
    parser.add_argument('--use_full_pca', type=str2bool, default=False, help='Use full PCA decomposition')
    parser.add_argument('--use_random_orthogonal', type=str2bool, default=False, help='Use random orthogonal projection instead of PCA (ablation study)')
    parser.add_argument('--use_orthogonal_noise', type=str2bool, default=False, help='Replace features with orthogonal noise (ablation study)')
    parser.add_argument('--use_sparse_random', type=str2bool, default=False, help='Use sparse random projection instead of PCA (ablation study)')
    parser.add_argument('--sparse_random_density', type=float, default=0.1, help='Density of non-zero entries in sparse random projection (0.1 = 10% non-zero)')
    parser.add_argument('--use_pca_whitening', type=str2bool, default=False, help='Apply whitening after PCA (normalize by eigenvalues)')
    parser.add_argument('--whitening_epsilon', type=float, default=0.01, help='Regularization epsilon for whitening to avoid numerical issues')
    parser.add_argument('--use_quantile_normalization', type=str2bool, default=False, help='Apply quantile normalization after PCA to align feature distributions across datasets')
    parser.add_argument('--quantile_norm_before_padding', type=str2bool, default=True, help='Apply quantile normalization before padding (True) or after padding (False). Experiment to see which works better.')
    parser.add_argument('--test_process_test_only', type=str2bool, default=True, help='For test datasets, only process test split (avoid data leakage and improve efficiency)')
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
    parser.add_argument('--use_mlp_projection', type=str2bool, default=False, help='DEBUG: Use MLP projection instead of PCA+padding (e.g., for PCBA)')
    parser.add_argument('--mlp_projection_input_dim', type=int, default=9, help='Input dimension for MLP projection (default 9 for PCBA)')

    # === Dynamic Encoder (DE) Configuration ===
    parser.add_argument('--use_dynamic_encoder', type=str2bool, default=False,
                       help='Use Dynamic Encoder for end-to-end feature projection (replaces PCA)')
    parser.add_argument('--de_sample_size', type=int, default=1024,
                       help='Number of nodes to sample for DE column sampling')
    parser.add_argument('--de_hidden_dim', type=int, default=512,
                       help='Hidden dimension for DE MLP')
    parser.add_argument('--lambda_de', type=float, default=0.01,
                       help='Weight for DE uniformity loss (prevents basis collapse)')
    parser.add_argument('--de_update_sample_every_n_steps', type=int, default=1,
                       help='Update DE sample every N forward passes (1=every step, higher=more stable)')
    parser.add_argument('--de_lr_scale', type=float, default=0.1,
                       help='Learning rate scale for DE parameters (default: 0.1, i.e., 10%% of base lr)')

    # === Edge Dropout Augmentation ===
    parser.add_argument('--edge_dropout_rate', type=float, default=0.0,
                       help='Edge dropout rate for data augmentation (0.0-0.5). Set to 0.0 to disable.')
    parser.add_argument('--edge_dropout_enabled', type=str2bool, default=True,
                       help='Enable edge dropout augmentation during training')
    parser.add_argument('--verbose_edge_dropout', type=str2bool, default=False,
                       help='Print edge dropout timing and statistics')

    # === Feature Dropout Augmentation ===
    parser.add_argument('--feature_dropout_rate', type=float, default=0.0,
                       help='Feature dropout rate for data augmentation (0.0-0.8). Set to 0.0 to disable.')
    parser.add_argument('--feature_dropout_enabled', type=str2bool, default=True,
                       help='Enable feature dropout augmentation during training (applied after projection)')
    parser.add_argument('--feature_dropout_type', type=str, default='channel_wise',
                       choices=['element_wise', 'channel_wise', 'gaussian_noise'],
                       help='Type of feature dropout: element_wise, channel_wise, or gaussian_noise')

    # === Random Projection Augmentation ===
    parser.add_argument('--use_random_projection_augmentation', type=str2bool, default=False,
                       help='Apply random projection augmentation: σ(WX+b) with random σ, W, b, and hidden_dim')
    parser.add_argument('--num_augmentations', type=int, default=1,
                       help='Number of augmented copies to create per graph (1=double dataset, 2=triple, etc.)')
    parser.add_argument('--augmentation_mode', type=str, default='preprocessing',
                       choices=['preprocessing', 'per_epoch'],
                       help='Augmentation mode: preprocessing (fixed augmentations created once) or per_epoch (new random augmentations each epoch)')
    parser.add_argument('--augmentation_regenerate_interval', type=int, default=1,
                       help='Regenerate augmentations every N epochs in per_epoch mode (1 = every epoch, 5 = every 5 epochs, etc.)')
    parser.add_argument('--augmentation_include_original', type=str2bool, default=False,
                       help='Whether to include original graphs in training (True) or train only on augmented graphs (False)')
    parser.add_argument('--augmentation_shuffle', type=str2bool, default=False,
                       help='Shuffle training dataset list each epoch (mix original and augmented graphs)')
    parser.add_argument('--augmentation_activation', type=str, default='random',
                       help='Activation function for augmentation: "random" (randomly sample each time) or fixed name like "relu", "gelu", "tanh", "sin", etc.')
    parser.add_argument('--augmentation_max_depth', type=int, default=1,
                       help='Maximum depth of MLP for augmentation (1=single layer σ(WX+b), >1=multi-layer MLP with random depth in [1, max_depth])')
    parser.add_argument('--augmentation_verbose', type=str2bool, default=False,
                       help='Print detailed augmentation information')
    parser.add_argument('--augmentation_use_random_noise', type=str2bool, default=False,
                       help='ABLATION: Replace σ(WX+b) with pure random Gaussian noise (tests if model learns from graph structure only)')
    parser.add_argument('--augmentation_dropout_rate', type=float, default=0.0,
                       help='Dropout rate to apply before projection in each layer (X -> dropout -> WX+b -> activation). 0.0 = no dropout, creates stronger augmentation diversity')
    parser.add_argument('--augmentation_use_feature_mixing', type=str2bool, default=False,
                       help='Apply feature mixing augmentation: interpolate features between randomly paired nodes')
    parser.add_argument('--augmentation_mix_ratio', type=float, default=0.4,
                       help='Ratio of nodes to participate in feature mixing (0.0 to 1.0)')
    parser.add_argument('--augmentation_mix_alpha', type=float, default=0.5,
                       help='Interpolation weight for feature mixing (0.5 = equal mix, 0.8 = keep 80%% original)')

    # === Contrastive Augmentation Loss ===
    parser.add_argument('--use_contrastive_augmentation_loss', type=str2bool, default=False,
                       help='Enable contrastive loss to minimize difference between original and augmented embeddings (uses cosine similarity)')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.1,
                       help='Weight for contrastive augmentation loss (multiplier for the loss term)')

    # === Bank of Tags (VQ-VAE Style Fixed Tags) ===
    parser.add_argument('--use_bank_of_tags', type=str2bool, default=False,
                       help='Use fixed random orthogonal Bank of Tags instead of learned class prototypes (like VQ-VAE codebook for class labels)')
    parser.add_argument('--bank_of_tags_max_classes', type=int, default=1024,
                       help='Maximum number of classes supported by Bank of Tags (tags are orthogonal when hidden_dim >= max_classes)')
    parser.add_argument('--bank_of_tags_refresh_interval', type=int, default=1,
                       help='Refresh tag permutation every N epochs to prevent overfitting (1=every epoch, 0=never refresh)')

    # === Test-Time Augmentation (TTA) ===
    parser.add_argument('--use_test_time_augmentation', type=str2bool, default=True,
                       help='Apply test-time augmentation during final evaluation: create multiple augmented versions and aggregate predictions')
    parser.add_argument('--tta_num_augmentations', type=int, default=20,
                       help='Number of augmented versions to create at test time (will be aggregated with original graph)')
    parser.add_argument('--tta_aggregation', type=str, default='probs',
                       choices=['logits', 'probs', 'voting'],
                       help='TTA aggregation strategy: logits (average logits before softmax), probs (average probabilities), voting (majority vote)')
    parser.add_argument('--tta_include_original', type=str2bool, default=True,
                       help='Include original graph in TTA aggregation (True) or only use augmented versions (False)')
    parser.add_argument('--tta_gate_by_valid', type=str2bool, default=True,
                       help='When TTA is enabled, only trust TTA if it improves the validation metric (per dataset)')
    parser.add_argument('--use_train_time_augmentation', type=str2bool, default=False,
                       help='Average logits across multiple TTA-style augmented views during full-batch training (costly)')
    parser.add_argument('--train_tta_num_augmentations', type=int, default=5,
                       help='Number of augmented views for train-time TTA (in addition to optional original)')
    parser.add_argument('--train_tta_include_original', type=str2bool, default=True,
                       help='Include original view in train-time TTA averaging')

    # === GPSE Embeddings ===
    parser.add_argument('--use_gpse', type=str2bool, default=False,
                       help='Use GPSE (Graph Positional and Structural Encoder) embeddings to enhance node features')
    parser.add_argument('--gpse_dir', type=str, default='../GPSE/datasets',
                       help='Directory containing pre-computed GPSE embeddings')
    parser.add_argument('--gpse_verbose', type=str2bool, default=False,
                       help='Print GPSE loading information')
    parser.add_argument('--use_lappe', type=str2bool, default=False,
                       help='Use Laplacian Positional Encoding (top-k eigenvectors of graph Laplacian)')
    parser.add_argument('--use_rwse', type=str2bool, default=False,
                       help='Use Random Walk Structural Encoding (return probabilities after k steps)')
    parser.add_argument('--verbose_feature_dropout', type=str2bool, default=False,
                       help='Print feature dropout timing and statistics')

    # === External Embeddings (FUG) ===
    parser.add_argument('--use_external_embeddings_nc', type=str2bool, default=False,
                       help='Use external embeddings for node classification (load from fug_root)')

    # === Mini-Batch Sampling for Large Datasets ===
    parser.add_argument('--use_minibatch_sampling', type=str2bool, default=False,
                       help='Enable mini-batch sampling with NeighborLoader for large datasets')
    parser.add_argument('--minibatch_node_threshold', type=int, default=100000,
                       help='Datasets with more nodes than this threshold will use mini-batch sampling')
    parser.add_argument('--minibatch_batch_size', type=int, default=1024,
                       help='Batch size for mini-batch sampling (nodes per batch)')
    parser.add_argument('--minibatch_num_neighbors', type=str, default='10,8,6,4',
                       help='Number of neighbors to sample per layer (comma-separated, outer to inner). Will be capped at 5 layers.')
    parser.add_argument('--minibatch_batches_per_epoch', type=int, default=100,
                       help='Number of batches to sample per epoch for large datasets')
    parser.add_argument('--minibatch_num_workers', type=int, default=2,
                       help='Number of worker threads for parallel batch loading')

    # === Split Rebalancing for Pretraining ===
    parser.add_argument('--split_rebalance_strategy', type=str, default='legacy',
                       choices=['smallest_for_valid', 'legacy', 'original'],
                       help='How to rebalance dataset splits for pretraining: '
                            'smallest_for_valid (use smallest split for validation), '
                            'legacy (train+valid for training, test for validation), '
                            'original (keep original splits)')
    parser.add_argument('--eval_max_batches', type=int, default=100,
                       help='Maximum number of batches for mini-batch evaluation (0 = no limit, evaluate all)')

    # === Link Prediction Specific ===
    parser.add_argument('--context_neg_ratio', type=int, default=3, help='Negative sampling ratio for context')
    parser.add_argument('--train_neg_ratio', type=int, default=3, help='Negative sampling ratio for training')
    parser.add_argument('--context_k', type=int, default=5, help='Number of context samples for link prediction')
    parser.add_argument('--remove_context_from_train', type=str2bool, default=True, help='Remove context from training set')
    parser.add_argument('--mask_target_edges', type=str2bool, default=False, help='Mask target edges during message passing')
    parser.add_argument('--lp_metric', type=str, default='auto', choices=['auto', 'auc', 'acc', 'hits@20', 'hits@50', 'hits@100', 'mrr'],
                       help='Metric to use for link prediction evaluation (auto=dataset default, auc, acc, or hits@K/mrr)')
    parser.add_argument('--lp_head_type', type=str, default='standard', choices=['standard', 'mplp'], help='Type of link prediction head')
    
    # === MPLP Head Configuration ===
    parser.add_argument('--mplp_signature_dim', type=int, default=1024, help='Dimension of random signature vectors for MPLP')
    parser.add_argument('--mplp_num_hops', type=int, default=2, help='Number of hops for MPLP structure propagation')
    parser.add_argument('--mplp_feature_combine', type=str, default='hadamard', choices=['hadamard', 'concat'], help='Method to combine node features in MPLP')
    parser.add_argument('--mplp_prop_type', type=str, default='combine', choices=['exact', 'combine'],
                        help='Structural feature type for MPLP head (default: combine)')
    parser.add_argument('--mplp_signature_sampling', type=str, default='torchhd', choices=['torchhd', 'gaussian'],
                        help='Signature sampling method for MPLP (default: torchhd)')
    parser.add_argument('--mplp_use_subgraph', type=str2bool, default=True,
                        help='Use 2-hop subgraph for MPLP structural features (default: True)')
    parser.add_argument('--mplp_use_degree', type=str, default='mlp', choices=['none', 'aa', 'ra', 'mlp'],
                        help='Degree-based node weighting for MPLP structural features')
    parser.add_argument('--lp_concat_common_neighbors', type=str2bool, default=False,
                        help='Concatenate common-neighbor count to edge embeddings before LP head')
    
    # === Graph Classification Specific ===
    parser.add_argument('--gc_train_dataset', type=str, default='bace,bbbp', help='Graph classification training datasets')
    parser.add_argument('--gc_test_dataset', type=str, default='hiv,pcba', help='Graph classification test datasets')
    parser.add_argument('--gc_batch_size', type=int, default=1024, help='Graph classification batch size')
    parser.add_argument('--gc_test_batch_size', type=int, default=4096, help='Graph classification test batch size')
    parser.add_argument('--graph_pooling', type=str, default='max', choices=['mean', 'max', 'sum'], help='Graph pooling method')
    parser.add_argument('--lambda_gc', type=float, default=0.41834063194474214, help='Weight for graph classification loss')
    parser.add_argument('--context_graph_num', type=int, default=5, help='Number of context graphs for graph classification')
    parser.add_argument('--gc_multitask_vectorized', type=str2bool, default=False,
                       help='Enable vectorized multi-task prototypical GC (single BCEWithLogits over all tasks, e.g., PCBA)')
    parser.add_argument('--gc_supervised_mlp', type=str2bool, default=False,
                       help='Use supervised MLP head for graph classification (bypasses PFN/transformer)')
    parser.add_argument('--gc_profile_context', type=str2bool, default=False,
                       help='Profile context prototype computation time (encode vs overhead) in vectorized GC')
    parser.add_argument('--gc_log_train_metrics', type=str2bool, default=True,
                       help='Log graph classification train metrics each epoch (can be expensive for large datasets)')
    parser.add_argument('--gc_train_eval_max_batches', type=int, default=20,
                       help='Max batches to use when computing GC train metrics (0 = all batches)')
    parser.add_argument('--gc_train_eval_shuffle', type=str2bool, default=True,
                       help='Shuffle train eval loader when using batch sampling for GC train metrics')

    # === Multi-Dataset Sampling Arguments ===
    parser.add_argument('--multi_dataset_sampling', type=str2bool, default=False,
                       help='Enable temperature-based multi-dataset random sampling for graph classification')
    parser.add_argument('--sampling_temperature', type=float, default=0.5,
                       help='Temperature for dataset sampling probability (0=uniform, 1=size-proportional). '
                            'Formula: p_i = (N_i)^T / sum_j((N_j)^T) where N_i = num_graphs_i * num_tasks_i')
    parser.add_argument('--verbose_sampling', type=str2bool, default=False,
                       help='Print detailed sampling statistics during training')
    
    # === OGB FUG embeddings arguments (for graph classification) ===
    parser.add_argument('--use_ogb_fug', type=str2bool, default=True,
                        help='Use OGB datasets with FUG embeddings (replaces node features with FUG molecular embeddings)')
    parser.add_argument('--use_original_features', type=str2bool, default=True,
                        help='Use original OGB raw features (9-dim) instead of FUG embeddings. Will be processed with PCA/padding to hidden_dim.')
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
    parser.add_argument('--context_batch_refresh_interval', type=int, default=0,
                       help='Refresh contexts every N batches within each task (0 = disabled)')
    parser.add_argument('--context_sampling_plan', type=str, default='ori', choices=['ori', 'random', 'decay'],
                       help='Context sampling strategy: ori=original fixed, random=random sampling, decay=gradual decay')
    parser.add_argument('--context_bounds', type=str, default='(5,20)(5,20)(5,20)',
                       help='Context bounds for NC/LP/GC: (lower,upper)(lower,upper)(lower,upper). Used for random sampling range and decay start/end.')

    # === GraphCL (Graph Contrastive Learning) ===
    parser.add_argument('--enable_graphcl', type=str2bool, default=False,
                       help='Enable GraphCL self-supervised contrastive learning as a separate task')
    parser.add_argument('--graphcl_dataset', type=str, default='bace,bbbp',
                       help='Datasets for GraphCL task (can be graph classification datasets or node classification datasets for subgraph extraction)')
    parser.add_argument('--lambda_graphcl', type=float, default=0.1,
                       help='Weight for GraphCL contrastive loss in multi-task learning')
    parser.add_argument('--graphcl_aug1_type', type=str, default='edge_drop',
                       choices=['edge_drop', 'feat_mask', 'subgraph', 'none'],
                       help='First augmentation type for GraphCL (creates view 1)')
    parser.add_argument('--graphcl_aug2_type', type=str, default='edge_drop',
                       choices=['edge_drop', 'feat_mask', 'subgraph', 'none'],
                       help='Second augmentation type for GraphCL (creates view 2)')
    parser.add_argument('--graphcl_aug1_ratio', type=float, default=0.2,
                       help='Augmentation ratio for first view (e.g., edge drop ratio or feature mask ratio)')
    parser.add_argument('--graphcl_aug2_ratio', type=float, default=0.2,
                       help='Augmentation ratio for second view')
    parser.add_argument('--graphcl_temperature', type=float, default=0.5,
                       help='Temperature parameter for InfoNCE contrastive loss')
    parser.add_argument('--graphcl_projection_dim', type=int, default=128,
                       help='Projection head output dimension for GraphCL (maps graph embeddings to contrastive space)')
    parser.add_argument('--graphcl_batch_size', type=int, default=256,
                       help='Batch size for GraphCL task')
    
    # === Profiling and Performance ===
    parser.add_argument('--enable_profiling', type=str2bool, default=False,
                       help='Enable PyTorch profiler during evaluation to track CPU/GPU usage')
    
    # === Joint Multi-Task Training (TSGFM-style) ===
# Removed joint multitask arguments - only using task-specific and full batch training
    
    # === Checkpointing ===
    parser.add_argument('--save_checkpoint', type=str2bool, default=True, help='Save model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='Custom checkpoint name')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--checkpoint_threshold', type=float, default=2.0,
                        help='Minimum combined score threshold for saving checkpoint (default: 0.0)')
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False, help='Use pretrained model')
    
    # === Logging and Monitoring ===
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['QUIET', 'INFO', 'DEBUG', 'VERBOSE'])
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval (epochs)')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom name for wandb run (default: auto-generated)')
    parser.add_argument('--log_individual_datasets', type=str2bool, default=False,
                       help='Log individual dataset loss and accuracy to wandb (default: False)')

    # === Transformer Configuration ===
    parser.add_argument('--skip_token_formulation', type=str2bool, default=False,
                       help='Skip token formulation (label concatenation) in Transformer, use GNN embeddings directly (default: False)')

    # === Experiment Tracking ===
    parser.add_argument('--sweep', type=str2bool, default=False, help='Running hyperparameter sweep')

    # === Ablation Study ===
    parser.add_argument('--use_first_half_embedding', type=str2bool, default=False,
                        help='Use first half of embedding after transformer instead of second half for prototype calculation')
    parser.add_argument('--use_full_embedding', type=str2bool, default=False,
                        help='Use full embedding (both halves) after transformer for prototype calculation instead of just one half')

    # === Visualization ===
    parser.add_argument('--plot_tsne', type=str2bool, default=False, help='Plot t-SNE visualization of processed features')
    parser.add_argument('--tsne_save_dir', type=str, default='./tsne_plots', help='Directory to save t-SNE plots')

    # === GraphPFN Configuration ===
    parser.add_argument('--use_graphpfn', type=str2bool, default=False,
                        help='Use GraphPFN (TabPFN-style dual attention transformer) instead of naive transformer')
    parser.add_argument('--graphpfn_emsize', type=int, default=192,
                        help='GraphPFN embedding dimension (TabPFN default: 192)')
    parser.add_argument('--graphpfn_nhead', type=int, default=6,
                        help='GraphPFN number of attention heads (TabPFN default: 6)')
    parser.add_argument('--graphpfn_nlayers', type=int, default=12,
                        help='GraphPFN number of transformer layers (TabPFN default: 12)')
    parser.add_argument('--graphpfn_nhid_factor', type=int, default=4,
                        help='GraphPFN MLP hidden dimension = emsize * nhid_factor (TabPFN default: 4)')
    parser.add_argument('--graphpfn_features_per_group', type=int, default=2,
                        help='GraphPFN number of features per group for dual attention (TabPFN default: 2, use 1 for coordinate-free with Fourier features)')
    parser.add_argument('--graphpfn_fourier_scale', type=float, default=1.0,
                        help='Fourier feature scale when features_per_group=1 (default: 1.0, higher=more sensitive to value differences)')
    parser.add_argument('--graphpfn_dropout', type=float, default=0.0,
                        help='GraphPFN dropout probability (TabPFN default: 0.0)')
    parser.add_argument('--graphpfn_attention_between_features', type=str2bool, default=True,
                        help='GraphPFN enable attention between feature groups - key TabPFN innovation (default: True)')
    parser.add_argument('--graphpfn_feature_positional_embedding', type=str, default='subspace',
                        choices=['subspace', 'learned', 'normal_rand_vec', 'uni_rand_vec', 'none'],
                        help='GraphPFN feature positional embedding type (TabPFN default: subspace). CRITICAL for distinguishing feature groups!')
    parser.add_argument('--graphpfn_seed', type=int, default=0,
                        help='GraphPFN random seed for feature positional embeddings (TabPFN default: 0)')
    parser.add_argument('--graphpfn_normalize_x', type=str2bool, default=True,
                        help='GraphPFN normalize input features (TabPFN default: True)')
    parser.add_argument('--graphpfn_cache_trainset', type=str2bool, default=True,
                        help='GraphPFN cache K,V for context nodes during inference (major speedup)')

    args = parser.parse_args()

    # === Validation: Check for incompatible flag combinations ===
    if args.use_dynamic_encoder:
        # When using DE, disable conflicting projection methods
        if args.use_identity_projection:
            print("Warning: --use_dynamic_encoder is enabled, disabling --use_identity_projection")
            args.use_identity_projection = False

        if getattr(args, 'use_external_embeddings_nc', False):
            print("Warning: --use_dynamic_encoder is enabled, disabling --use_external_embeddings_nc")
            args.use_external_embeddings_nc = False

    return args
