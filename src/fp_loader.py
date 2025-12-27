"""
Minimal Fingerprint PE Loader for Inductnode
Just loads the .npy files from PAS-OGB and formats for inductnode
"""

import numpy as np
import torch
from pathlib import Path


def load_pas_fingerprint_pe(dataset_name, pas_path="/home/maweishuo/PAS-OGB", 
                           fp_type='morgan', use_pca=False, pe_dim=64):
    """
    Load fingerprints from PAS-OGB as PE for inductnode
    
    Args:
        dataset_name: 'ogbg-molhiv', 'ogbg-molpcba', etc.
        pas_path: Path to PAS-OGB project  
        fp_type: 'morgan', 'maccs', or 'both'
        use_pca: Whether to apply PCA (default: use full features)
        pe_dim: Target dimension if using PCA
        
    Returns:
        dict: PE data for GraphDatasetWithPE
    """
    
    # Map dataset names
    paths = {
        'ogbg-molhiv': 'ogb-molhiv/dataset/ogbg_molhiv',
        'hiv': 'ogb-molhiv/dataset/ogbg_molhiv',
        'ogbg-molpcba': 'ogb-molpcba/dataset/ogbg_molpcba', 
        'pcba': 'ogb-molpcba/dataset/ogbg_molpcba'
    }
    
    dataset_path = Path(pas_path) / paths[dataset_name]
    
    # Load fingerprints
    if fp_type == 'morgan':
        fp = np.load(dataset_path / "mgf_feat.npy").astype(np.float32)
    elif fp_type == 'maccs':
        fp = np.load(dataset_path / "maccs_feat.npy").astype(np.float32) 
    elif fp_type == 'both':
        morgan = np.load(dataset_path / "mgf_feat.npy").astype(np.float32)
        maccs = np.load(dataset_path / "maccs_feat.npy").astype(np.float32)
        fp = np.concatenate([morgan, maccs], axis=1)
    
    print(f"Loaded {fp_type} fingerprints: {fp.shape}")
    
    # Optional PCA
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pe_dim)
        fp = pca.fit_transform(fp)
        print(f"PCA reduced to: {fp.shape}")
    
    # Convert to inductnode format
    embeddings = torch.tensor(fp, dtype=torch.float32)
    slices = torch.arange(len(fp) + 1)
    
    return {'fingerprint_pe': (embeddings, slices)}


if __name__ == "__main__":
    # Test
    pe_data = load_pas_fingerprint_pe('ogbg-molhiv', fp_type='morgan')
    print(f"PE shape: {pe_data['fingerprint_pe'][0].shape}")