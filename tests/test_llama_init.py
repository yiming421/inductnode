
import torch
import sys
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We need to mock the imports inside src.model_llama because it imports inside __init__
# But since we import it here, we can just patch transformers.LlamaConfig globally
import transformers
from src.model_llama import LlamaPFNPredictorNodeCls

def run_test():
    print("=== Testing Llama Initialization Hypothesis ===")
    
    # Configuration
    hidden_dim = 64
    num_heads = 4
    num_context = 10  # 5 shots * 2 classes
    num_target = 1
    device = "cpu"
    
    # Synthetic Data with Structure
    # Create two distinct class centers
    center_0 = torch.randn(1, hidden_dim, device=device) * 2.0
    center_1 = torch.randn(1, hidden_dim, device=device) * 2.0
    
    # Context: First 5 are Class 0 (cluster around center_0), Next 5 are Class 1 (cluster around center_1)
    # Add small noise to make them distinct but correlated
    c0 = center_0 + torch.randn(5, hidden_dim, device=device) * 0.5
    c1 = center_1 + torch.randn(5, hidden_dim, device=device) * 0.5
    context_x = torch.cat([c0, c1], dim=0)
    
    context_y = torch.tensor([0]*5 + [1]*5, device=device)
    
    # Target: From Class 0 (should attend to first 5)
    target_x = center_0 + torch.randn(num_target, hidden_dim, device=device) * 0.5
    
    # Class Prototypes (random is fine here, assuming attention is mostly on node features)
    class_x = torch.randn(2, hidden_dim, device=device)
    
    # Dummy data object
    # nc_head needs data.y to determine num_classes, and data.context_sample for process_node_features
    # y must be large enough for context_sample indices
    data = SimpleNamespace(
        x=torch.randn(100, hidden_dim), 
        y=torch.zeros(100, dtype=torch.long), # Valid labels for context_sample
        context_sample=torch.arange(10) # Dummy context indices
    )
    data.y[:5] = 0
    data.y[5:10] = 1

    # --- Test 1: Original Initialization (0.02) ---
    print("\n1. Testing Original Init (0.02) + Pre-Amp Boost...")
    
    torch.manual_seed(42)
    model_orig = LlamaPFNPredictorNodeCls(
        hidden_dim=hidden_dim, 
        llama_num_heads=num_heads,
        num_layers=2,
        skip_token_formulation=True, # Simplify for test
        disable_rope=True
    )
    model_orig.enable_diagnostics(True)
    model_orig.eval()
    
    with torch.no_grad():
        _, _ = model_orig(data, context_x, target_x, context_y, class_x)
        diag_orig = model_orig.get_diagnostics()
        
    entropy_orig = diag_orig['attention/layer_0_avg_entropy']
    print(f"   -> Average Attention Entropy: {entropy_orig:.4f} (1.0 = Uniform)")

    # --- Test 2: Proposed Fix (Xavier Init) ---
    print("\n2. Testing Xavier Init (1/sqrt(d)) ...")
    
    xavier_val = 1.0 / (hidden_dim ** 0.5) # 1/8 = 0.125
    print(f"   -> Calculated Xavier Range: {xavier_val:.4f}")
    
    # Capture the original init to call it later
    original_llama_config_init = transformers.LlamaConfig.__init__
    
    # Define patch
    def patched_init(self, *args, **kwargs):
        # Force overwrite initializer_range
        kwargs['initializer_range'] = xavier_val
        original_llama_config_init(self, *args, **kwargs)
        
    torch.manual_seed(42)
    with patch.object(transformers.LlamaConfig, '__init__', side_effect=patched_init, autospec=True):
        model_fixed = LlamaPFNPredictorNodeCls(
            hidden_dim=hidden_dim, 
            llama_num_heads=num_heads,
            num_layers=2,
            skip_token_formulation=True,
            disable_rope=True
        )
    
    model_fixed.enable_diagnostics(True)
    model_fixed.eval()
    
    with torch.no_grad():
        _, _ = model_fixed(data, context_x, target_x, context_y, class_x)
        diag_fixed = model_fixed.get_diagnostics()
        
    entropy_fixed = diag_fixed['attention/layer_0_avg_entropy']
    print(f"   -> Average Attention Entropy: {entropy_fixed:.4f}")

    # --- Comparison ---
    print("\n=== Results ===")
    print(f"Original Entropy: {entropy_orig:.4f}")
    print(f"Fixed Entropy:    {entropy_fixed:.4f}")
    
    if entropy_fixed < entropy_orig - 0.1:
        print("\n✅ HYPOTHESIS CONFIRMED: Xavier initialization significantly reduces entropy.")
    else:
        print("\n❌ HYPOTHESIS REJECTED: Entropy did not decrease significantly.")

if __name__ == "__main__":
    run_test()
