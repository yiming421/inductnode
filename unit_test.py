import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add

# --- Definition of the AttentionPool Module (Copied from your provided code) ---
class AttentionPool(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=1, dp=0.2):
        super(AttentionPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # Dimension for attention calculation per head
        self.nhead = nhead
        self.dp = dp

        # Linear transformation for attention score calculation
        self.lin = nn.Linear(in_channels, nhead * out_channels)
        # Attention mechanism (scores features of size out_channels)
        self.att = nn.Linear(out_channels, 1)

    def forward(self, context_h_input, context_y, num_classes=None):
        # Determine num_classes if not provided
        if num_classes is None:
            if context_y.numel() == 0: # Handle empty context_y
                # Output shape should be [0, nhead * in_channels]
                return torch.empty(0, self.nhead * self.in_channels, 
                                   device=context_h_input.device, dtype=context_h_input.dtype)
            if context_y.max().numel() == 0 : # if context_y contains only one element e.g. tensor([0])
                 num_classes = 1 if context_y.numel() > 0 else 0
            else:
                 num_classes = context_y.max().item() + 1 if context_y.numel() > 0 else 0
        
        # Handle empty input context_h_input
        if context_h_input.numel() == 0:
            # If context_h is empty, but classes are expected (e.g. from context_y or num_classes)
            # return zeros of the correct output shape.
            return torch.zeros(num_classes, self.nhead * self.in_channels, 
                               device=context_h_input.device, dtype=context_h_input.dtype)

        # Apply dropout to the original features that will be pooled
        context_h_ori_dropout = F.dropout(context_h_input, p=self.dp, training=self.training)
        
        # Transform features for attention calculation
        context_h_transformed = self.lin(context_h_ori_dropout) # [N_ctx, nhead * out_channels]
        # Reshape for multi-head processing
        context_h_transformed = context_h_transformed.view(-1, self.nhead, self.out_channels) # [N_ctx, nhead, out_channels]
        
        # Calculate raw attention logits
        att_score = self.att(context_h_transformed).squeeze(-1) # [N_ctx, nhead]
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        
        # Normalize attention scores per class, per head
        att_weights = scatter_softmax(att_score, context_y, dim=0) # [N_ctx, nhead]
        
        # Weight original in_channels features using the attention weights
        # context_h_ori_dropout: [N_ctx, in_channels]
        # att_weights.unsqueeze(-1): [N_ctx, nhead, 1]
        # Broadcasting context_h_ori_dropout to [N_ctx, 1, in_channels]
        # Resulting att_h: [N_ctx, nhead, in_channels]
        att_h = context_h_ori_dropout.unsqueeze(1) * att_weights.unsqueeze(-1)
        
        # Initialize tensor for pooled features
        # Shape: [num_classes, nhead, in_channels]
        pooled_h = torch.zeros(num_classes, self.nhead, self.in_channels, 
                               device=context_h_input.device, dtype=context_h_input.dtype)
        
        # Aggregate weighted features per class, per head
        pooled_h = scatter_add(att_h, context_y, dim=0, out=pooled_h) # [num_classes, nhead, in_channels]
        
        # Concatenate features from all heads
        # Reshape to [num_classes, nhead * in_channels]
        final_h = pooled_h.view(num_classes, self.nhead * self.in_channels)
        return final_h

# --- Unit Test Class ---
class TestAttentionPool(unittest.TestCase):

    def setUp(self):
        # Common setup for tests, if any (e.g., device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # For reproducibility of dropout for some tests if needed, though usually we test structure.
        torch.manual_seed(42)

    def test_basic_single_head(self):
        in_channels = 3
        out_channels_att = 5 # Dimension for attention calculation
        nhead = 1
        
        model = AttentionPool(in_channels, out_channels_att, nhead=nhead, dp=0.0).to(self.device)
        model.eval() # Turn off dropout for predictable output

        # Mock data: 4 nodes, 2 classes
        # Nodes 0, 2 belong to class 0; Nodes 1, 3 belong to class 1
        context_h = torch.tensor([[1., 1., 1.], 
                                  [2., 2., 2.], 
                                  [3., 3., 3.], 
                                  [4., 4., 4.]], device=self.device, dtype=torch.float)
        context_y = torch.tensor([0, 1, 0, 1], device=self.device, dtype=torch.long)
        num_classes = 2

        output = model(context_h, context_y, num_classes)
        
        self.assertEqual(output.shape, (num_classes, nhead * in_channels))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs")
        # Check if values are plausible (e.g., not all zeros if inputs are non-zero,
        # though exact values depend on learned weights)
        self.assertTrue(output.abs().sum() > 0, "Output is all zeros, check attention/aggregation.")


    def test_multi_head(self):
        in_channels = 4
        out_channels_att = 6
        nhead = 2
        
        model = AttentionPool(in_channels, out_channels_att, nhead=nhead, dp=0.0).to(self.device)
        model.eval()

        # Mock data: 5 nodes, 3 classes
        context_h = torch.randn(5, in_channels, device=self.device, dtype=torch.float)
        context_y = torch.tensor([0, 1, 2, 0, 1], device=self.device, dtype=torch.long)
        num_classes = 3

        output = model(context_h, context_y, num_classes)
        
        self.assertEqual(output.shape, (num_classes, nhead * in_channels))
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(output.abs().sum() > 0)

    def test_num_classes_inference(self):
        in_channels = 2
        out_channels_att = 3
        nhead = 1
        
        model = AttentionPool(in_channels, out_channels_att, nhead=nhead, dp=0.0).to(self.device)
        model.eval()

        context_h = torch.randn(6, in_channels, device=self.device, dtype=torch.float)
        # Classes 0, 1, 2, 3. So num_classes should be 4.
        context_y = torch.tensor([0, 1, 2, 3, 0, 1], device=self.device, dtype=torch.long) 
        
        output = model(context_h, context_y) # num_classes not provided
        
        expected_num_classes = context_y.max().item() + 1
        self.assertEqual(output.shape, (expected_num_classes, nhead * in_channels))

    def test_single_class_single_node(self):
        in_channels = 2
        out_channels_att = 3
        nhead = 1
        model = AttentionPool(in_channels, out_channels_att, nhead=nhead, dp=0.0).to(self.device)
        model.eval()

        context_h = torch.tensor([[0.5, 0.5]], device=self.device, dtype=torch.float)
        context_y = torch.tensor([0], device=self.device, dtype=torch.long)
        num_classes = 1
        
        output = model(context_h, context_y, num_classes)
        self.assertEqual(output.shape, (num_classes, nhead * in_channels))
        # For a single node in a class, its features should be perfectly reconstructed (scaled by attention=1)
        # This assumes dropout is off and weights are somewhat reasonable.
        # The exact value depends on the initialized weights of lin and att,
        # but the sum of attention weights for that single node in its class must be 1.
        # So, the output should be context_h if the internal transformations are identity-like for this test.
        # More robustly, check that it's not NaN and has the right shape.
        self.assertFalse(torch.isnan(output).any())


    def test_training_mode_dropout(self):
        in_channels = 10
        out_channels_att = 5
        nhead = 2
        dp = 0.5 # Non-zero dropout
        
        model = AttentionPool(in_channels, out_channels_att, nhead=nhead, dp=dp).to(self.device)
        model.train() # Set to training mode

        context_h = torch.randn(20, in_channels, device=self.device, dtype=torch.float)
        context_y = torch.randint(0, 3, (20,), device=self.device, dtype=torch.long)
        num_classes = 3

        # Run multiple times and check if outputs are different (due to dropout)
        # This is a probabilistic test, but with enough features/nodes, differences should appear.
        output1 = model(context_h, context_y, num_classes)
        output2 = model(context_h, context_y, num_classes)
        
        self.assertEqual(output1.shape, (num_classes, nhead * in_channels))
        self.assertEqual(output2.shape, (num_classes, nhead * in_channels))
        
        # If dropout is active and effective, outputs should differ.
        # This might not always hold if all dropped-out features were zero or weights cancel out,
        # but it's a reasonable check for typical random data.
        if context_h.numel() > 0 and dp > 0: # Only if dropout can have an effect
             self.assertFalse(torch.allclose(output1, output2), "Outputs are identical in train mode with dropout, check dropout application.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)