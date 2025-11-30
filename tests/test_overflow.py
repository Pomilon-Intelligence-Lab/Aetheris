import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from aetheris.modules.expert import Expert
from aetheris.modules.moe import SparseMoELayer
from aetheris.config import AetherisConfig

class TestOverflow(unittest.TestCase):
    def setUp(self):
        self.config = AetherisConfig(
            vocab_size=100,
            d_model=128,
            n_layer=2,
            num_experts=2,
            top_k=1,
            d_ff=512,  # Large enough to potentially cause issues
            ssm_d_state=16,
            ssm_expand=2,
            max_seq_len=64
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_expert_overflow_protection(self):
        """Test if Expert handles large inputs without producing NaNs in float16"""
        expert = Expert(self.config.d_model, self.config.d_ff).to(self.device)
        # Manually cast weights to float16 to simulate mixed precision training environment
        expert.half()
        
        # Create a large input in float16 that would normally cause overflow in intermediate layers
        # The limit of float16 is ~65504. 
        # If w1 projects this up, it can easily exceed that.
        large_input = torch.ones(1, self.config.d_model, dtype=torch.float16).to(self.device) * 100.0
        
        # Force weights to be large to guarantee overflow if protection isn't working
        with torch.no_grad():
            expert.w1.weight.fill_(10.0)
            expert.w2.weight.fill_(0.1)

        # 100 * 10 = 1000. Sum over d_model(128) -> 128000. 
        # This summation happens in the matrix multiplication.
        # If the matmul internal accumulation is float16, it effectively overflows.
        
        output = expert(large_input)
        
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs")
        self.assertFalse(torch.isinf(output).any(), "Output contains Infs")

    def test_moe_accumulation_stability(self):
        """Test if MoE layer handles accumulation in float32"""
        moe = SparseMoELayer(self.config).to(self.device)
        moe.half()
        
        x = torch.randn(2, 10, self.config.d_model, dtype=torch.float16).to(self.device)
        
        # Pass through
        output, loss = moe(x)
        
        self.assertFalse(torch.isnan(output).any(), "MoE Output contains NaNs")
        self.assertEqual(output.dtype, torch.float16)

if __name__ == '__main__':
    unittest.main()
