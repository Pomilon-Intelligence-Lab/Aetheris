import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from aetheris.config import AetherisConfig
from aetheris.model import HybridMambaMoE

class TestHybridMambaMoE(unittest.TestCase):
    def setUp(self):
        self.config = AetherisConfig(
            vocab_size=100,
            d_model=32,
            n_layer=4,
            num_experts=2,
            top_k=1,
            d_ff=64,
            ssm_d_state=8,
            ssm_expand=2,
            max_seq_len=64
        )
        self.model = HybridMambaMoE(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def test_forward_pass(self):
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        output = self.model(input_ids)
        
        self.assertIn('logits', output)
        self.assertEqual(output['logits'].shape, (batch_size, seq_len, self.config.vocab_size))

    def test_forward_pass_with_labels(self):
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = input_ids.clone()
        
        output = self.model(input_ids, labels=labels)
        
        self.assertIn('loss', output)
        self.assertIn('ce_loss', output)
        self.assertIn('aux_loss', output)
        self.assertIn('logits', output)
        
        self.assertTrue(output['loss'] > 0)

if __name__ == '__main__':
    unittest.main()
