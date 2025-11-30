import pytest
from unittest.mock import MagicMock, patch
from aetheris.inference import InferenceEngine

@pytest.fixture
def mock_model():
    with patch("aetheris.inference.HybridMambaMoE") as MockModel:
        mock_instance = MockModel.return_value
        # Mock model output
        mock_instance.to.return_value = mock_instance
        mock_instance.eval.return_value = None
        
        # Mock forward pass
        mock_output = MagicMock()
        # Shape: (batch_size, seq_len, vocab_size)
        mock_output.__getitem__.return_value = torch.randn(1, 1, 50257) 
        # Actually we need 'logits' key access
        mock_instance.return_value = {'logits': torch.randn(1, 10, 50257)}
        
        yield mock_instance

@pytest.fixture
def mock_tokenizer():
    with patch("aetheris.inference.get_tokenizer") as mock_get_tokenizer:
        mock_tok = MagicMock()
        mock_tok.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tok.decode.return_value = "token"
        mock_tok.eos_token_id = 50256
        mock_get_tokenizer.return_value = mock_tok
        yield mock_tok

@pytest.fixture
def mock_utils():
    with patch("aetheris.inference.load_latest_checkpoint") as mock_load:
        yield mock_load

import torch

def test_inference_initialization(mock_model, mock_tokenizer, mock_utils):
    engine = InferenceEngine(config_path="configs/default.yaml")
    assert engine.model is not None
    assert engine.tokenizer is not None
    mock_utils.assert_called_once()

def test_generate_full(mock_model, mock_tokenizer, mock_utils):
    engine = InferenceEngine()
    
    # Mock model output for generation loop
    # We need to ensure the model returns logits of correct shape
    # The loop calls model(generated_ids)
    
    # Let's mock the actual model call inside generate
    engine.model.config.torch_dtype = torch.float32
    
    # We need to return a dict with logits
    # Shape: (batch, seq_len, vocab_size)
    engine.model.side_effect = lambda x: {'logits': torch.randn(1, x.shape[1], 50257)}

    output = engine.generate_full("test prompt", max_new_tokens=5)
    assert isinstance(output, str)
    assert len(output) > 0

def test_generate_stream(mock_model, mock_tokenizer, mock_utils):
    engine = InferenceEngine()
    engine.model.config.torch_dtype = torch.float32
    engine.model.side_effect = lambda x: {'logits': torch.randn(1, x.shape[1], 50257)}

    generator = engine.generate("test prompt", max_new_tokens=5, stream=True)
    tokens = list(generator)
    assert len(tokens) == 5
    assert all(isinstance(t, str) for t in tokens)
