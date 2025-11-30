import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from aetheris.api.server import app, get_engine
import aetheris.api.server

# Mock the engine globally
@pytest.fixture
def mock_engine():
    with patch("aetheris.api.server.engine") as mock_eng:
        # Mock generate_full
        mock_eng.generate_full.return_value = "This is a generated response."
        
        # Mock generate (streaming)
        def mock_stream(*args, **kwargs):
            yield "This "
            yield "is "
            yield "streamed."
        mock_eng.generate.side_effect = mock_stream
        
        # Need to ensure get_engine returns this mock
        # We can also just set aetheris.api.server.engine
        aetheris.api.server.engine = mock_eng
        yield mock_eng

client = TestClient(app)

def test_list_models(mock_engine):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "aetheris-hybrid-mamba-moe"

def test_chat_completions_non_stream(mock_engine):
    payload = {
        "model": "aetheris-hybrid-mamba-moe",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "This is a generated response."

def test_chat_completions_stream(mock_engine):
    payload = {
        "model": "aetheris-hybrid-mamba-moe",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    # SSE format checking
    assert "text/event-stream" in response.headers["content-type"]
    
    # We can iterate over the response lines to check content
    content = ""
    for line in response.iter_lines():
        if line:
            # TestClient iter_lines yields strings, not bytes, unless configured otherwise
            # or depending on the version. If it's bytes, we decode. If it's str, we don't.
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            
            if line.startswith("data: ") and line != "data: [DONE]":
                import json
                chunk = json.loads(line[6:])
                if chunk["choices"][0]["delta"].get("content"):
                    content += chunk["choices"][0]["delta"]["content"]
    
    assert content == "This is streamed."

def test_completions(mock_engine):
    payload = {
        "model": "aetheris-hybrid-mamba-moe",
        "prompt": "Once upon a time",
        "max_tokens": 10
    }
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"] == "This is a generated response."
