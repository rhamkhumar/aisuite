import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.ollama_provider import OllamaProvider


@pytest.fixture(autouse=True)
def set_api_url_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OLLAMA_API_URL", "http://localhost:11434")


def test_completion():
    """Test that completions request successfully."""

    user_greeting = "Howdy!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "best-model-ever"
    chosen_temperature = 0.77
    response_text_content = "mocked-text-response-from-ollama-model"

    ollama = OllamaProvider()
    mock_response = {"message": {"content": response_text_content}}

    with patch(
        "httpx.post",
        return_value=MagicMock(status_code=200, json=lambda: mock_response),
    ) as mock_post:
        response = ollama.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={
                "model": selected_model,
                "messages": message_history,
                "stream": False,
                "temperature": chosen_temperature,
            },
            timeout=30,
        )

        assert response.choices[0].message.content == response_text_content


def test_single_embedding_creation():
    """Test that embeddings request with a single input successfully."""
    ollama = OllamaProvider()
    mock_response = {
        "embeddings": [
            [
                0.010071029,
                -0.0017594862,
                0.05007221,
                0.04692972,
                0.054916814,
                0.008599704,
                0.105441414,
                -0.025878139,
                0.12958129,
                0.031952348,
            ]
        ],
    }

    with patch(
        "httpx.post",
        return_value=MagicMock(status_code=200, json=lambda: mock_response),
    ) as mock_post:
        response = ollama.embeddings_create(
            input=["Howdy!"],
            model="best-model-ever",
        )

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "best-model-ever", "input": ["Howdy!"]},
            timeout=30,
        )

        assert response.data[0].embedding == [
            0.010071029,
            -0.0017594862,
            0.05007221,
            0.04692972,
            0.054916814,
            0.008599704,
            0.105441414,
            -0.025878139,
            0.12958129,
            0.031952348,
        ]


def test_multiple_embeddings_creation():
    """Test that embeddings request with multiple inputs successfully."""
    ollama = OllamaProvider()
    mock_response = {
        "embeddings": [
            [
                0.010071029,
                -0.0017594862,
                0.05007221,
                0.04692972,
                0.054916814,
                0.008599704,
                0.105441414,
                -0.025878139,
                0.12958129,
                0.031952348,
            ],
            [
                -0.0098027075,
                0.06042469,
                0.025257962,
                -0.006364387,
                0.07272725,
                0.017194884,
                0.09032035,
                -0.051705178,
                0.09951512,
                0.09072481,
            ],
        ],
    }

    with patch(
        "httpx.post",
        return_value=MagicMock(status_code=200, json=lambda: mock_response),
    ) as mock_post:
        response = ollama.embeddings_create(
            input=["Howdy!", "Hello!"],
            model="best-model-ever",
        )

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "best-model-ever", "input": ["Howdy!", "Hello!"]},
            timeout=30,
        )
        assert response.data[0].embedding == [
            0.010071029,
            -0.0017594862,
            0.05007221,
            0.04692972,
            0.054916814,
            0.008599704,
            0.105441414,
            -0.025878139,
            0.12958129,
            0.031952348,
        ]
        assert response.data[1].embedding == [
            -0.0098027075,
            0.06042469,
            0.025257962,
            -0.006364387,
            0.07272725,
            0.017194884,
            0.09032035,
            -0.051705178,
            0.09951512,
            0.09072481,
        ]
