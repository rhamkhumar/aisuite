# Ollama

To use Ollama with `aisuite`, you'll need to ensure that your Ollama environment is properly configured. Ollama does not require an external API key but must be installed and running on your system or other devices.

## Setup Ollama
1. **Install Ollama**: Follow the installation instructions from [Ollama's official website](https://ollama.com/) to install the Ollama CLI on your system.
2. **Start the Ollama Service**: Ensure the Ollama service is running on your machine. You can start it with the following command:
    ```shell
    ollama serve
    ```
3. **Download the Models**: Ensure that the models you plan to use are downloaded and available in your Ollama instance. You can explore the list of supported models in the [Ollama Library](https://ollama.com/library)
4. **Default API URL**: If the `OLLAMA_API_URL` is not set or explicitly passed in the configuration, `aisuite` will default to using `"http://localhost:11434"`.

## Create a Chat Completion
The following example demonstrates how to create a chat completion using the Ollama provider:

```python
import aisuite as ai

client = ai.Client() 

provider = "ollama"
model_id = "qwq"  # Replace with the model name running on your device

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

## Get Text Embeddings
You can generate text embeddings for your input using the embedding models available on your Ollama instance:

```python
import aisuite as ai

client = ai.Client()

provider = "ollama"
embedding_model_id = "bge-large"

input = ["This is a test sentence."]

response = client.embeddings.create(
    model=f"{provider}:{embedding_model_id}",
    input=input,
)

print(response.data[0].embedding)
```