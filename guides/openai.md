# OpenAI

To use OpenAI with `aisuite`, you’ll need an [OpenAI account](https://platform.openai.com/). After logging in, go to the [API Keys](https://platform.openai.com/account/api-keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export OPENAI_API_KEY="your-openai-api-key"
```

## Create a Chat Completion

Install the `openai` Python client:

Example with pip:
```shell
pip install openai
```

Example with poetry:
```shell
poetry add openai
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "openai"
model_id = "gpt-4-turbo"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

## Get Text Embeddings
You can generate text embeddings using OpenAI embedding models with the `aisuite` client:
```python
import aisuite as ai

client = ai.Client()

provider = "openai"
embedding_model_id = "text-embedding-ada-002"

input_text = ["This is an example sentence for embeddings."]

response = client.embeddings.create(
    model=f"{provider}:{embedding_model_id}",
    input=input_text,
)

embeddings = response.data[0].embedding
print("Embeddings:", embeddings)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
