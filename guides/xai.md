# XAI

To use XAI with `aisuite`, you’ll need an [XAI account](https://console.x.ai/). After logging in, go to the [API Keys](https://console.x.ai/team/) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export XAI_API_KEY="your-xai-api-key"
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
client.provider_configs={
    "xai": {"api_key": "XAI_API_KEY"},

}
models = ["xai:grok-beta"]
messages = [
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},

]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.75
    )
    print(response.choices[0].message.content)

```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
