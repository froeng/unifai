# UnifAI - Unified AI Interface

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified, extensible interface for multiple AI providers (OpenAI, Anthropic, and local models) with automatic fallback support.

## Features

- 🔄 **Automatic Fallback**: Seamlessly switch between providers when one fails
- 🎯 **Unified Interface**: Same API for OpenAI, Anthropic, and local models
- 🧩 **Extensible**: Easy to add new AI providers
- 🔧 **Type Safety**: Full TypeScript-style type hints and Pydantic support
- ⚡ **Beta Features**: Support for structured outputs and tool calling
- 🔒 **Backward Compatible**: Drop-in replacement for existing OpenAI code

## Quick Start

### Installation

```bash
pip install openai anthropic pydantic
```

### Basic Usage

```python
from unifai import Client

# Initialize with your preferred models (in priority order)
client = Client("gpt-4o-mini", "claude-3-haiku-20240307", "local")

# Use like a standard OpenAI client
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, world!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Structured Output (Beta)

```python
from pydantic import BaseModel
from unifai import Client

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

client = Client("gpt-4o-mini", "claude-3-haiku-20240307")

response = client.beta.chat.completions.parse(
    messages=[{"role": "user", "content": "Analyze: This movie is fantastic!"}],
    response_format=AnalysisResult
)

result = response.choices[0].message.parsed
print(f"Sentiment: {result.sentiment} (confidence: {result.confidence})")
```

## Supported Providers

### OpenAI
- ✅ Chat completions
- ✅ Model listing
- ✅ Beta features (structured outputs)
- ✅ Tool calling

### Anthropic
- ✅ Chat completions (via adapter)
- ✅ Model listing (via adapter)
- ✅ Beta features (structured outputs via adapter)
- ✅ Tool calling (via adapter)

### Local Models
- ✅ OpenAI-compatible servers (like Ollama with OpenAI compatibility)
- ✅ Automatic model detection
- ✅ Beta feature emulation

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Client Configuration

```python
from unifai import Client

# Default: tries local, then gpt-4o-mini
client = Client()

# Custom model priority
client = Client("claude-3-haiku-20240307", "gpt-4o-mini", "local")

# Check which model is active
active_model = client.get_active_model()
print(f"Using: {active_model}")
```

## Advanced Usage

### Direct Provider Access

```python
from unifai.adapters import OpenAIAdapter, AnthropicAdapter

# Use specific providers directly
openai = OpenAIAdapter()
anthropic = AnthropicAdapter()

# Both provide the same interface
for provider in [openai, anthropic]:
    response = provider.create_chat_completion(
        model="auto",  # Uses provider's default model
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
```

### Custom Fallback Logic

```python
from unifai import Client

# The client automatically handles failures and retries with the next provider
client = Client("gpt-4o-mini", "claude-3-haiku-20240307", "local")

try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Complex reasoning task"}],
        max_tokens=1000
    )
except Exception as e:
    print(f"All providers failed: {e}")
```

## Architecture

UnifAI uses a modular architecture with clean separation of concerns:

```
unifai/
├── client.py           # Main Client with fallback logic
├── fallback.py         # FallbackProxy for method chaining
├── adapters/           # Provider-specific adapters
│   ├── base.py         # Abstract base adapter
│   ├── openai_adapter.py
│   └── anthropic_adapter.py
└── utils/              # Shared utilities
    ├── helpers.py      # Message conversion, usage extraction
    └── response_adapter.py  # Response normalization
```

## Migration Guide

UnifAI is designed as a drop-in replacement for OpenAI Python client code:

```python
# Before (OpenAI only)
from openai import OpenAI
client = OpenAI()

# After (UnifAI with multiple providers)
from unifai import Client
client = Client("gpt-4o-mini", "claude-3-haiku-20240307")

# Same API, more resilient
response = client.chat.completions.create(...)
```

## Contributing

### Adding New Providers

1. Create a new adapter in `unifai/adapters/`:

```python
from .base import BaseAdapter

class MyProviderAdapter(BaseAdapter):
    def create_chat_completion(self, model, messages, **kwargs):
        # Implement your provider's chat completion
        pass
    
    def list_models(self):
        # Implement model listing
        pass
```

2. Update `client.py` to handle your provider's initialization

### Development Setup

```bash
git clone <repository>
cd unifai
pip install -e .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

### v1.0.0
- 🎉 Initial release
- ✅ OpenAI and Anthropic support
- ✅ Local model support
- ✅ Automatic fallback
- ✅ Beta features and structured outputs
- ✅ Full backward compatibility

## Support

- 📖 [Documentation](unifai/README.md)
- 🐛 [Issue Tracker](https://github.com/your-username/unifai/issues)
- 💬 [Discussions](https://github.com/your-username/unifai/discussions)

---

**UnifAI** - One interface, multiple AI providers, maximum reliability.
