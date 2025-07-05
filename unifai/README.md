# UnifAI - Unified AI Interface

This is a refactored version of the AI module that provides a unified interface for both OpenAI and Anthropic APIs.

## Architecture

The refactored code is organized into several modules for better maintainability:

### Core Structure

```
unifai/
├── __init__.py           # Main exports
├── AI.py                 # Compatibility layer (maintains backward compatibility)
├── client.py            # Main Client class with fallback logic
├── fallback.py          # FallbackProxy for handling multiple API endpoints
├── adapters/            # API adapters
│   ├── __init__.py
│   ├── base.py          # Base adapter interface
│   ├── openai_adapter.py   # OpenAI API adapter
│   └── anthropic_adapter.py # Anthropic API adapter
└── utils/               # Utility functions
    ├── __init__.py
    ├── helpers.py       # Helper functions (convert_openai_messages, extract_usage)
    └── response_adapter.py # OpenAIResponseAdapter
```

### Key Components

#### 1. Client (`client.py`)
The main entry point that:
- Manages multiple AI providers (OpenAI, Anthropic, local models)
- Provides fallback functionality when one provider fails
- Maintains a unified interface across different APIs

#### 2. Adapters (`adapters/`)
- **BaseAdapter**: Abstract base class defining the common interface
- **OpenAIAdapter**: Handles OpenAI API calls (including local models)
- **AnthropicAdapter**: Converts Anthropic API to OpenAI-compatible format

#### 3. Fallback System (`fallback.py`)
- **FallbackProxy**: Automatically tries multiple providers until one succeeds
- Handles method calls and attribute access across different clients

#### 4. Utilities (`utils/`)
- **OpenAIResponseAdapter**: Makes response objects behave consistently
- **Helper functions**: Message conversion, usage extraction, etc.

## Usage

### Basic Usage (unchanged from original)

```python
from unifai import Client

# Initialize with default models (local, gpt-4o-mini)
client = Client()

# Or specify custom models
client = Client("claude-3-haiku-20240307", "gpt-4o-mini", "local")

# Use the client (automatically falls back through available models)
response = client.chat.completions.create(
    model="gpt-4o-mini",  # This will be overridden by available models
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Advanced Usage

```python
# Check which model is currently active
active_model = client.get_active_model()
print(f"Currently using: {active_model}")

# Direct adapter usage
from unifai.adapters import OpenAIAdapter, AnthropicAdapter

# Use OpenAI directly
openai_adapter = OpenAIAdapter()
response = openai_adapter.create_chat_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Use Anthropic directly  
anthropic_adapter = AnthropicAdapter()
response = anthropic_adapter.create_chat_completion(
    model="claude-3-haiku-20240307", 
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Benefits of Refactoring

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new AI providers by implementing BaseAdapter
3. **Maintainability**: Code is organized and easier to understand
4. **Testability**: Each component can be tested independently
5. **Backward Compatibility**: Existing code continues to work unchanged

## Migration Guide

No changes are required for existing code. The refactored version maintains full backward compatibility through the `AI.py` compatibility layer.

If you want to use the new modular structure directly:

```python
# Old way (still works)
from unifai.AI import Client

# New way (recommended for new code)
from unifai import Client
```

## Adding New Providers

To add a new AI provider:

1. Create a new adapter in `adapters/` that inherits from `BaseAdapter`
2. Implement the required methods: `create_chat_completion()`, `list_models()`
3. Add initialization logic in `Client._initialize_model()`
4. Update the `adapters/__init__.py` to export the new adapter

Example:

```python
# adapters/my_provider_adapter.py
from .base import BaseAdapter

class MyProviderAdapter(BaseAdapter):
    def __init__(self, api_key=None):
        # Initialize your provider's client
        super().__init__(my_provider_client)
    
    def create_chat_completion(self, model, messages, **kwargs):
        # Implement chat completion
        pass
    
    def list_models(self):
        # Implement model listing
        pass
```
