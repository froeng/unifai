from typing import List, Tuple, Optional, Any

from .adapters import OpenAIAdapter, AnthropicAdapter
from .fallback import FallbackProxy


class Client:
    """
    Attempts to initialize multiple models in priority order and provides
    a dynamic fallback for nested attributes and methods.
    """
    
    def __init__(self, *models: str):
        if not models:
            models = ("local", "gpt-4o-mini")
        self._models = models

    def clients(self) -> List[Tuple[Any, str, str]]:
        """Get all available clients."""
        _clients = []
        for m in self._models:
            model_name, client = self._initialize_model(m)
            if client is not None:
                _clients.append((client, model_name, m))
        return _clients

    def _initialize_model(self, model_name: str) -> Tuple[Optional[str], Optional[Any]]:
        """Initialize a specific model adapter."""
        _model_name = model_name
        try:
            if model_name.startswith("claude"):
                adapter = AnthropicAdapter()
                return (model_name, adapter)
            elif model_name == "local":
                adapter = OpenAIAdapter(base_url="http://localhost:8000/v1")
                _model_name = adapter.get_default_model()
                return (_model_name, adapter.client)
            else:
                adapter = OpenAIAdapter()
                return (model_name, adapter.client)
        except Exception as e:
            print(f"Failed to initialize '{model_name}': {e}")
            return (None, None)

    def __getattr__(self, item: str) -> Any:
        """Provide fallback attribute access across all clients."""
        new_pairs = []
        last_exc = None

        for client, model_name, model_category in self.clients():
            try:
                sub_attr = getattr(client, item)
                new_pairs.append((sub_attr, model_name, model_category))
            except AttributeError as e:
                print(f"Failed on model '{model_name}': {e}")
                last_exc = e

        if not new_pairs:
            raise AttributeError(f"None of the clients implement '{item}'.") from last_exc

        return FallbackProxy(new_pairs)

    def get_active_model(self) -> Optional[str]:
        """Get the first working model."""
        for client, model_name, model_category in self.clients():
            try:
                client.chat.completions.create(
                    model=model_name,
                    max_tokens=20,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "respond with 'world'"},
                        {"role": "user", "content": "hello"}
                    ]
                )
                return model_name
            except Exception as e:
                print(e)
                continue
        return None
