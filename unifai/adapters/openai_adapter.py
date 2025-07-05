from typing import Any, Dict, List, Optional
from openai import OpenAI as _OpenAI

from .base import BaseAdapter
from ..utils import OpenAIResponseAdapter


class LocalBetaProxy:
    """
    Mimics the 'beta' attribute on OpenAI clients for local models.
    """
    def __init__(self, client):
        self._client = client


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if base_url:  # Local model
            client = _OpenAI(api_key="EMPTY", base_url=base_url)
            if not hasattr(client, "beta"):
                client.beta = LocalBetaProxy(client)
        else:  # Remote OpenAI
            client = _OpenAI(api_key=api_key)
        
        super().__init__(client)
    
    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Create a chat completion using OpenAI format."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
    
    def list_models(self) -> Any:
        """List available models."""
        return self.client.models.list()
    
    def get_default_model(self) -> str:
        """Get the default model for this adapter."""
        try:
            models = self.client.models.list()
            return models.data[0].id if models.data else "gpt-3.5-turbo"
        except:
            return "gpt-3.5-turbo"
