from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAdapter(ABC):
    """Base adapter interface for AI clients."""
    
    def __init__(self, client):
        self.client = client
    
    @abstractmethod
    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Create a chat completion."""
        pass
    
    @abstractmethod
    def list_models(self) -> Any:
        """List available models."""
        pass
