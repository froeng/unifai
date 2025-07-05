import json
from typing import Any, Dict


class OpenAIResponseAdapter:
    """
    Adapter to make dictionary responses behave like OpenAI response objects.
    Added a 'get' method to support attribute access similar to dict.get().
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, attr: str) -> Any:
        if attr in self._data:
            val = self._data[attr]
            if isinstance(val, dict):
                return OpenAIResponseAdapter(val)
            elif isinstance(val, list):
                return [OpenAIResponseAdapter(item) if isinstance(item, dict) else item for item in val]
            return val
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the internal data to a JSON-serializable dictionary."""
        return self._data

    def __repr__(self) -> str:
        return json.dumps(self._data, indent=2)

    def __iter__(self):
        """Make the object iterable for JSON serialization."""
        return iter(self._data.items())
    
    def __json__(self) -> Dict[str, Any]:
        """Explicitly define how to serialize this object to JSON."""
        return self.to_dict()
