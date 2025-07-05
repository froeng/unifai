"""
Compatibility layer for the refactored AI module.
This maintains backward compatibility while using the new modular structure.
"""

# Import the new client
from .client import Client

# For backward compatibility, expose the main classes
# that were previously in this file
from .utils import OpenAIResponseAdapter, convert_openai_messages, extract_usage
from .adapters import AnthropicAdapter, OpenAIAdapter
from .fallback import FallbackProxy

# Maintain the old interface for existing code
__all__ = [
    'Client',
    'OpenAIResponseAdapter', 
    'AnthropicAdapter',
    'OpenAIAdapter',
    'FallbackProxy',
    'convert_openai_messages',
    'extract_usage'
]
