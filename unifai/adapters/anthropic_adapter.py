from typing import Any, Dict, List, Optional
import anthropic

from .base import BaseAdapter
from ..utils import OpenAIResponseAdapter, extract_usage


class BetaCompletionsAdapter:
    def __init__(self, client):
        self.client = client

    def parse(self, model: str, messages: List[Dict[str, str]], 
              response_format: any = None,
              tools: List[Dict[str, Any]] = None,
              temperature: Optional[float] = 1.0,
              max_tokens: Optional[int] = 1000,
              **kwargs) -> Dict[str, Any]:
        """
        Adapt OpenAI's beta chat completion format to Anthropic's structured output.
        """
        # Extract system messages and concatenate them
        system_messages = []
        non_system_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                system_messages.append(message["content"])
            else:
                non_system_messages.append(message)
        
        # Concatenate all system messages
        system_content = "\n\n".join(system_messages) if system_messages else ""
        
        anthropic_kwargs = {
            "model": model,
            "messages": non_system_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add system parameter if we have system content
        if system_content:
            anthropic_kwargs["system"] = system_content

        if response_format is not None:
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                anthropic_kwargs["response_format"] = {"type": "json_object"}
            else:
                # assume default response format is a pydantic BaseModel
                anthropic_kwargs["tools"] = [
                    {
                        "name": "build_result",
                        "description": "build the object",
                        "input_schema": response_format.model_json_schema()
                    }
                ]
                anthropic_kwargs["tool_choice"]={"type": "tool", "name": "build_result"}

        if tools:
            anthropic_tools = [
                {"function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"]
                }} for tool in tools if tool["type"] == "function"
            ]
            if anthropic_tools:
                anthropic_kwargs["tools"] = anthropic_tools
                tool_choice = kwargs.get("tool_choice", None)
                if tool_choice and tool_choice != "auto":
                    if tool_choice == "required":
                        anthropic_kwargs["tool_choice"] = "required"
                    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                        func_name = tool_choice["function"]["name"]
                        if any(t["function"]["name"] == func_name for t in anthropic_tools):
                            anthropic_kwargs["tool_choice"] = {"type": "function", "name": func_name}

        response = self.client.messages.create(**anthropic_kwargs)
        usage = extract_usage(response.usage)
        openai_format = {
            "id": response.id,
            "object": "chat.completion",
            "model": response.model,
            "choices": [{
                "message": {}
            }],
            "usage": usage
        }

        # Handle parsed response format
        if response_format and hasattr(response_format, 'model_validate'):
            # Parse the tool response into the Pydantic model
            tool_input = response.content[0].input
            parsed_obj = response_format.model_validate(tool_input)
            openai_format["choices"][0]["message"]["parsed"] = parsed_obj
        else:
            openai_format["choices"][0]["message"]["parsed"] = response.content[0].input

        return OpenAIResponseAdapter(openai_format)


class BetaChatAdapter:
    def __init__(self, client):
        self.client = client
        self.completions = BetaCompletionsAdapter(client)


class AnthropicBetaAdapter:
    """
    Adapter for Anthropic's structured output capabilities to mimic OpenAI's beta features.
    """
    def __init__(self, client):
        self.client = client
        self.chat = BetaChatAdapter(client)


class ChatCompletionsAdapter:
    def __init__(self, client):
        self.client = client
        
    def create(self, model: str, messages: List[Dict[str, str]], 
               max_tokens: Optional[int] = 1000, 
               temperature: Optional[float] = 1.0,
               **kwargs) -> Dict[str, Any]:
        """
        Adapt OpenAI's chat completion format to Anthropic's format.
        """
        # Extract system messages and concatenate them
        system_messages = []
        non_system_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                system_messages.append(message["content"])
            else:
                non_system_messages.append(message)
        
        # Concatenate all system messages
        system_content = "\n\n".join(system_messages) if system_messages else ""
        
        anthropic_kwargs = {
            "model": model,
            "messages": non_system_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add system parameter if we have system content
        if system_content:
            anthropic_kwargs["system"] = system_content
        
        response = self.client.messages.create(**anthropic_kwargs)

        usage = extract_usage(response.usage)
        openai_format = {
            "id": response.id,
            "object": "chat.completion",
            "model": response.model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.content[0].text
                },
                "index": 0,
                "finish_reason": response.stop_reason
            }],
            "usage": usage
        }

        return OpenAIResponseAdapter(openai_format)


class ChatAdapter:
    def __init__(self, client):
        self.client = client
        self.completions = ChatCompletionsAdapter(client)


class ModelsAdapter:
    def __init__(self, client):
        self.client = client

    def list(self):
        try:
            available_models = self.client.models.list()
            models_data = [
                {"id": model.id, "object": "model", "created": 0}
                for model in available_models.data
            ]
            return OpenAIResponseAdapter({
                "object": "list",
                "data": models_data
            })
        except Exception as e:
            print(f"Error listing Anthropic models: {e}")
            return OpenAIResponseAdapter({
                "object": "list",
                "data": []
            })


class AnthropicAdapter(BaseAdapter):
    """
    Adapter class to make Anthropic's API interface compatible with OpenAI's.
    Acts as a shim between the two APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        super().__init__(client)
        self.chat = ChatAdapter(self.client)
        self.models = ModelsAdapter(self.client)
        self.beta = AnthropicBetaAdapter(self.client)
    
    def create_chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Create a chat completion using Anthropic format."""
        return self.chat.completions.create(model=model, messages=messages, **kwargs)
    
    def list_models(self) -> Any:
        """List available models."""
        return self.models.list()
    
    def get_default_model(self) -> str:
        """Get the default model for this adapter."""
        return "claude-3-haiku-20240307"  # Default Anthropic model
