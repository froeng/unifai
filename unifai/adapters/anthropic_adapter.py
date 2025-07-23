import json
from typing import Any, Dict, List, Optional
import anthropic

from .base import BaseAdapter
from ..utils import OpenAIResponseAdapter, extract_usage


def _extract_system_messages(messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
    """Extract and concatenate system messages, return system content and non-system messages."""
    system_messages = []
    non_system_messages = []
    
    for message in messages:
        if message.get("role") == "system":
            system_messages.append(message["content"])
        elif message.get("role") == "tool":
            # Convert OpenAI tool message format to Anthropic format
            tool_result = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", "unknown"),
                        "content": message["content"]
                    }
                ]
            }
            non_system_messages.append(tool_result)
        else:
            non_system_messages.append(message)
    
    system_content = "\n\n".join(system_messages) if system_messages else ""
    return system_content, non_system_messages


def _build_anthropic_kwargs(model: str, messages: List[Dict[str, str]], 
                           max_tokens: int, temperature: float, 
                           system_content: str) -> Dict[str, Any]:
    """Build base Anthropic API kwargs."""
    anthropic_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    if system_content:
        anthropic_kwargs["system"] = system_content
    
    return anthropic_kwargs


def _handle_response_format(anthropic_kwargs: Dict[str, Any], response_format: Any) -> None:
    """Handle response format configuration for structured output."""
    if response_format is None:
        return
        
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
        anthropic_kwargs["tool_choice"] = {"type": "tool", "name": "build_result"}


def _handle_tools(anthropic_kwargs: Dict[str, Any], tools: List[Dict[str, Any]], **kwargs) -> None:
    """Handle tools configuration."""
    if not tools:
        return
        
    anthropic_tools = []
    for tool in tools:
        if tool["type"] == "function":
            anthropic_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "input_schema": tool["function"]["parameters"]
            }
            anthropic_tools.append(anthropic_tool)
    
    if anthropic_tools:
        anthropic_kwargs["tools"] = anthropic_tools
        tool_choice = kwargs.get("tool_choice", None)
        if tool_choice and tool_choice != "auto":
            if tool_choice == "required":
                anthropic_kwargs["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                func_name = tool_choice["function"]["name"]
                if any(t["name"] == func_name for t in anthropic_tools):
                    anthropic_kwargs["tool_choice"] = {"type": "tool", "name": func_name}


def _create_structured_response(response: Any, response_format: Any) -> Dict[str, Any]:
    """Create OpenAI-format response for structured output."""
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

    return openai_format


def _create_standard_response(response: Any) -> Dict[str, Any]:
    """Create OpenAI-format response for standard chat completion."""
    usage = extract_usage(response.usage)
    openai_format = {
        "id": response.id,
        "object": "chat.completion",
        "model": response.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None
            },
            "index": 0,
            "finish_reason": response.stop_reason
        }],
        "usage": usage
    }
    
    # Handle different response types
    if hasattr(response, 'content') and response.content:
        content_block = response.content[0]
        
        if hasattr(content_block, 'type'):
            if content_block.type == "text":
                # Regular text response
                openai_format["choices"][0]["message"]["content"] = content_block.text
            elif content_block.type == "tool_use":
                # Tool call response
                tool_calls = []
                for i, block in enumerate(response.content):
                    if block.type == "tool_use":
                        tool_call = {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input) if hasattr(block, 'input') else "{}"
                            }
                        }
                        tool_calls.append(tool_call)
                
                if tool_calls:
                    openai_format["choices"][0]["message"]["tool_calls"] = tool_calls
                    openai_format["choices"][0]["message"]["content"] = ""
        else:
            # Fallback for text content
            if hasattr(content_block, 'text'):
                openai_format["choices"][0]["message"]["content"] = content_block.text
    
    return openai_format


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
        system_content, non_system_messages = _extract_system_messages(messages)
        anthropic_kwargs = _build_anthropic_kwargs(model, non_system_messages, max_tokens, temperature, system_content)
        
        _handle_response_format(anthropic_kwargs, response_format)
        _handle_tools(anthropic_kwargs, tools, **kwargs)

        response = self.client.messages.create(**anthropic_kwargs)
        openai_format = _create_structured_response(response, response_format)
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
               tools: List[Dict[str, Any]] = None,
               tool_choice: Optional[Any] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Adapt OpenAI's chat completion format to Anthropic's format.
        Supports both regular chat and tool calls.
        """
        system_content, non_system_messages = _extract_system_messages(messages)
        anthropic_kwargs = _build_anthropic_kwargs(model, non_system_messages, max_tokens, temperature, system_content)
        
        # Handle tools if provided
        if tools:
            _handle_tools(anthropic_kwargs, tools, tool_choice=tool_choice, **kwargs)
        
        response = self.client.messages.create(**anthropic_kwargs)
        openai_format = _create_standard_response(response)
        return OpenAIResponseAdapter(openai_format)

    def parse(self, model: str, messages: List[Dict[str, str]], 
              response_format: any = None,
              tools: List[Dict[str, Any]] = None,
              temperature: Optional[float] = 1.0,
              max_tokens: Optional[int] = 1000,
              **kwargs) -> Dict[str, Any]:
        """
        Adapt OpenAI's chat completion format with structured output to Anthropic's format.
        """
        system_content, non_system_messages = _extract_system_messages(messages)
        anthropic_kwargs = _build_anthropic_kwargs(model, non_system_messages, max_tokens, temperature, system_content)
        
        _handle_response_format(anthropic_kwargs, response_format)
        _handle_tools(anthropic_kwargs, tools, **kwargs)

        response = self.client.messages.create(**anthropic_kwargs)
        openai_format = _create_structured_response(response, response_format)
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
