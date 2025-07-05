from typing import Any, Dict, List, Optional, Tuple


def convert_openai_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Convert OpenAI messages to Anthropic's expected format."""
    system_message = None
    converted = []
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            converted.append({
                "role": "user" if msg["role"] == "user" else "assistant",
                "content": msg["content"]
            })
    return system_message, converted


def extract_usage(usage: Any) -> Dict[str, int]:
    """Extract token usage safely from an object or dict."""
    prompt = usage.input_tokens
    completion = usage.output_tokens
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion
    }
