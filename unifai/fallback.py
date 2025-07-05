import inspect
from typing import List, Tuple, Any, Callable


class FallbackProxy:
    """
    Handles fallback logic for nested attributes or final method calls.
    """
    
    def __init__(self, pairs: List[Tuple[Any, str, str]]):
        # pairs is a list of (object, model_name, model_category)
        self._pairs = pairs

    def __getattr__(self, item: str) -> Any:
        new_pairs = []
        last_exc = None
        
        for obj, model_name, model_category in self._pairs:
            try:
                sub_attr = getattr(obj, item)
                new_pairs.append((sub_attr, model_name, model_category))
            except AttributeError as e:
                print(f"Failed on model '{model_name}': {e}")
                last_exc = e

        if not new_pairs:
            raise AttributeError(f"Attribute '{item}' not found on any fallback client.") from last_exc

        if all(callable(pair[0]) for pair in new_pairs):
            return self._wrap_callable(new_pairs, item)
        return FallbackProxy(new_pairs)

    def _wrap_callable(self, pairs: List[Tuple[Callable, str, str]], method_name: str) -> Callable:
        def wrapper(*args, **kwargs):
            last_exc = None
            for method_obj, model_name, model_category in pairs:
                try:
                    # Check if "model" is a required argument or in kwargs
                    sig = inspect.signature(method_obj)
                    params = sig.parameters
                    if "model" in params and "model" not in kwargs:
                        kwargs["model"] = model_name
                    return method_obj(*args, **kwargs)
                except Exception as e:
                    print(f"Failed on model '{model_name}': {e}")
                    last_exc = e
            raise RuntimeError(f"All fallback attempts failed for '{method_name}': {last_exc}")
        return wrapper
