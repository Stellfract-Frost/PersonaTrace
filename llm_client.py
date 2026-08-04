"""
LLM client module.
Supports any OpenAI-compatible API endpoint.
Not tied to a specific model provider.
"""
import json
import re
import logging
from typing import List, Dict, Any, Optional
import requests
from .config import PersonaConfig
logger = logging.getLogger(__name__)
class LLMClient:
    """OpenAI-compatible LLM client.
    Works with OpenAI, DeepSeek, Moonshot, Together AI, or any service
    that implements the OpenAI Chat Completions API.
    Example:
        >>> config = PersonaConfig(llm_api_key="sk-...", llm_base_url="https://api.openai.com/v1")
        >>> client = LLMClient(config)
        >>> response = client.chat([{"role": "user", "content": "Hello"}])
    """
    def __init__(self, config: PersonaConfig):
        self.config = config
        self.api_key = config.llm_api_key
        self.base_url = config.llm_base_url.rstrip("/")
        self.model = config.llm_model
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        """Send a chat request and return the text response.
        Args:
            messages: message list in [{"role": "system/user/assistant", "content": "..."}] format
            max_tokens: max tokens to generate
            temperature: sampling temperature
            model: override the default model
        Returns:
            LLM-generated text response
        Raises:
            RuntimeError: if the API call fails
        """
        if not self.api_key:
            raise RuntimeError(
                "LLM API key not configured. Set the PERSONA_LLM_API_KEY environment variable "
                "or pass it via PersonaConfig(llm_api_key=...)."
            )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error("LLM API call failed: %s", e)
            raise RuntimeError(f"LLM API call failed: {e}") from e
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a chat request and parse the JSON response.
        Automatically extracts ```json code blocks or bare JSON objects.
        Args:
            messages: message list
            max_tokens: max tokens to generate
            model: override the default model
        Returns:
            Parsed JSON dict, or None if parsing fails
        """
        text = self.chat(messages, max_tokens=max_tokens, model=model)
        return extract_json(text)
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from text.
    Handles the following formats:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - bare {...}
    Args:
        text: text that may contain JSON
    Returns:
        Parsed dict, or None on failure
    """
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to extract the first JSON object
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
