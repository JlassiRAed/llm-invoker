"""Provider implementations for different AI model APIs."""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import requests
import aiohttp
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from .config import config
from .utils import handle_rate_limit, log_token_usage, is_rate_limit_error, extract_wait_time

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Base class for all AI model providers."""
    
    def __init__(self, name: str, models: List[str], api_key: Optional[str] = None):
        self.name = name
        self.models = models
        self.api_key = api_key or config.get_provider_key(name)
        self.base_url = self._get_base_url()
        
    @abstractmethod
    def _get_base_url(self) -> str:
        """Get the base URL for the provider API."""
        pass
    
    @abstractmethod
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make an API request to the provider."""
        pass
    
    async def invoke(self, model: str, messages: Union[str, List[Dict], Dict], **kwargs) -> Dict[str, Any]:
        """Invoke a model with the given messages."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            # Single message dict
            messages = [messages]
        
        # Handle multimodal content
        processed_messages = self._process_multimodal_messages(messages)
        
        try:
            response = await self._make_request(model, processed_messages, **kwargs)
            
            # Log token usage if available
            if 'usage' in response:
                log_token_usage(self.name, model, response['usage'])
            
            return {
                'provider': self.name,
                'model': model,
                'response': response,
                'success': True
            }
        except Exception as e:
            error_str = str(e)
            
            # Detect rate limiting and raise exception to trigger failover
            if is_rate_limit_error(error_str):
                logger.warning(f"Rate limit detected for {self.name}:{model}. Triggering failover.")
                # Raise exception to trigger failover instead of waiting
                raise Exception(f"Rate limit exceeded for {self.name}:{model}")
            
            logger.error(f"Provider {self.name}:{model} error: {e}")
            return {
                'provider': self.name,
                'model': model,
                'error': str(e),
                'success': False
            }
    
    def _process_multimodal_messages(self, messages: List[Dict]) -> List[Dict]:
        """Process messages to handle multimodal content (text, images, etc.)."""
        processed = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                # Multimodal content - handle different content types
                processed_content = []
                for content_item in msg['content']:
                    if isinstance(content_item, dict):
                        if content_item.get('type') == 'text':
                            processed_content.append(content_item)
                        elif content_item.get('type') == 'image_url':
                            # Handle image content
                            processed_content.append(content_item)
                        elif content_item.get('type') == 'image':
                            # Convert to standard format
                            processed_content.append({
                                'type': 'image_url',
                                'image_url': content_item.get('url') or content_item.get('image_url')
                            })
                    else:
                        # String content
                        processed_content.append({
                            'type': 'text',
                            'text': str(content_item)
                        })
                
                processed.append({
                    'role': msg['role'],
                    'content': processed_content
                })
            else:
                # Simple text content
                processed.append(msg)
        
        return processed


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    def _get_base_url(self) -> str:
        return "https://api.openai.com/v1"
    
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 429:
                    await handle_rate_limit(response)
                response.raise_for_status()
                return await response.json()


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""
    
    def _get_base_url(self) -> str:
        return "https://api.anthropic.com/v1"
    
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages format for Anthropic
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        data = {
            "model": model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            **{k: v for k, v in kwargs.items() if k != "max_tokens"}
        }
        
        if system_msg:
            data["system"] = system_msg
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 429:
                    await handle_rate_limit(response)
                response.raise_for_status()
                return await response.json()


class GitHubProvider(BaseProvider):
    """GitHub Models API provider."""
    
    def _get_base_url(self) -> str:
        return "https://models.inference.ai.azure.com"
    
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 429:
                    await handle_rate_limit(response)
                response.raise_for_status()
                return await response.json()


class GoogleProvider(BaseProvider):
    """Google Generative AI provider."""
    
    def _get_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"
    
    async def invoke(self, model: str, messages: Union[str, List[Dict], Dict], **kwargs) -> Dict[str, Any]:
        """Override invoke to handle Google-specific response format."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        
        # Handle multimodal content
        processed_messages = self._process_multimodal_messages(messages)
        
        try:
            raw_response = await self._make_request(model, processed_messages, **kwargs)
            
            # Extract text from Google's response format
            response_text = self._extract_text_from_response(raw_response)
            
            # Log token usage if available
            if 'usageMetadata' in raw_response:
                usage = raw_response['usageMetadata']
                log_token_usage(self.name, model, {
                    'prompt_tokens': usage.get('promptTokenCount', 0),
                    'completion_tokens': usage.get('candidatesTokenCount', 0),
                    'total_tokens': usage.get('totalTokenCount', 0)
                })
            
            return {
                'provider': self.name,
                'model': model,
                'response': response_text,
                'success': True,
                'raw_response': raw_response
            }
        except Exception as e:
            error_str = str(e)
            
            if is_rate_limit_error(error_str):
                logger.warning(f"Rate limit detected for {self.name}:{model}. Triggering failover.")
                raise Exception(f"Rate limit exceeded for {self.name}:{model}")
            
            logger.error(f"Provider {self.name}:{model} error: {e}")
            return {
                'provider': self.name,
                'model': model,
                'error': error_str,
                'success': False
            }
    
    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text from Google's response format."""
        try:
            if 'candidates' in response and len(response['candidates']) > 0:
                candidate = response['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text'].strip()
            
            # Fallback extraction
            return str(response.get('text', str(response)))
            
        except Exception as e:
            logger.warning(f"Failed to extract text from Google response: {e}")
            return str(response)
    
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/models/{model}:generateContent"
        params = {"key": self.api_key}
        
        # Convert messages to Google format - Fix for proper conversation handling
        contents = []
        for msg in messages:
            # Google expects specific role mapping
            if msg["role"] == "user":
                contents.append({
                    "parts": [{"text": msg["content"]}],
                    "role": "user"
                })
            elif msg["role"] == "assistant":
                contents.append({
                    "parts": [{"text": msg["content"]}],
                    "role": "model"
                })
            elif msg["role"] == "system":
                # For system messages, prepend to first user message or create user message
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"][0]["text"] = f"{msg['content']}\n\n{contents[-1]['parts'][0]['text']}"
                else:
                    contents.append({
                        "parts": [{"text": msg["content"]}],
                        "role": "user"
                    })
        
        # Ensure we have at least one message and it's a user message
        if not contents:
            contents = [{"parts": [{"text": "Hello"}], "role": "user"}]
        elif contents[0]["role"] != "user":
            contents.insert(0, {"parts": [{"text": "Hello"}], "role": "user"})
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 1.0),
                "topP": kwargs.get("top_p", 0.95),
                "topK": kwargs.get("top_k", 40),
                "maxOutputTokens": kwargs.get("max_tokens", 8192)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, params=params) as response:
                if response.status == 429:
                    await handle_rate_limit(response)
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Google API error {response.status}: {error_text}")
                    raise aiohttp.ClientResponseError(
                        response.request_info, 
                        response.history, 
                        status=response.status, 
                        message=f"Google API error: {error_text}"
                    )
                return await response.json()


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider."""
    
    def _get_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"
    
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/multiagent_failover_invoke",
            "X-Title": "llm-Invoker"
        }
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 429:
                    await handle_rate_limit(response)
                response.raise_for_status()
                return await response.json()


class HuggingFaceProvider(BaseProvider):
    """Hugging Face Inference API provider."""
    
    def _get_base_url(self) -> str:
        return "https://api-inference.huggingface.co/models"
    
    async def _make_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Convert messages to text for HuggingFace
        text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        data = {
            "inputs": text,
            "parameters": kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 429:
                    await handle_rate_limit(response)
                response.raise_for_status()
                return await response.json()


# Provider registry
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "github": GitHubProvider,
    "google": GoogleProvider,
    "openrouter": OpenRouterProvider,
    "huggingface": HuggingFaceProvider,
}


def create_provider(provider_name: str, models: List[str]) -> BaseProvider:
    """Create a provider instance."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    provider_class = PROVIDERS[provider_name]
    return provider_class(provider_name, models)