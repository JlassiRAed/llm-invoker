"""Utility functions for the multiagent failover invoke library."""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_rate_limit_error(error_message: str) -> bool:
    """Detect if an error is related to rate limiting."""
    error_lower = error_message.lower()
    rate_limit_indicators = [
        'rate limit', 'rate_limit', 'rate-limit',
        'too many requests', 'quota exceeded', 
        'rate limited', 'rate limiting',
        '429', 'http 429',
        'requests per minute', 'requests per hour',
        'usage limit', 'api limit',
        'throttled', 'throttling'
    ]
    
    return any(indicator in error_lower for indicator in rate_limit_indicators)


def extract_wait_time(error_message: str) -> int:
    """Extract wait time from rate limit error message."""
    import re
    
    # Look for patterns like "retry after 60 seconds" or "wait 1800 seconds"
    patterns = [
        r'retry.{0,10}(\d+).{0,10}second',
        r'wait.{0,10}(\d+).{0,10}second', 
        r'(\d+).{0,10}second.{0,10}retry',
        r'retry-after:\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_message.lower())
        if match:
            return int(match.group(1))
    
    # Default wait time if no specific time found
    return 60


async def handle_rate_limit(response, default_delay: int = 60) -> None:
    """Handle rate limiting responses with exponential backoff."""
    retry_after = default_delay
    
    # Try to get retry-after header
    if hasattr(response, 'headers'):
        retry_after_header = response.headers.get('retry-after') or response.headers.get('Retry-After')
        if retry_after_header:
            try:
                retry_after = int(retry_after_header)
            except ValueError:
                pass
    
    logger.warning(f"Rate limit hit. Should trigger failover instead of waiting {retry_after} seconds...")
    # Instead of sleeping, raise an exception to trigger failover
    raise Exception(f"Rate limit exceeded. Retry after {retry_after} seconds.")


def log_token_usage(provider: str, model: str, usage: Dict[str, Any]) -> None:
    """Log token usage for monitoring purposes."""
    if not usage:
        return
    
    # Extract token counts (different providers have different formats)
    prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
    completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
    
    logger.info(
        f"Token usage - Provider: {provider}, Model: {model}, "
        f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
    )
    
    # Send to LangSmith if configured
    try:
        from .config import config
        if config.langsmith_api_key:
            _send_to_langsmith(provider, model, usage)
    except Exception as e:
        logger.debug(f"Failed to send token usage to LangSmith: {e}")


def _send_to_langsmith(provider: str, model: str, usage: Dict[str, Any]) -> None:
    """Send token usage data to LangSmith."""
    try:
        from langsmith import Client
        from .config import config
        
        client = Client(api_key=config.langsmith_api_key)
        
        # Create a run for token tracking
        client.create_run(
            name=f"{provider}:{model}_token_usage",
            run_type="llm",
            inputs={"provider": provider, "model": model},
            outputs={"usage": usage},
            project_name=config.langsmith_project
        )
    except ImportError:
        logger.debug("LangSmith not available for token usage tracking")
    except Exception as e:
        logger.debug(f"Error sending to LangSmith: {e}")


def log_strategy_execution(strategy: str, provider_count: int) -> None:
    """Log strategy execution for monitoring."""
    logger.info(f"Executing {strategy} strategy with {provider_count} providers")


def format_provider_response(
    provider: str, 
    model: str, 
    response: Any, 
    success: bool = True, 
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Format provider response in a standardized way."""
    return {
        'provider': provider,
        'model': model,
        'response': response,
        'success': success,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }


def validate_provider_config(provider_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate provider configurations."""
    from .config import config
    
    validated_configs = {}
    errors = []
    
    for provider_name, models in provider_configs.items():
        # Check if provider is supported
        from .providers import PROVIDERS
        if provider_name not in PROVIDERS:
            errors.append(f"Unsupported provider: {provider_name}")
            continue
        
        # Check if API key is configured
        if not config.is_provider_configured(provider_name):
            errors.append(f"No API key configured for provider: {provider_name}")
            continue
        
        # Validate models list
        if not isinstance(models, list) or not models:
            errors.append(f"Provider {provider_name} must have a non-empty list of models")
            continue
        
        validated_configs[provider_name] = models
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return validated_configs


def parse_provider_config(config_str: str) -> Dict[str, list]:
    """Parse provider configuration string into structured format."""
    # Example: "github['gpt-4o','grok-3'],google['gemini-2.0-flash']"
    import re
    
    configs = {}
    pattern = r'(\w+)\[(.*?)\]'
    matches = re.findall(pattern, config_str)
    
    for provider, models_str in matches:
        # Parse models list
        models = [model.strip().strip("'\"") for model in models_str.split(',')]
        models = [model for model in models if model]  # Remove empty strings
        configs[provider] = models
    
    return configs


def get_default_provider_configs() -> Dict[str, list]:
    """Get default provider configurations."""
    return {
        "github": ["gpt-4o", "gpt-4o-mini"],
        "google": ["gemini-2.0-flash-exp", "gemini-1.5-pro"],
        "openrouter": ["deepseek/deepseek-r1", "meta-llama/llama-3.2-3b-instruct:free"],
        "openai": ["gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-haiku-20241022", "claude-3-haiku-20240307"]
    }


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count for text."""
    # Simple approximation: ~4 characters per token
    return len(text) // 4


def truncate_for_context(text: str, max_tokens: int = 4000) -> str:
    """Truncate text to fit within token limit."""
    estimated_tokens = estimate_token_count(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Truncate to approximate token limit
    target_chars = max_tokens * 4
    if len(text) > target_chars:
        return text[:target_chars] + "..."
    
    return text


def setup_langsmith_tracing() -> None:
    """Setup LangSmith tracing if configured."""
    try:
        from .config import config
        if config.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
            logger.info("LangSmith tracing enabled")
    except Exception as e:
        logger.debug(f"Failed to setup LangSmith tracing: {e}")


def create_error_response(error_msg: str, provider: str = "", model: str = "") -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        'success': False,
        'error': error_msg,
        'provider': provider,
        'model': model,
        'timestamp': datetime.now().isoformat()
    }