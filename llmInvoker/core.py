"""Core functionality for multiagent failover invoke library."""
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple

from .config import config
from .providers import create_provider, BaseProvider, PROVIDERS
from .strategies import create_strategy, BaseStrategy, STRATEGIES
from .history import ConversationHistory
from .utils import (
    validate_provider_config, 
    parse_provider_config, 
    get_default_provider_configs,
    setup_langsmith_tracing,
    create_error_response
)


class llmInvoker:
    """Main class for handling multi-agent model invocation with failover strategies."""
    
    def __init__(
        self,
        strategy: str = "failover",
        max_retries: int = 3,
        timeout: int = 30,
        enable_history: bool = True,
        enable_langsmith: bool = True
    ):
        """
        Initialize the llmInvoker.
        
        Args:
            strategy: The invocation strategy ("failover", "parallel", "round_robin")
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for each request
            enable_history: Whether to maintain conversation history
            enable_langsmith: Whether to enable LangSmith tracing
        """
        self.strategy_name = strategy
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_history = enable_history
        
        # Initialize strategy
        self.strategy = create_strategy(
            strategy, 
            max_retries=max_retries, 
            timeout=timeout
        )
        
        # Initialize conversation history
        if enable_history:
            self.history = ConversationHistory()
        else:
            self.history = None
        
        # Setup LangSmith tracing
        if enable_langsmith:
            setup_langsmith_tracing()
        
        # Provider configurations
        self.provider_configs: Dict[str, List[str]] = {}
        self.providers: List[Tuple[BaseProvider, str]] = []
    
    def configure_providers(self, **provider_configs) -> 'llmInvoker':
        """
        Configure providers and their models.
        
        Example:
            invoker.configure_providers(
                github=["gpt-4o", "gpt-4o-mini"],
                google=["gemini-2.0-flash"],
                openrouter=["deepseek/deepseek-r1"]
            )
        """
        # Validate configurations
        validated_configs = validate_provider_config(provider_configs)
        self.provider_configs = validated_configs
        
        # Create provider instances
        self.providers = []
        for provider_name, models in validated_configs.items():
            provider = create_provider(provider_name, models)
            for model in models:
                self.providers.append((provider, model))
        
        return self
    
    def configure_from_string(self, config_str: str) -> 'llmInvoker':
        """
        Configure providers from a string format.
        
        Example:
            invoker.configure_from_string("github['gpt-4o','gpt-4o-mini'],google['gemini-2.0-flash']")
        """
        provider_configs = parse_provider_config(config_str)
        return self.configure_providers(**provider_configs)
    
    def use_defaults(self) -> 'llmInvoker':
        """Use default provider configurations."""
        default_configs = get_default_provider_configs()
        
        # Filter to only configured providers
        available_configs = {}
        for provider_name, models in default_configs.items():
            if config.is_provider_configured(provider_name):
                available_configs[provider_name] = models
        
        if not available_configs:
            raise ValueError("No providers are configured. Please set API keys in .env file.")
        
        return self.configure_providers(**available_configs)
    
    async def invoke(
        self, 
        message: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke the configured strategy with the given message.
        
        Args:
            message: The user message (string, single message dict, or list of message dicts)
                    Supports multimodal content with images, text, etc.
            **kwargs: Additional parameters for the model (temperature, max_tokens, etc.)
        
        Returns:
            Dictionary containing the response and metadata
        """
        if not self.providers:
            return create_error_response("No providers configured. Call configure_providers() first.")
        
        try:
            # Execute strategy
            result = await self.strategy.execute(self.providers, message, **kwargs)
            
            # Update history if enabled and strategy has history
            if self.enable_history and self.history and hasattr(self.strategy, 'history'):
                self.history = self.strategy.history
            
            return result
            
        except Exception as e:
            return create_error_response(f"Invocation failed: {str(e)}")
    
    def invoke_sync(
        self, 
        message: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous version of invoke.
        
        Args:
            message: The user message (string, single message dict, or list of message dicts)
                    Supports multimodal content with images, text, etc.
            **kwargs: Additional parameters for the model
        
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Run async method in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.invoke(message, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            return create_error_response(f"Sync invocation failed: {str(e)}")
    
    def get_history(self) -> Optional[ConversationHistory]:
        """Get the conversation history."""
        return self.history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        if self.history:
            self.history.clear()
        if hasattr(self.strategy, 'history'):
            self.strategy.history.clear()
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about configured providers."""
        stats = {
            'total_providers': len(set(p[0].name for p in self.providers)),
            'total_models': len(self.providers),
            'strategy': self.strategy_name,
            'providers': {}
        }
        
        for provider, model in self.providers:
            if provider.name not in stats['providers']:
                stats['providers'][provider.name] = {
                    'models': [],
                    'configured': config.is_provider_configured(provider.name)
                }
            stats['providers'][provider.name]['models'].append(model)
        
        return stats
    
    def export_history(self, file_path: str) -> None:
        """Export conversation history to a file."""
        if not self.history:
            raise ValueError("History is not enabled or empty")
        self.history.export_to_json(file_path)
    
    def import_history(self, file_path: str) -> None:
        """Import conversation history from a file."""
        if not self.history:
            self.history = ConversationHistory()
        self.history.import_from_json(file_path)


# Convenience function for quick usage
def invoke(
    message: Union[str, List[Dict], Dict],
    strategy: str = "failover",
    providers: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for quick invocation.
    
    Args:
        message: The user message (string, single message dict, or list of message dicts)
                Supports multimodal content with images, text, etc.
        strategy: The invocation strategy ("failover", "parallel", "round_robin")
        providers: Dictionary of provider configurations
        **kwargs: Additional parameters for the model
    
    Example:
        response = invoke(
            "Hello, how are you?",
            strategy="failover",
            providers={"github": ["gpt-4o"], "google": ["gemini-2.0-flash"]}
        )
        
        # Multimodal example
        response = invoke(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,..."}
                ]
            },
            strategy="failover",
            providers={"github": ["gpt-4o"]}
        )
    """
    invoker = llmInvoker(strategy=strategy)
    
    if providers:
        invoker.configure_providers(**providers)
    else:
        invoker.use_defaults()
    
    return invoker.invoke_sync(message, **kwargs)


# Convenience functions for different strategies
def invoke_failover(
    message: Union[str, List[Dict], Dict],
    providers: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for failover strategy."""
    return invoke(message, strategy="failover", providers=providers, **kwargs)


def invoke_parallel(
    message: Union[str, List[Dict], Dict],
    providers: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for parallel strategy."""
    return invoke(message, strategy="parallel", providers=providers, **kwargs)