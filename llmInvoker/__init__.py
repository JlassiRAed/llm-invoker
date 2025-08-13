"""
MultiAgent Failover Invoke Library

A Python library for managing multi-agent model invocation with failover strategies.
Designed for POC development with automatic provider switching and conversation history management.
"""

__version__ = "0.1.0"
__author__ = "Jlassi Raed"
__email__ = "raed.jlassi@etudiant-enit.utm.tn"

# Core functionality
from .core import llmInvoker, invoke, invoke_failover, invoke_parallel

# Provider management
from .providers import (
    BaseProvider, 
    OpenAIProvider, 
    AnthropicProvider, 
    GitHubProvider, 
    GoogleProvider, 
    OpenRouterProvider, 
    HuggingFaceProvider,
    create_provider,
    PROVIDERS
)

# Strategy management
from .strategies import (
    BaseStrategy,
    FailoverStrategy,
    ParallelStrategy,
    RoundRobinStrategy,
    create_strategy,
    STRATEGIES
)

# History and utilities
from .history import ConversationHistory, ConversationEntry
from .config import config, Config
from .utils import (
    setup_langsmith_tracing,
    validate_provider_config,
    parse_provider_config,
    get_default_provider_configs
)

# Main exports
__all__ = [
    # Core
    "llmInvoker",
    "invoke",
    "invoke_failover", 
    "invoke_parallel",
    
    # Providers
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider", 
    "GitHubProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "HuggingFaceProvider",
    "create_provider",
    "PROVIDERS",
    
    # Strategies
    "BaseStrategy",
    "FailoverStrategy",
    "ParallelStrategy", 
    "RoundRobinStrategy",
    "create_strategy",
    "STRATEGIES",
    
    # History
    "ConversationHistory",
    "ConversationEntry",
    
    # Configuration
    "config",
    "Config",
    
    # Utilities
    "setup_langsmith_tracing",
    "validate_provider_config",
    "parse_provider_config",
    "get_default_provider_configs",
]