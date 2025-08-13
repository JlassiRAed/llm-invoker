"""Configuration management for multiagent failover invoke library."""
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for API keys and default settings."""
    
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # LangSmith Configuration
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "multiagent_failover_poc")
        
        # Default Settings
        self.default_strategy = os.getenv("DEFAULT_STRATEGY", "failover")
        self.default_max_retries = int(os.getenv("DEFAULT_MAX_RETRIES", "3"))
        self.default_timeout = int(os.getenv("DEFAULT_TIMEOUT", "30"))
        self.default_rate_limit_delay = int(os.getenv("DEFAULT_RATE_LIMIT_DELAY", "60"))
    
    def get_provider_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        provider_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "github": self.github_token,
            "google": self.google_api_key,
            "huggingface": self.huggingface_api_key,
            "openrouter": self.openrouter_api_key,
        }
        return provider_keys.get(provider.lower())
    
    def get_all_provider_keys(self) -> Dict[str, Optional[str]]:
        """Get all configured provider API keys."""
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "github": self.github_token,
            "google": self.google_api_key,
            "huggingface": self.huggingface_api_key,
            "openrouter": self.openrouter_api_key,
            "langsmith": self.langsmith_api_key,
        }
    
    def is_provider_configured(self, provider: str) -> bool:
        """Check if a provider is properly configured."""
        key = self.get_provider_key(provider)
        return key is not None and key.strip() != ""


# Global configuration instance
config = Config()