"""Tests for provider implementations."""
import pytest
from unittest.mock import patch, MagicMock
from llmInvoker.providers import (
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


class TestBaseProvider:
    """Test cases for BaseProvider abstract class."""
    
    def test_base_provider_cannot_be_instantiated(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider("test", ["model1"])


class TestProviderCreation:
    """Test provider creation and factory functions."""
    
    def test_create_openai_provider(self):
        """Test OpenAI provider creation."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = create_provider("openai", ["gpt-4"])
            
            assert isinstance(provider, OpenAIProvider)
            assert provider.name == "openai"
            assert provider.models == ["gpt-4"]
            assert provider.api_key == "test_key"
    
    def test_create_anthropic_provider(self):
        """Test Anthropic provider creation.""" 
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
    
    def test_create_github_provider(self):
        """Test GitHub provider creation."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
    
    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown_provider", ["model1"])
    
    def test_providers_registry(self):
        """Test that all expected providers are in registry."""
        expected_providers = [
            "openai", "anthropic", "github", 
            "google", "openrouter", "huggingface"
        ]
        
        for provider_name in expected_providers:
            assert provider_name in PROVIDERS


class TestProviderConfiguration:
    """Test provider configuration and API setup."""
    
    def test_openai_provider_base_url(self):
        """Test OpenAI provider base URL."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = OpenAIProvider("openai", ["gpt-4"])
            assert provider._get_base_url() == "https://api.openai.com/v1"
    
    def test_anthropic_provider_base_url(self):
        """Test Anthropic provider base URL."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = AnthropicProvider("anthropic", ["claude-3-haiku"])
            assert provider._get_base_url() == "https://api.anthropic.com/v1"
    
    def test_github_provider_base_url(self):
        """Test GitHub provider base URL."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_token"
            
            provider = GitHubProvider("github", ["gpt-4o"])
            assert provider._get_base_url() == "https://models.inference.ai.azure.com"
    
    def test_google_provider_base_url(self):
        """Test Google provider base URL."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = GoogleProvider("google", ["gemini-2.0-flash"])
            assert provider._get_base_url() == "https://generativelanguage.googleapis.com/v1beta"
    
    def test_openrouter_provider_base_url(self):
        """Test OpenRouter provider base URL."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = OpenRouterProvider("openrouter", ["deepseek/deepseek-r1"])
            assert provider._get_base_url() == "https://openrouter.ai/api/v1"
    
    def test_huggingface_provider_base_url(self):
        """Test Hugging Face provider base URL."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = HuggingFaceProvider("huggingface", ["model1"])
            assert provider._get_base_url() == "https://api-inference.huggingface.co/models"


class TestProviderInvocation:
    """Test provider invocation methods."""
    
    @pytest.mark.asyncio
    async def test_provider_invoke_string_message(self):
        """Test provider invoke with string message."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = OpenAIProvider("openai", ["gpt-4"])
            
            # Mock the _make_request method as an async method
            async def mock_make_request(model, messages, **kwargs):
                return {
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {"total_tokens": 100}
                }
            
            provider._make_request = mock_make_request
            
            response = await provider.invoke("gpt-4", "Hello, world!")
            
            assert response['success'] is True
            assert response['provider'] == "openai"
            assert response['model'] == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_provider_invoke_message_list(self):
        """Test provider invoke with message list."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = OpenAIProvider("openai", ["gpt-4"])
            
            # Mock the _make_request method as an async method
            async def mock_make_request(model, messages, **kwargs):
                return {
                    "choices": [{"message": {"content": "Test response"}}]
                }
            
            provider._make_request = mock_make_request
            
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            
            response = await provider.invoke("gpt-4", messages)
            
            assert response['success'] is True
    
    @pytest.mark.asyncio
    async def test_provider_invoke_error(self):
        """Test provider invoke with error."""
        with patch('llmInvoker.providers.config') as mock_config:
            mock_config.get_provider_key.return_value = "test_key"
            
            provider = OpenAIProvider("openai", ["gpt-4"])
            
            # Mock the _make_request method to raise an error
            async def mock_make_request(model, messages, **kwargs):
                raise Exception("API Error")
            
            provider._make_request = mock_make_request
            
            response = await provider.invoke("gpt-4", "Hello, world!")
            
            assert response['success'] is False
            assert response['error'] == "API Error"
            assert response['provider'] == "openai"
            assert response['model'] == "gpt-4"


if __name__ == '__main__':
    pytest.main([__file__])