"""Tests for core functionality."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from llmInvoker.core import llmInvoker, invoke
from llmInvoker.providers import BaseProvider
from llmInvoker.strategies import FailoverStrategy


class MockProvider(BaseProvider):
    """Mock provider for testing."""
    
    def __init__(self, name: str, models: list, should_fail: bool = False):
        super().__init__(name, models)
        self.should_fail = should_fail
        self.call_count = 0
    
    def _get_base_url(self) -> str:
        return "https://mock.api.com"
    
    async def _make_request(self, model: str, messages: list, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise Exception("Mock provider failure")
        
        return {
            "choices": [{
                "message": {
                    "content": f"Mock response from {self.name}:{model}"
                }
            }],
            "usage": {"total_tokens": 100}
        }


class TestLlmInvoker:
    """Test cases for llmInvoker."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        invoker = llmInvoker()
        assert invoker.strategy_name == "failover"
        assert invoker.max_retries == 3
        assert invoker.timeout == 30
        assert invoker.enable_history is True
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        invoker = llmInvoker(
            strategy="parallel",
            max_retries=5,
            timeout=60,
            enable_history=False
        )
        assert invoker.strategy_name == "parallel"
        assert invoker.max_retries == 5
        assert invoker.timeout == 60
        assert invoker.enable_history is False
    
    def test_configure_providers(self):
        """Test provider configuration."""
        invoker = llmInvoker()
        
        # Mock the config to avoid actual API key checks
        with patch('llmInvoker.core.validate_provider_config') as mock_validate:
            with patch('llmInvoker.core.create_provider') as mock_create:
                # Mock successful validation
                mock_validate.return_value = {
                    "github": ["gpt-4o"],
                    "google": ["gemini-2.0-flash"]
                }
                
                # Mock provider creation
                mock_provider1 = MagicMock()
                mock_provider1.name = "github"
                mock_provider2 = MagicMock()
                mock_provider2.name = "google"
                mock_create.side_effect = [mock_provider1, mock_provider2]
                
                result = invoker.configure_providers(
                    github=["gpt-4o"],
                    google=["gemini-2.0-flash"]
                )
                
                assert result is invoker  # Should return self for chaining
                assert len(invoker.providers) == 2
                assert "github" in invoker.provider_configs
                assert "google" in invoker.provider_configs
    
    def test_configure_from_string(self):
        """Test string-based configuration."""
        invoker = llmInvoker()
        
        with patch('llmInvoker.core.validate_provider_config') as mock_validate:
            with patch('llmInvoker.core.create_provider') as mock_create:
                # Mock successful validation
                mock_validate.return_value = {
                    "github": ["gpt-4o"],
                    "google": ["gemini-2.0-flash"]
                }
                
                # Mock provider creation
                mock_provider1 = MagicMock()
                mock_provider1.name = "github"
                mock_provider2 = MagicMock()
                mock_provider2.name = "google"
                mock_create.side_effect = [mock_provider1, mock_provider2]
                
                config_str = "github['gpt-4o'],google['gemini-2.0-flash']"
                result = invoker.configure_from_string(config_str)
                
                assert result is invoker
                assert "github" in invoker.provider_configs
                assert "google" in invoker.provider_configs
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """Test successful invocation."""
        invoker = llmInvoker(strategy="failover")
        
        # Add mock provider
        mock_provider = MockProvider("test", ["model1"], should_fail=False)
        invoker.providers = [(mock_provider, "model1")]
        
        response = await invoker.invoke(message="Test message")
        
        assert response['success'] is True
        assert response['provider'] == "test"
        assert response['model'] == "model1"
        assert mock_provider.call_count == 1
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_invoke_no_providers(self):
        """Test invocation with no providers configured."""
        invoker = llmInvoker()
        
        response = await invoker.invoke(message="Test message")
        
        assert response['success'] is False
        assert "No providers configured" in response['error']
    
    def test_invoke_sync(self):
        """Test synchronous invocation."""
        invoker = llmInvoker()
        
        # Mock the async invoke method
        async def mock_invoke(message, **kwargs):
            return {"success": True, "response": "Test response"}
        
        invoker.invoke = mock_invoke
        
        response = invoker.invoke_sync(message="Test message")
        assert response['success'] is True
    
    def test_get_provider_stats(self):
        """Test provider statistics."""
        invoker = llmInvoker()
        
        with patch('llmInvoker.core.config') as mock_config:
            mock_config.is_provider_configured.return_value = True
            
            # Add mock providers
            mock_provider1 = MockProvider("provider1", ["model1", "model2"])
            mock_provider2 = MockProvider("provider2", ["model3"])
            
            invoker.providers = [
                (mock_provider1, "model1"),
                (mock_provider1, "model2"),
                (mock_provider2, "model3")
            ]
            
            stats = invoker.get_provider_stats()
            
            assert stats['total_providers'] == 2
            assert stats['total_models'] == 3
            assert 'provider1' in stats['providers']
            assert 'provider2' in stats['providers']
    
    def test_clear_history(self):
        """Test history clearing."""
        invoker = llmInvoker(enable_history=True)
        
        # History should be initialized
        assert invoker.history is not None
        
        # Clear should not raise an error
        invoker.clear_history()
        
        # Test with disabled history
        invoker_no_history = llmInvoker(enable_history=False)
        invoker_no_history.clear_history()  # Should not raise error


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_invoke_function(self):
        """Test the invoke convenience function."""
        with patch('llmInvoker.core.llmInvoker') as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker.invoke_sync.return_value = {"success": True}
            mock_invoker_class.return_value = mock_invoker
            
            response = invoke(
                message="Test message",
                strategy="failover",
                providers={"github": ["gpt-4o"]}
            )
            
            # Verify invoker was created and configured
            mock_invoker_class.assert_called_once_with(strategy="failover")
            mock_invoker.configure_providers.assert_called_once_with(github=["gpt-4o"])
            mock_invoker.invoke_sync.assert_called_once_with("Test message")
    
    def test_invoke_function_with_defaults(self):
        """Test invoke function with default providers."""
        with patch('llmInvoker.core.llmInvoker') as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker.invoke_sync.return_value = {"success": True}
            mock_invoker_class.return_value = mock_invoker
            
            response = invoke(message="Test message")
            
            # Should use defaults when no providers specified
            mock_invoker.use_defaults.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
