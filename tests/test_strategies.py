"""Tests for strategy implementations."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from llmInvoker.strategies import (
    FailoverStrategy, 
    ParallelStrategy, 
    RoundRobinStrategy,
    create_strategy
)
from llmInvoker.providers import BaseProvider


class MockProvider(BaseProvider):
    """Mock provider for testing strategies."""
    
    def __init__(self, name: str, models: list, should_fail: bool = False):
        super().__init__(name, models)
        self.should_fail = should_fail
        self.call_count = 0
    
    def _get_base_url(self) -> str:
        return "https://mock.api.com"
    
    async def _make_request(self, model: str, messages, **kwargs):
        """Implement the abstract method."""
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
    
    async def invoke(self, model: str, messages, **kwargs):
        self.call_count += 1
        if self.should_fail:
            return {
                'success': False,
                'error': f'Mock error from {self.name}',
                'provider': self.name,
                'model': model
            }
        
        return {
            'success': True,
            'response': {
                'choices': [{
                    'message': {
                        'content': f'Mock response from {self.name}:{model}'
                    }
                }]
            },
            'provider': self.name,
            'model': model
        }


class TestFailoverStrategy:
    """Test cases for FailoverStrategy."""
    
    @pytest.mark.asyncio
    async def test_failover_success_first_provider(self):
        """Test successful invocation with first provider."""
        strategy = FailoverStrategy(max_retries=1, timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=False)
        provider2 = MockProvider("provider2", ["model2"], should_fail=False)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        response = await strategy.execute(providers_models, "Test message")
        
        assert response['success'] is True
        assert response['provider'] == "provider1"
        assert response['model'] == "model1"
        assert provider1.call_count == 1
        assert provider2.call_count == 0  # Should not call second provider
    
    @pytest.mark.asyncio
    async def test_failover_switch_on_failure(self):
        """Test failover to second provider when first fails."""
        strategy = FailoverStrategy(max_retries=1, timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=True)
        provider2 = MockProvider("provider2", ["model2"], should_fail=False)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        response = await strategy.execute(providers_models, "Test message")
        
        assert response['success'] is True
        assert response['provider'] == "provider2"
        assert response['model'] == "model2"
        assert provider1.call_count == 1
        assert provider2.call_count == 1
    
    @pytest.mark.asyncio
    async def test_failover_all_providers_fail(self):
        """Test when all providers fail."""
        strategy = FailoverStrategy(max_retries=1, timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=True)
        provider2 = MockProvider("provider2", ["model2"], should_fail=True)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        response = await strategy.execute(providers_models, "Test message")
        
        assert response['success'] is False
        assert "All providers failed" in response['error']
        assert provider1.call_count == 1
        assert provider2.call_count == 1


class TestParallelStrategy:
    """Test cases for ParallelStrategy."""
    
    @pytest.mark.asyncio
    async def test_parallel_all_success(self):
        """Test parallel execution when all providers succeed."""
        strategy = ParallelStrategy(timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=False)
        provider2 = MockProvider("provider2", ["model2"], should_fail=False)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        response = await strategy.execute(providers_models, "Test message")
        
        assert response['success'] is True
        assert len(response['successful_responses']) == 2
        assert response['total_providers'] == 2
        assert response['successful_providers'] == 2
        assert provider1.call_count == 1
        assert provider2.call_count == 1
    
    @pytest.mark.asyncio
    async def test_parallel_partial_success(self):
        """Test parallel execution with partial success."""
        strategy = ParallelStrategy(timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=True)
        provider2 = MockProvider("provider2", ["model2"], should_fail=False)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        response = await strategy.execute(providers_models, "Test message")
        
        assert response['success'] is True  # Success if at least one succeeds
        assert len(response['successful_responses']) == 1
        assert response['total_providers'] == 2
        assert response['successful_providers'] == 1
    
    @pytest.mark.asyncio
    async def test_parallel_all_fail(self):
        """Test parallel execution when all providers fail."""
        strategy = ParallelStrategy(timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=True)
        provider2 = MockProvider("provider2", ["model2"], should_fail=True)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        response = await strategy.execute(providers_models, "Test message")
        
        assert response['success'] is False
        assert len(response['successful_responses']) == 0
        assert response['successful_providers'] == 0


class TestRoundRobinStrategy:
    """Test cases for RoundRobinStrategy."""
    
    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        """Test round-robin provider selection."""
        strategy = RoundRobinStrategy(timeout=30)
        
        provider1 = MockProvider("provider1", ["model1"], should_fail=False)
        provider2 = MockProvider("provider2", ["model2"], should_fail=False)
        
        providers_models = [(provider1, "model1"), (provider2, "model2")]
        
        # First call should use first provider
        response1 = await strategy.execute(providers_models, "Test message 1")
        assert response1['provider'] == "provider1"
        
        # Second call should use second provider
        response2 = await strategy.execute(providers_models, "Test message 2")
        assert response2['provider'] == "provider2"
        
        # Third call should cycle back to first provider
        response3 = await strategy.execute(providers_models, "Test message 3")
        assert response3['provider'] == "provider1"
    
    @pytest.mark.asyncio
    async def test_round_robin_no_providers(self):
        """Test round-robin with no providers."""
        strategy = RoundRobinStrategy(timeout=30)
        
        response = await strategy.execute([], "Test message")
        
        assert response['success'] is False
        assert "No providers configured" in response['error']


class TestCreateStrategy:
    """Test strategy factory function."""
    
    def test_create_failover_strategy(self):
        """Test creating failover strategy."""
        strategy = create_strategy("failover", max_retries=5, timeout=60)
        
        assert isinstance(strategy, FailoverStrategy)
        assert strategy.max_retries == 5
        assert strategy.timeout == 60
    
    def test_create_parallel_strategy(self):
        """Test creating parallel strategy."""
        strategy = create_strategy("parallel", timeout=45)
        
        assert isinstance(strategy, ParallelStrategy)
        assert strategy.timeout == 45
    
    def test_create_round_robin_strategy(self):
        """Test creating round-robin strategy."""
        strategy = create_strategy("round_robin")
        
        assert isinstance(strategy, RoundRobinStrategy)
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("unknown_strategy")


if __name__ == '__main__':
    pytest.main([__file__])