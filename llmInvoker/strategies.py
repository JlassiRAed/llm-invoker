"""Invocation strategies for handling multiple providers and models."""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .providers import BaseProvider
from .history import ConversationHistory
from .utils import log_strategy_execution, is_rate_limit_error

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for all invocation strategies."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout
        self.history = ConversationHistory()
    
    @abstractmethod
    async def execute(
        self, 
        providers_models: List[Tuple[BaseProvider, str]], 
        messages: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the strategy with given providers and models."""
        pass


class FailoverStrategy(BaseStrategy):
    """Failover strategy that tries providers/models in order until one succeeds."""
    
    async def execute(
        self, 
        providers_models: List[Tuple[BaseProvider, str]], 
        messages: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute failover strategy."""
        log_strategy_execution("failover", len(providers_models))
        
        last_error = None
        for attempt in range(self.max_retries):
            for provider, model in providers_models:
                try:
                    # Add conversation history context
                    enhanced_messages = self.history.get_enhanced_messages(messages)
                    
                    logger.info(f"Trying provider {provider.name}:{model} (attempt {attempt + 1})")
                    
                    result = await asyncio.wait_for(
                        provider.invoke(model, enhanced_messages, **kwargs),
                        timeout=self.timeout
                    )
                    
                    if result['success']:
                        # Store successful interaction in history
                        self.history.add_interaction(
                            messages, result['response'], provider.name, model
                        )
                        
                        logger.info(f"Success with provider {provider.name}:{model}")
                        return {
                            'strategy': 'failover',
                            'provider': provider.name,
                            'model': model,
                            'response': result['response'],
                            'attempt': attempt + 1,
                            'success': True
                        }
                    else:
                        last_error = result.get('error', 'Unknown error')
                        logger.warning(f"Provider {provider.name}:{model} failed: {last_error}")
                        
                except asyncio.TimeoutError:
                    last_error = f"Timeout after {self.timeout}s"
                    logger.warning(f"Provider {provider.name}:{model} timed out")
                except Exception as e:
                    last_error = str(e)
                    
                    # Check if it's a rate limit error
                    if is_rate_limit_error(last_error):
                        logger.info(f"Rate limit detected for {provider.name}:{model}, switching to next provider")
                    else:
                        logger.warning(f"Provider {provider.name}:{model} failed with exception: {last_error}")
                
                # Wait before trying next provider (shorter wait for rate limits)
                if len(providers_models) > 1:
                    if is_rate_limit_error(str(last_error)):
                        await asyncio.sleep(0.1)  # Quick switch for rate limits
                    else:
                        await asyncio.sleep(1)
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return {
            'strategy': 'failover',
            'success': False,
            'error': f"All providers failed after {self.max_retries} attempts. Last error: {last_error}",
            'attempts': self.max_retries
        }


class ParallelStrategy(BaseStrategy):
    """Parallel strategy that invokes all providers/models simultaneously."""
    
    async def execute(
        self, 
        providers_models: List[Tuple[BaseProvider, str]], 
        messages: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute parallel strategy."""
        log_strategy_execution("parallel", len(providers_models))
        
        # Add conversation history context
        enhanced_messages = self.history.get_enhanced_messages(messages)
        
        # Create tasks for all providers
        tasks = []
        for provider, model in providers_models:
            task = asyncio.create_task(
                self._invoke_with_timeout(provider, model, enhanced_messages, **kwargs)
            )
            tasks.append((task, provider.name, model))
        
        # Wait for all tasks to complete
        results = {}
        successful_responses = []
        
        for task, provider_name, model in tasks:
            try:
                result = await task
                results[f"{provider_name}:{model}"] = result
                
                if result['success']:
                    successful_responses.append({
                        'provider': provider_name,
                        'model': model,
                        'response': result['response']
                    })
                    
                    # Store successful interaction in history
                    self.history.add_interaction(
                        messages, result['response'], provider_name, model
                    )
                    
            except Exception as e:
                results[f"{provider_name}:{model}"] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'strategy': 'parallel',
            'results': results,
            'successful_responses': successful_responses,
            'success': len(successful_responses) > 0,
            'total_providers': len(providers_models),
            'successful_providers': len(successful_responses)
        }
    
    async def _invoke_with_timeout(
        self, 
        provider: BaseProvider, 
        model: str, 
        messages: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """Invoke a provider with timeout."""
        try:
            return await asyncio.wait_for(
                provider.invoke(model, messages, **kwargs),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Timeout after {self.timeout}s",
                'provider': provider.name,
                'model': model
            }


class RoundRobinStrategy(BaseStrategy):
    """Round-robin strategy that cycles through providers/models."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        super().__init__(max_retries, timeout)
        self.current_index = 0
    
    async def execute(
        self, 
        providers_models: List[Tuple[BaseProvider, str]], 
        messages: Union[str, List[Dict], Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute round-robin strategy."""
        log_strategy_execution("round_robin", len(providers_models))
        
        if not providers_models:
            return {
                'strategy': 'round_robin',
                'success': False,
                'error': 'No providers configured'
            }
        
        # Select provider based on round-robin
        provider, model = providers_models[self.current_index % len(providers_models)]
        self.current_index += 1
        
        # Add conversation history context
        enhanced_messages = self.history.get_enhanced_messages(messages)
        
        try:
            result = await asyncio.wait_for(
                provider.invoke(model, enhanced_messages, **kwargs),
                timeout=self.timeout
            )
            
            if result['success']:
                # Store successful interaction in history
                self.history.add_interaction(
                    messages, result['response'], provider.name, model
                )
            
            return {
                'strategy': 'round_robin',
                'provider': provider.name,
                'model': model,
                'response': result.get('response'),
                'success': result['success'],
                'error': result.get('error')
            }
            
        except asyncio.TimeoutError:
            return {
                'strategy': 'round_robin',
                'provider': provider.name,
                'model': model,
                'success': False,
                'error': f"Timeout after {self.timeout}s"
            }
        except Exception as e:
            return {
                'strategy': 'round_robin',
                'provider': provider.name,
                'model': model,
                'success': False,
                'error': str(e)
            }


# Strategy registry
STRATEGIES = {
    "failover": FailoverStrategy,
    "parallel": ParallelStrategy,
    "round_robin": RoundRobinStrategy,
}


def create_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """Create a strategy instance."""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(**kwargs)