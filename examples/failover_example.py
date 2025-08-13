"""
Failover Strategy Example

This example demonstrates how to use the failover strategy to automatically
switch between providers/models when failures occur.
"""

import asyncio
from llmInvoker import llmInvoker, invoke_failover


async def example_1_basic_failover():
    """Basic failover example using the main class."""
    print("=== Example 1: Basic Failover ===")
    
    # Initialize the invoker
    invoker = llmInvoker(strategy="failover", max_retries=3, timeout=30)
    
    # Configure providers and models
    invoker.configure_providers(
        github=["gpt-4o", "gpt-4o-mini"],
        google=["gemini-2.0-flash-exp"],
        openrouter=["deepseek/deepseek-r1", "meta-llama/llama-3.2-3b-instruct:free"]
    )
    
    # Make a request
    message = "Explain the concept of machine learning in simple terms."
    
    try:
        response = await invoker.invoke(message)
        
        if response['success']:
            print(f"Success! Provider: {response['provider']}, Model: {response['model']}")
            print(f"Response: {response['response']}")
        else:
            print(f"Failed: {response['error']}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_2_convenience_function():
    """Using the convenience function for quick failover."""
    print("\n=== Example 2: Convenience Function ===")
    
    # Quick failover using convenience function
    response = invoke_failover(
        message="What are the benefits of renewable energy?",
        providers={
            "github": ["gpt-4o"],
            "google": ["gemini-2.0-flash-exp"],
            "openrouter": ["deepseek/deepseek-r1"]
        },
        temperature=0.7,
        max_tokens=150
    )
    
    if response['success']:
        print(f"Success! Provider: {response['provider']}")
        print(f"Response: {response['response']}")
    else:
        print(f"Failed: {response['error']}")


async def example_3_with_history():
    """Failover with conversation history."""
    print("\n=== Example 3: Failover with History ===")
    
    invoker = llmInvoker(strategy="failover", enable_history=True)
    
    # Use default configurations for available providers
    invoker.use_defaults()
    
    # First message
    response1 = await invoker.invoke("My name is Alice. What's a good hobby for beginners?")
    if response1['success']:
        print(f"First response from {response1['provider']}: {response1['response']}")
    
    # Second message (will include history context)
    response2 = await invoker.invoke("Can you recommend specific resources for that hobby?")
    if response2['success']:
        print(f"Second response from {response2['provider']}: {response2['response']}")
    
    # Show history summary
    history = invoker.get_history()
    if history:
        summary = history.get_summary()
        print(f"Conversation summary: {summary}")


async def example_4_error_handling():
    """Example with proper error handling and retries."""
    print("\n=== Example 4: Error Handling ===")
    
    invoker = llmInvoker(
        strategy="failover",
        max_retries=2,
        timeout=15,  # Shorter timeout for demonstration
        enable_langsmith=True
    )
    
    # Configure with some potentially problematic setups
    invoker.configure_providers(
        github=["gpt-4o"],  # This might work
        google=["gemini-2.0-flash-exp"],  # This might work
        openrouter=["some-nonexistent-model"]  # This will likely fail
    )
    
    message = "Explain quantum computing basics."
    
    response = await invoker.invoke(message)
    
    if response['success']:
        print(f"Success after potential failures!")
        print(f"Provider: {response['provider']}")
        print(f"Attempts: {response.get('attempt', 1)}")
    else:
        print(f"All providers failed: {response['error']}")
    
    # Show provider statistics
    stats = invoker.get_provider_stats()
    print(f"Provider stats: {stats}")


async def example_5_string_configuration():
    """Using string-based provider configuration."""
    print("\n=== Example 5: String Configuration ===")
    
    invoker = llmInvoker(strategy="failover")
    
    # Configure using string format (similar to your original request)
    config_string = "github['gpt-4o','gpt-4o-mini'],google['gemini-2.0-flash-exp'],openrouter['deepseek/deepseek-r1']"
    invoker.configure_from_string(config_string)
    
    response = invoke_failover(
        message="What are the main differences between Python and JavaScript?",
        providers={
            "github": ["gpt-4o"],
            "google": ["gemini-2.0-flash-exp"],
            "openrouter": ["deepseek/deepseek-r1"]
        },
        temperature=0.5
    )
    
    if response['success']:
        print(f"Response from {response['provider']}:{response['model']}")
        print(f"Content: {response['response']}")
    else:
        print(f"Failed: {response['error']}")


def main():
    """Run all examples."""
    print("MultiAgent Failover Invoke - Failover Strategy Examples")
    print("=" * 60)
    
    # Check if we're in an async context
    try:
        loop = asyncio.get_running_loop()
        print("Running in async context...")
    except RuntimeError:
        # Create new event loop
        asyncio.run(run_examples())


async def run_examples():
    """Run all async examples."""
    await example_1_basic_failover()
    example_2_convenience_function()
    await example_3_with_history()
    await example_4_error_handling()
    await example_5_string_configuration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    main()