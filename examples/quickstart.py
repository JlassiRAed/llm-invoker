"""
Quick Start Example

This is the simplest way to get started with llmInvoker.
"""

from llmInvoker import llmInvoker, invoke_failover


def main():
    print("MultiAgent Failover Invoke - Quick Start")
    print("=" * 40)
    
    # Method 1: Using the convenience function (simplest)
    print("\n1. Using convenience function:")
    response = invoke_failover(
        message="What are the key benefits of renewable energy?",
        providers={
            "github": ["gpt-4o"],
            "google": ["gemini-2.0-flash-exp"]
        }
    )
    
    if response['success']:
        print(f"✓ Success! Response from {response['provider']}:")
        print(response['response'])
    else:
        print(f"✗ Failed: {response['error']}")
    
    # Method 2: Using the main class
    print("\n2. Using main class:")
    invoker = llmInvoker(strategy="failover")
    
    # Use default configurations (all available providers)
    try:
        invoker.use_defaults()
        response = invoker.invoke_sync(
            message="Explain machine learning in one paragraph."
        )
        
        if response['success']:
            print(f"✓ Success! Response from {response['provider']}:")
            print(response['response'])
        else:
            print(f"✗ Failed: {response['error']}")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure you have API keys configured in your .env file")
    
    # Method 3: Manual configuration
    print("\n3. Manual configuration:")
    invoker2 = llmInvoker(strategy="failover")
    
    try:
        invoker2.configure_providers(
            github=["gpt-4o", "gpt-4o-mini"],
            google=["gemini-2.0-flash-exp"]
        )
        
        response = invoker2.invoke_sync(
            message="What is the future of AI?"
        )
        
        if response['success']:
            print(f"✓ Success! Response from {response['provider']}:")
            print(response['response'])
        else:
            print(f"✗ Failed: {response['error']}")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    
    # Method 4: Multimodal example
    print("\n4. Multimodal content (if supported):")
    invoker3 = llmInvoker(strategy="failover")
    
    try:
        invoker3.configure_providers(
            github=["gpt-4o"],  # Supports vision
            google=["gemini-2.0-flash-exp"]  # Supports vision
        )
        
        # Example with multimodal content
        multimodal_message = {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Describe what you would see in a typical nature landscape."
                }
            ]
        }
        
        response = invoker3.invoke_sync(
            message=multimodal_message
        )
        
        if response['success']:
            print(f"✓ Success! Multimodal response from {response['provider']}:")
            print(response['response'])
        else:
            print(f"✗ Failed: {response['error']}")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    
    print("\n" + "=" * 40)
    print("Quick start completed!")
    print("\nNext steps:")
    print("- Check examples/failover_example.py for more detailed examples")
    print("- Check examples/parallel_invoke_example.py for parallel execution")
    print("- Check examples/langchain_integration.py for framework integration")


if __name__ == "__main__":
    main()
