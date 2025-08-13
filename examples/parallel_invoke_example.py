"""
Parallel Strategy Example

This example demonstrates the parallel invocation strategy, which sends
the same prompt to multiple providers simultaneously and returns all responses.
"""

import asyncio
from llmInvoker import llmInvoker


async def example_1_basic_parallel():
    """Basic parallel invocation with default providers."""
    print("=== Example 1: Basic Parallel Invocation ===")
    
    # Initialize with parallel strategy and use default providers
    invoker = llmInvoker(strategy="parallel", enable_history=True)
    invoker.use_defaults()  # Use only configured providers
    
    prompt = "Explain the concept of machine learning in exactly 2 sentences."
    
    print(f"\n📤 Sending prompt to all available providers in parallel...")
    print(f"Prompt: {prompt}")
    
    response = await invoker.invoke(prompt)
    
    if response['success']:
        print(f"\n✅ Parallel invocation successful!")
        all_responses = response.get('all_responses', [])
        successful_responses = [r for r in all_responses if r.get('response') and r.get('response').strip()]
        
        print(f"Total providers attempted: {len(all_responses)}")
        print(f"Successful responses: {len(successful_responses)}")
        
        # Display successful responses
        for i, provider_response in enumerate(successful_responses, 1):
            provider = provider_response.get('provider', 'Unknown')
            model = provider_response.get('model', 'Unknown')
            content = provider_response.get('response', 'No response')
            
            print(f"\n--- Response {i} ({provider}/{model}) ---")
            print(content)
            print("-" * 50)
        
        # Show failed responses for debugging
        failed_responses = [r for r in all_responses if not (r.get('response') and r.get('response').strip())]
        if failed_responses:
            print(f"\n⚠️  Failed responses: {len(failed_responses)}")
            for resp in failed_responses:
                provider = resp.get('provider', 'Unknown')
                model = resp.get('model', 'Unknown')
                error = resp.get('error', 'No error message')
                print(f"  • {provider}/{model}: {error}")
    else:
        error_msg = response.get('error', response.get('message', 'Unknown error'))
        print(f"\n❌ All providers failed: {error_msg}")


async def example_2_parallel_comparison():
    """Compare responses from different models for the same prompt."""
    print("\n=== Example 2: Model Response Comparison ===")
    
    # Initialize parallel invoker with fewer providers to reduce rate limiting
    invoker = llmInvoker(strategy="parallel", enable_history=True)
    
    # Configure with fewer providers to reduce rate limiting
    try:
        invoker.configure_providers(
            github=["gpt-4o-mini"],  # Only use gpt-4o-mini as it has fewer rate limits
            openrouter=["deepseek/deepseek-r1"]  # Only one OpenRouter model
        )
    except ValueError as e:
        print(f"⚠️  Configuration warning: {e}")
        print("🔄 Falling back to default providers...")
        invoker.use_defaults()
    
    prompt = "Write a simple Python function to add two numbers."
    
    print(f"\n📤 Comparing model responses...")
    print(f"Prompt: {prompt}")
    
    response = await invoker.invoke(prompt)
    
    if response['success']:
        all_responses = response.get('all_responses', [])
        successful_responses = [r for r in all_responses if r.get('response') and r.get('response').strip()]
        
        print(f"\n✅ Got {len(successful_responses)} successful responses for comparison:")
        
        for i, provider_response in enumerate(successful_responses, 1):
            provider = provider_response.get('provider', 'Unknown')
            model = provider_response.get('model', 'Unknown')
            content = provider_response.get('response', 'No response')
            
            print(f"\n🤖 Model {i}: {provider}/{model}")
            print("Response:")
            # Show first 200 chars for comparison
            preview = content[:200] + "..." if len(content) > 200 else content
            print(preview)
            print("=" * 60)
    else:
        error_msg = response.get('error', response.get('message', 'Unknown error'))
        print(f"\n❌ Parallel comparison failed: {error_msg}")


async def example_3_parallel_with_analysis():
    """Parallel invocation with response analysis."""
    print("\n=== Example 3: Response Analysis ===")
    
    # Use fewer providers to avoid rate limiting
    invoker = llmInvoker(strategy="parallel", enable_history=True)
    try:
        invoker.configure_providers(
            github=["gpt-4o-mini"],
            openrouter=["deepseek/deepseek-r1"]
        )
    except ValueError:
        invoker.use_defaults()
    
    prompt = "What are the top 3 benefits of renewable energy?"
    
    print(f"\n📤 Analyzing parallel responses...")
    print(f"Prompt: {prompt}")
    
    response = await invoker.invoke(prompt)
    
    if response['success']:
        all_responses = response.get('all_responses', [])
        successful_responses = [r for r in all_responses if r.get('response') and r.get('response').strip()]
        
        # Analyze responses
        print(f"\n📊 Response Analysis:")
        print(f"  • Total attempts: {len(all_responses)}")
        print(f"  • Successful responses: {len(successful_responses)}")
        
        if successful_responses:
            response_lengths = []
            providers_used = set()
            
            for provider_response in successful_responses:
                provider = provider_response.get('provider', 'Unknown')
                model = provider_response.get('model', 'Unknown')
                content = provider_response.get('response', '')
                
                providers_used.add(f"{provider}/{model}")
                response_lengths.append(len(content))
                
                print(f"  • {provider}/{model}: {len(content)} characters")
            
            if response_lengths:
                avg_length = sum(response_lengths) / len(response_lengths)
                print(f"  • Average response length: {avg_length:.0f} characters")
                print(f"  • Shortest response: {min(response_lengths)} characters")
                print(f"  • Longest response: {max(response_lengths)} characters")
            
            print(f"  • Unique providers: {len(providers_used)}")
            
            # Show shortest and longest responses
            if len(successful_responses) > 1:
                shortest = min(successful_responses, key=lambda x: len(x.get('response', '')))
                longest = max(successful_responses, key=lambda x: len(x.get('response', '')))
                
                print(f"\n📝 Shortest response ({shortest.get('provider')}/{shortest.get('model')}):")
                print(shortest.get('response', 'No content')[:150] + "...")
                
                print(f"\n📝 Longest response ({longest.get('provider')}/{longest.get('model')}):")
                print(longest.get('response', 'No content')[:150] + "...")
        else:
            print("  • No successful responses to analyze")
    
    else:
        error_msg = response.get('error', response.get('message', 'Unknown error'))
        print(f"\n❌ Analysis failed: {error_msg}")


async def example_4_parallel_performance():
    """Test parallel invocation performance."""
    print("\n=== Example 4: Performance Testing ===")
    
    import time
    
    # Test parallel strategy with limited providers
    print("\n⚡ Testing parallel strategy performance...")
    invoker_parallel = llmInvoker(strategy="parallel", enable_history=True)
    try:
        invoker_parallel.configure_providers(
            github=["gpt-4o-mini"],
            openrouter=["deepseek/deepseek-r1"]
        )
    except ValueError:
        invoker_parallel.use_defaults()
    
    start_time = time.time()
    parallel_response = await invoker_parallel.invoke("What is Python?")
    parallel_time = time.time() - start_time
    
    # Test failover strategy for comparison
    print("\n🔄 Testing failover strategy performance...")
    invoker_failover = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker_failover.configure_providers(
            github=["gpt-4o-mini"],
            openrouter=["deepseek/deepseek-r1"]
        )
    except ValueError:
        invoker_failover.use_defaults()
    
    start_time = time.time()
    failover_response = await invoker_failover.invoke("What is Python?")
    failover_time = time.time() - start_time
    
    # Performance comparison
    print(f"\n⏱️  Performance Comparison:")
    print(f"  • Parallel strategy: {parallel_time:.2f} seconds")
    if parallel_response['success']:
        all_responses = parallel_response.get('all_responses', [])
        successful_count = len([r for r in all_responses if r.get('response') and r.get('response').strip()])
        print(f"    - Successful responses: {successful_count}/{len(all_responses)}")
    
    print(f"  • Failover strategy: {failover_time:.2f} seconds")
    if failover_response['success']:
        print(f"    - Single response received")
    
    if parallel_response['success'] and failover_response['success']:
        efficiency = failover_time / parallel_time if parallel_time > 0 else 0
        print(f"  • Parallel efficiency: {efficiency:.2f}x")


async def example_5_error_handling():
    """Demonstrate error handling in parallel strategy."""
    print("\n=== Example 5: Error Handling ===")
    
    # Use limited providers to test error handling
    invoker = llmInvoker(strategy="parallel", enable_history=True)
    try:
        invoker.configure_providers(
            github=["gpt-4o-mini"],
            openrouter=["deepseek/deepseek-r1"]
        )
    except ValueError:
        invoker.use_defaults()
    
    # Test with a reasonable prompt that might hit rate limits
    test_prompt = "Write a short essay about artificial intelligence in 100 words."
    
    print(f"\n🚨 Testing error handling...")
    print(f"Prompt: {test_prompt}")
    
    response = await invoker.invoke(test_prompt)
    
    if response['success']:
        all_responses = response.get('all_responses', [])
        successful_responses = [r for r in all_responses if r.get('response') and r.get('response').strip()]
        failed_responses = [r for r in all_responses if not (r.get('response') and r.get('response').strip())]
        
        print(f"\n📊 Results:")
        print(f"  • Total attempts: {len(all_responses)}")
        print(f"  • Successful: {len(successful_responses)}")
        print(f"  • Failed: {len(failed_responses)}")
        
        if successful_responses:
            print(f"\n✅ Successful responses:")
            for resp in successful_responses:
                provider = resp.get('provider', 'Unknown')
                model = resp.get('model', 'Unknown')
                content = resp.get('response', '')
                print(f"  • {provider}/{model}: {len(content)} characters")
        
        if failed_responses:
            print(f"\n❌ Failed responses:")
            for resp in failed_responses:
                provider = resp.get('provider', 'Unknown')
                model = resp.get('model', 'Unknown')
                error = resp.get('error', 'Unknown error')
                print(f"  • {provider}/{model}: {error}")
    
    else:
        error_msg = response.get('error', response.get('message', 'Unknown error'))
        print(f"\n❌ All providers failed: {error_msg}")


async def run_examples():
    """Run all parallel strategy examples."""
    await example_1_basic_parallel()
    await example_2_parallel_comparison()
    await example_3_parallel_with_analysis()
    await example_4_parallel_performance()
    await example_5_error_handling()


def main():
    """Main function to run examples."""
    print("MultiAgent Failover Invoke - Parallel Strategy Examples")
    print("=" * 60)
    
    # Simple async execution
    asyncio.run(run_examples())
    
    print("\n✅ All parallel strategy examples completed!")


if __name__ == "__main__":
    print(__doc__)
    main()