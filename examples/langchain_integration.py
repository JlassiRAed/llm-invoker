"""
LangChain Integration Example

This example demonstrates how to integrate llmInvoker with LangChain workflows
and use it as a drop-in replacement for LangChain LLMs.
"""

import asyncio
from typing import Optional, Dict, Any
from llmInvoker import llmInvoker


class LangChainWrapper:
    """
    A wrapper class that makes llmInvoker compatible with LangChain workflows.
    """
    
    def __init__(self, strategy: str = "failover", enable_history: bool = True):
        """Initialize the LangChain wrapper."""
        self.invoker = llmInvoker(strategy=strategy, enable_history=enable_history)
        # Use only working providers
        try:
            self.invoker.configure_providers(github=["gpt-4o-mini"])
        except ValueError:
            self.invoker.use_defaults()
    
    async def __call__(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Call the LLM with a prompt and return the response content.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (not implemented yet)
            max_tokens: Maximum tokens to generate (not implemented yet)
            **kwargs: Additional parameters (not implemented yet)
            
        Returns:
            str: The response content
            
        Raises:
            Exception: If all providers fail
        """
        response = await self.invoker.invoke(prompt)
        
        if response['success']:
            content = response['response']
            # Extract actual text content from response
            if isinstance(content, dict):
                if 'choices' in content and content['choices']:
                    content = content['choices'][0].get('message', {}).get('content', str(content))
                elif 'candidates' in content and content['candidates']:
                    parts = content['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        content = parts[0]['text']
                    else:
                        content = str(content)
                else:
                    content = str(content)
            
            return str(content)
        else:
            error_msg = response.get('error', response.get('message', 'Unknown error'))
            raise Exception(f"All providers failed: {error_msg}")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about provider usage."""
        return self.invoker.get_provider_stats()
    
    def get_history(self):
        """Get conversation history."""
        return self.invoker.get_history()


class AsyncLangChainChain:
    """
    A simple chain that mimics LangChain's chain behavior.
    """
    
    def __init__(self, llm: LangChainWrapper, template: str):
        """
        Initialize the chain.
        
        Args:
            llm: The LLM wrapper
            template: Template string with {variable} placeholders
        """
        self.llm = llm
        self.template = template
    
    async def run(self, **kwargs) -> str:
        """
        Run the chain with the given variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            str: The LLM response
        """
        prompt = self.template.format(**kwargs)
        return await self.llm(prompt)


async def example_1_basic_integration():
    """Basic LangChain-like integration."""
    print("=== Example 1: Basic LangChain Integration ===")
    
    # Create a LangChain-compatible wrapper
    llm_wrapper = LangChainWrapper(strategy="failover", enable_history=True)
    
    # Use like a LangChain LLM
    prompt = "Explain the concept of machine learning in simple terms."
    
    print(f"\n📤 Sending prompt: {prompt}")
    
    try:
        response = await llm_wrapper(prompt)
        print(f"\n✅ Response received:")
        print(f"{response[:200]}...")
        
        # Get provider stats
        stats = llm_wrapper.get_provider_stats()
        print(f"\n📊 Provider Stats:")
        print(f"  • Total requests: {stats.get('total_requests', 0)}")
        print(f"  • Success rate: {stats.get('success_rate', 0):.2%}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def example_2_chain_integration():
    """Demonstrate chain-like behavior."""
    print("\n=== Example 2: Chain Integration ===")
    
    # Create LLM wrapper
    llm = LangChainWrapper(strategy="failover", enable_history=True)
    
    # Create a simple chain
    template = """
You are a helpful assistant. Answer the question about {topic} in a {style} manner.

Question: {question}

Answer:"""
    
    chain = AsyncLangChainChain(llm, template)
    
    # Run the chain
    print(f"\n🔗 Running chain with template...")
    
    try:
        result = await chain.run(
            topic="artificial intelligence",
            style="beginner-friendly",
            question="What is neural network?"
        )
        
        print(f"\n✅ Chain result:")
        print(f"{result[:300]}...")
        
    except Exception as e:
        print(f"\n❌ Chain error: {e}")


async def example_3_conversation_chain():
    """Demonstrate conversation with history."""
    print("\n=== Example 3: Conversation Chain ===")
    
    # Create LLM with history enabled
    llm = LangChainWrapper(strategy="failover", enable_history=True)
    
    conversation = [
        "What is Python?",
        "How is it different from Java?",
        "Which one should I learn first?"
    ]
    
    print(f"\n💬 Having a conversation...")
    
    for i, question in enumerate(conversation, 1):
        print(f"\nUser ({i}): {question}")
        
        try:
            response = await llm(question)
            print(f"Assistant: {response[:150]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Show history
    history = llm.get_history()
    print(f"\n📊 Conversation history: {len(history.entries)} entries")


async def example_4_parallel_comparison():
    """Compare responses using parallel strategy."""
    print("\n=== Example 4: Parallel Response Comparison ===")
    
    # Create parallel LLM wrapper
    parallel_llm = LangChainWrapper(strategy="parallel", enable_history=True)
    
    question = "What are the benefits of renewable energy?"
    print(f"\n⚡ Getting parallel responses for: {question}")
    
    try:
        # This will get responses from all available providers
        response = await parallel_llm.invoker.invoke(question)
        
        if response['success']:
            all_responses = response.get('all_responses', [])
            successful_responses = [r for r in all_responses if r.get('response') and str(r.get('response')).strip()]
            
            print(f"\n✅ Got {len(successful_responses)} successful responses:")
            
            for i, resp in enumerate(successful_responses, 1):
                provider = resp.get('provider', 'Unknown')
                model = resp.get('model', 'Unknown')
                content = str(resp.get('response', ''))
                
                # Extract text content
                if isinstance(resp.get('response'), dict):
                    raw_resp = resp.get('response')
                    if 'choices' in raw_resp and raw_resp['choices']:
                        content = raw_resp['choices'][0].get('message', {}).get('content', str(raw_resp))
                    elif 'candidates' in raw_resp and raw_resp['candidates']:
                        parts = raw_resp['candidates'][0].get('content', {}).get('parts', [])
                        if parts and 'text' in parts[0]:
                            content = parts[0]['text']
                
                print(f"\n🤖 Response {i} ({provider}/{model}):")
                print(f"{content[:200]}...")
        
        else:
            print(f"❌ All providers failed: {response.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Parallel comparison error: {e}")


async def example_5_custom_prompt_template():
    """Use custom prompt templates."""
    print("\n=== Example 5: Custom Prompt Templates ===")
    
    llm = LangChainWrapper(strategy="failover", enable_history=True)
    
    # Define custom templates
    templates = {
        "summarize": "Summarize the following text in {max_words} words:\n\n{text}",
        "translate": "Translate the following {source_lang} text to {target_lang}:\n\n{text}",
        "explain": "Explain {concept} to a {audience} in {tone} tone."
    }
    
    # Example 1: Summarization
    print(f"\n📝 Template: Summarization")
    try:
        summary_prompt = templates["summarize"].format(
            max_words=50,
            text="Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
        )
        
        summary = await llm(summary_prompt)
        print(f"Summary: {summary[:150]}...")
        
    except Exception as e:
        print(f"❌ Summarization error: {e}")
    
    # Example 2: Explanation
    print(f"\n🎓 Template: Explanation")
    try:
        explain_prompt = templates["explain"].format(
            concept="blockchain technology",
            audience="5-year-old",
            tone="simple and fun"
        )
        
        explanation = await llm(explain_prompt)
        print(f"Explanation: {explanation[:150]}...")
        
    except Exception as e:
        print(f"❌ Explanation error: {e}")


async def example_6_error_handling():
    """Demonstrate robust error handling."""
    print("\n=== Example 6: Error Handling ===")
    
    llm = LangChainWrapper(strategy="failover", enable_history=True)
    
    # Test with various prompts that might cause issues
    test_prompts = [
        "What is AI?",  # Simple prompt
        "Write a very long essay about the history of the universe with exactly 10000 words.",  # Potentially problematic
        "Hello! How are you?",  # Simple again
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}: {prompt[:50]}...")
        
        try:
            response = await llm(prompt)
            print(f"✅ Success: {response[:100]}...")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            
            # Try to get more info about the failure
            stats = llm.get_provider_stats()
            print(f"   Provider stats: {stats.get('total_requests', 0)} requests, {stats.get('success_rate', 0):.1%} success rate")


async def run_examples():
    """Run all LangChain integration examples."""
    try:
        await example_1_basic_integration()
        await example_2_chain_integration()
        await example_3_conversation_chain()
        await example_4_parallel_comparison()
        await example_5_custom_prompt_template()
        await example_6_error_handling()
        
        print("\n✅ All LangChain integration examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run examples."""
    print("MultiAgent Failover Invoke - LangChain Integration Examples")
    print("=" * 65)
    
    # Simple async execution without complex event loop management
    asyncio.run(run_examples())


if __name__ == "__main__":
    print(__doc__)
    main()