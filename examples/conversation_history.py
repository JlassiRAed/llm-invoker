"""
Conversation History Example

This example demonstrates all features related to conversation history management
including import/export, analysis, and advanced usage patterns.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from llmInvoker import llmInvoker


async def example_1_basic_history():
    """Basic conversation history usage."""
    print("=== Example 1: Basic History Management ===")
    
    # Initialize with history enabled (default)
    invoker = llmInvoker(strategy="failover", enable_history=True)
    invoker.use_defaults()
    
    # Have a conversation
    print("\n📝 Starting conversation...")
    response1 = await invoker.invoke("Hello, my name is Alice. What's your name?")
    if response1['success']:
        content = response1['response']
        # Extract actual text content from response
        if isinstance(content, dict) and 'choices' in content:
            content = content['choices'][0]['message']['content']
        elif isinstance(content, dict) and 'candidates' in content:
            content = content['candidates'][0]['content']['parts'][0]['text']
        elif isinstance(content, dict):
            content = str(content)
        print(f"Assistant: {content}")
    
    response2 = await invoker.invoke("What did I just tell you my name was?")
    if response2['success']:
        content = response2['response']
        # Extract actual text content from response
        if isinstance(content, dict) and 'choices' in content:
            content = content['choices'][0]['message']['content']
        elif isinstance(content, dict) and 'candidates' in content:
            content = content['candidates'][0]['content']['parts'][0]['text']
        elif isinstance(content, dict):
            content = str(content)
        print(f"Assistant: {content}")
    
    # Check history
    history = invoker.get_history()
    print(f"\n📊 Conversation entries: {len(history.entries)}")
    
    return invoker


async def example_2_history_analysis():
    """Analyze conversation history and get statistics."""
    print("\n=== Example 2: History Analysis ===")
    
    invoker = llmInvoker(strategy="failover", enable_history=True)
    # Use fewer providers to reduce rate limiting
    try:
        invoker.configure_providers(
            github=["gpt-4o-mini"]
        )
    except ValueError:
        invoker.use_defaults()
    
    # Have multiple interactions
    questions = [
        "What is machine learning?",
        "How does it differ from traditional programming?", 
        "Can you give me a simple example?"
    ]
    
    print("\n💬 Having extended conversation...")
    for i, question in enumerate(questions, 1):
        print(f"User ({i}): {question}")
        response = await invoker.invoke(question)
        
        if response['success']:
            content = response['response']
            # Handle different response types
            if isinstance(content, dict):
                # Try to extract content from dict response
                if 'choices' in content and len(content['choices']) > 0:
                    content = content['choices'][0].get('message', {}).get('content', str(content))
                elif 'candidates' in content and len(content['candidates']) > 0:
                    parts = content['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        content = parts[0]['text']
                    else:
                        content = str(content)
                else:
                    content = str(content)
            
            # Ensure content is a string and truncate for display
            content_str = str(content)
            print(f"Assistant: {content_str[:100]}...")
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
    
    # Analyze history
    history = invoker.get_history()
    summary = history.get_summary()
    
    print(f"\n📊 Conversation Analysis:")
    print(f"  • Total entries: {summary.get('total_entries', 0)}")
    print(f"  • Providers used: {summary.get('providers_used', [])}")
    print(f"  • Models used: {summary.get('models_used', [])}")
    print(f"  • Total messages: {summary.get('total_messages', 0)}")
    
    # Provider usage statistics
    provider_stats = history.get_provider_usage_stats()
    print(f"\n📈 Provider Statistics:")
    for provider, stats in provider_stats.items():
        print(f"  • {provider}: {stats} requests")
    
    return invoker


async def example_3_export_import():
    """Export and import conversation history."""
    print("\n=== Example 3: Export/Import History ===")
    
    # Create a conversation first
    invoker1 = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker1.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker1.use_defaults()
    
    print("\n💾 Creating conversation to export...")
    response = await invoker1.invoke("Tell me about renewable energy in one sentence.")
    if response['success']:
        print("✅ Conversation created")
    
    # Export history
    export_file = "test_conversation_history.json"
    history = invoker1.get_history()
    history.export_to_json(export_file)
    print(f"📤 History exported to {export_file}")
    
    # Create new invoker and import history
    invoker2 = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker2.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker2.use_defaults()
    
    # Import the history
    new_history = invoker2.get_history()
    new_history.import_from_json(export_file)
    print(f"📥 History imported to new invoker")
    
    # Continue conversation with imported context
    print(f"\n🔄 Continuing conversation with imported context...")
    response = await invoker2.invoke("What are the main types mentioned?")
    if response['success']:
        content = str(response['response'])
        if isinstance(response['response'], dict):
            # Extract actual content
            raw_response = response['response']
            if 'choices' in raw_response and len(raw_response['choices']) > 0:
                content = raw_response['choices'][0].get('message', {}).get('content', str(raw_response))
            elif 'candidates' in raw_response and len(raw_response['candidates']) > 0:
                parts = raw_response['candidates'][0].get('content', {}).get('parts', [])
                if parts and 'text' in parts[0]:
                    content = parts[0]['text']
        
        print(f"✅ Continued conversation: {content[:150]}...")
    
    # Verify history continuity
    final_history = invoker2.get_history()
    print(f"📊 Final history length: {len(final_history.entries)} entries")
    
    # Cleanup
    try:
        Path(export_file).unlink()
        print(f"🧹 Cleaned up {export_file}")
    except FileNotFoundError:
        pass


async def example_4_context_management():
    """Demonstrate context management in conversations."""
    print("\n=== Example 4: Context Management ===")
    
    invoker = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker.use_defaults()
    
    # Have a conversation that builds context
    print("\n🧠 Building conversation context...")
    
    context_questions = [
        "I'm planning a trip to Japan.",
        "What are the best months to visit?",
        "What should I pack for spring?"
    ]
    
    for question in context_questions:
        print(f"\nUser: {question}")
        response = await invoker.invoke(question)
        
        if response['success']:
            content = str(response['response'])
            # Handle dict responses
            if isinstance(response['response'], dict):
                raw_response = response['response']
                if 'choices' in raw_response and len(raw_response['choices']) > 0:
                    content = raw_response['choices'][0].get('message', {}).get('content', str(raw_response))
                elif 'candidates' in raw_response and len(raw_response['candidates']) > 0:
                    parts = raw_response['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        content = parts[0]['text']
            
            print(f"Assistant: {content[:200]}...")
    
    # Get context messages (without limit parameter)
    history = invoker.get_history()
    try:
        context_messages = history.get_context_messages()
        print(f"\n📋 Context messages ({len(context_messages)} total):")
        
        # Show last few messages for brevity
        recent_messages = context_messages[-4:] if len(context_messages) > 4 else context_messages
        for i, msg in enumerate(recent_messages, 1):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))[:100]
            print(f"  {i}. {role}: {content}...")
    except Exception as e:
        print(f"⚠️  Could not retrieve context messages: {e}")
        
        # Alternative: show entries directly
        print(f"\n📋 Recent conversation entries ({len(history.entries)} total):")
        for i, entry in enumerate(history.entries[-2:], 1):  # Show last 2
            print(f"  {i}. User: {entry.user_message[:50]}...")
            assistant_msg = str(entry.assistant_response)
            if isinstance(entry.assistant_response, dict):
                # Extract text from response
                resp = entry.assistant_response
                if 'choices' in resp and resp['choices']:
                    assistant_msg = resp['choices'][0].get('message', {}).get('content', str(resp))
                elif 'candidates' in resp and resp['candidates']:
                    parts = resp['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        assistant_msg = parts[0]['text']
            print(f"      Assistant: {assistant_msg[:50]}...")


async def example_5_history_without_context():
    """Compare behavior with and without history."""
    print("\n=== Example 5: With vs Without History ===")
    
    # Test without history
    print("\n🚫 Without history:")
    invoker_no_history = llmInvoker(strategy="failover", enable_history=False)
    try:
        invoker_no_history.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker_no_history.use_defaults()
    
    await invoker_no_history.invoke("My favorite color is blue.")
    response1 = await invoker_no_history.invoke("What's my favorite color?")
    
    if response1['success']:
        content = str(response1['response'])
        if isinstance(response1['response'], dict):
            raw_response = response1['response']
            if 'choices' in raw_response and len(raw_response['choices']) > 0:
                content = raw_response['choices'][0].get('message', {}).get('content', str(raw_response))
        print(f"Response: {content[:150]}...")
    
    # Test with history
    print("\n✅ With history:")
    invoker_with_history = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker_with_history.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker_with_history.use_defaults()
    
    await invoker_with_history.invoke("My favorite color is blue.")
    response2 = await invoker_with_history.invoke("What's my favorite color?")
    
    if response2['success']:
        content = str(response2['response'])
        if isinstance(response2['response'], dict):
            raw_response = response2['response']
            if 'choices' in raw_response and len(raw_response['choices']) > 0:
                content = raw_response['choices'][0].get('message', {}).get('content', str(raw_response))
        print(f"Response: {content[:150]}...")


async def example_6_advanced_history_management():
    """Advanced history management features."""
    print("\n=== Example 6: Advanced History Management ===")
    
    invoker = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker.use_defaults()
    
    # Create some history
    print("\n🔧 Creating history entries...")
    for i in range(3):
        await invoker.invoke(f"This is test message {i+1}")
    
    history = invoker.get_history()
    print(f"📊 Created {len(history.entries)} entries")
    
    # Inspect individual entries
    print("\n🔍 Inspecting entries:")
    for i, entry in enumerate(history.entries[:2], 1):  # Show first 2
        print(f"  Entry {i}:")
        print(f"    - Timestamp: {entry.timestamp}")
        print(f"    - Provider: {entry.provider}")
        print(f"    - Model: {entry.model}")
        print(f"    - User message: {entry.user_message[:50]}...")
    
    # Clear history
    print(f"\n🧹 Clearing history...")
    history.clear()
    print(f"📊 History entries after clear: {len(history.entries)}")


async def example_7_export_formats():
    """Different export formats and cleanup."""
    print("\n=== Example 7: Export Formats ===")
    
    invoker = llmInvoker(strategy="failover", enable_history=True)
    try:
        invoker.configure_providers(github=["gpt-4o-mini"])
    except ValueError:
        invoker.use_defaults()
    
    # Create sample conversation
    print("\n📝 Creating sample conversation...")
    await invoker.invoke("Hello, how are you?")
    await invoker.invoke("Tell me a short joke.")
    
    # Export in JSON format
    json_file = "conversation_export.json"
    history = invoker.get_history()
    history.export_to_json(json_file)
    print(f"✅ Exported to JSON: {json_file}")
    
    # Custom export to text format
    text_file = "conversation_export.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("Conversation History Export\n")
        f.write("=" * 30 + "\n\n")
        
        for i, entry in enumerate(history.entries, 1):
            f.write(f"Entry {i} - {entry.timestamp}\n")
            f.write(f"Provider: {entry.provider} / Model: {entry.model}\n")
            f.write(f"User: {entry.user_message}\n")
            
            # Handle assistant response safely
            assistant_response = entry.assistant_response
            if isinstance(assistant_response, dict):
                # Extract text from response dict
                if 'choices' in assistant_response and assistant_response['choices']:
                    assistant_response = assistant_response['choices'][0].get('message', {}).get('content', str(assistant_response))
                elif 'candidates' in assistant_response and assistant_response['candidates']:
                    parts = assistant_response['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'text' in parts[0]:
                        assistant_response = parts[0]['text']
                    else:
                        assistant_response = str(assistant_response)
                else:
                    assistant_response = str(assistant_response)
            
            f.write(f"Assistant: {str(assistant_response)[:200]}...\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"✅ Exported to text: {text_file}")
    
    # Cleanup files
    print(f"\n🧹 Cleaning up export files...")
    for file in [json_file, text_file]:
        try:
            Path(file).unlink()
            print(f"  ✅ Removed {file}")
        except FileNotFoundError:
            print(f"  ⚠️  {file} not found")


async def run_examples():
    """Run all conversation history examples."""
    try:
        await example_1_basic_history()
        await example_2_history_analysis()
        await example_3_export_import()
        await example_4_context_management()
        await example_5_history_without_context()
        await example_6_advanced_history_management()
        await example_7_export_formats()
        
        print("\n✅ All conversation history examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run examples."""
    print("🗨️  Conversation History Feature Examples")
    print("=" * 50)
    
    asyncio.run(run_examples())


if __name__ == "__main__":
    print(__doc__)
    main()