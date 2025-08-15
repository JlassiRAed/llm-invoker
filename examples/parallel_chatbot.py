#!/usr/bin/env python3
"""
Parallel Chatbot Example

This chatbot uses the parallel strategy to get responses from multiple providers
simultaneously and compare their outputs side-by-side.
"""

import asyncio
import sys
import os
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path to import llmInvoker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmInvoker import llmInvoker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ParallelChatbot:
    """A chatbot that compares responses from multiple LLM providers in parallel."""
    
    def __init__(self):
        self.invoker = llmInvoker(strategy="parallel", enable_history=True)
        self.conversation_count = 0
        self.setup_providers()
    
    def setup_providers(self):
        """Configure the providers for parallel comparison."""
        try:
            # Configure specific providers for comparison
            self.invoker.configure_providers(
                github=["gpt-4o"],                    # GitHub Models GPT-4o
                google=["gemini-2.0-flash"]           # Google Gemini 2.0 Flash (stable)
            )
            print("✅ Providers configured successfully!")
            print("📊 Will compare: GitHub GPT-4o vs Google Gemini 2.0 Flash")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("💡 Make sure you have GITHUB_TOKEN and GOOGLE_API_KEY in your .env file")
            sys.exit(1)
    
    def print_banner(self):
        """Print the chatbot banner."""
        print("=" * 80)
        print("🔄 PARALLEL LLM CHATBOT - Dual Response Comparison")
        print("=" * 80)
        print("💬 Compare responses from GitHub GPT-4o and Google Gemini 2.0 Flash")
        print("📝 Type your message and see both responses side-by-side")
        print("⚡ Commands: /help, /stats, /history, /clear, /quit")
        print("=" * 80)
    
    def print_help(self):
        """Print help information."""
        print("\n📋 Available commands:")
        print("  /help     - Show this help message")
        print("  /stats    - Show provider statistics")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the chatbot")
        print("  /exit     - Exit the chatbot")
        print("\n💬 Just type your message to get parallel responses!")
    
    def format_response(self, provider: str, model: str, response: Any, response_time: float = None) -> str:
        """Format a single provider response."""
        # Extract text content from different response formats
        if isinstance(response, dict):
            # Google Gemini format
            if 'candidates' in response:
                text = response['candidates'][0]['content']['parts'][0]['text']
            # OpenAI/GitHub format
            elif 'choices' in response:
                text = response['choices'][0]['message']['content']
            # Other formats
            elif 'content' in response:
                text = response['content']
            else:
                text = str(response)
        else:
            text = str(response)
        
        # Truncate very long responses for better display
        if len(text) > 1000:
            text = text[:997] + "..."
        
        time_info = f" ({response_time:.2f}s)" if response_time else ""
        header = f"🤖 {provider.upper()} ({model}){time_info}"
        separator = "─" * len(header)
        
        return f"\n{header}\n{separator}\n{text}\n"
    
    def display_parallel_responses(self, result: Dict[str, Any]):
        """Display responses from all providers in parallel."""
        if not result.get('success'):
            error = result.get('error', 'Unknown error')
            print(f"\n❌ No responses received from any provider")
            if error != 'Unknown error':
                print(f"💡 Error: {error}")
            return
        
        # Get responses from the parallel strategy result
        responses = result.get('results', {})
        successful_responses = result.get('successful_responses', [])
        
        if not responses and not successful_responses:
            print("\n❌ No responses received from any provider")
            return
        
        # Count successful responses
        success_count = len([r for r in responses.values() if r.get('success', False)])
        
        print("\n" + "=" * 80)
        print(f"📊 PARALLEL RESPONSES - {success_count} providers responded")
        print("=" * 80)
        
        # Display each response
        for provider_model, response_data in responses.items():
            try:
                # Parse provider info (format: "provider:model")
                if ':' in provider_model:
                    provider, model = provider_model.split(':', 1)
                else:
                    provider, model = provider_model, "unknown"
                
                if response_data.get('success', False):
                    response = response_data.get('response', 'No response')
                    response_time = response_data.get('response_time')
                    
                    formatted = self.format_response(provider, model, response, response_time)
                    print(formatted)
                else:
                    error = response_data.get('error', 'Unknown error')
                    print(f"\n❌ {provider.upper()} ({model}) failed: {error}")
                
            except Exception as e:
                print(f"\n❌ Error formatting response from {provider_model}: {e}")
        
        print("=" * 80)
    
    def show_stats(self):
        """Show provider statistics."""
        try:
            stats = self.invoker.get_provider_stats()
            print("\n📊 Provider Statistics:")
            print("-" * 40)
            
            for provider, data in stats.get('providers', {}).items():
                status = "✅" if data.get('configured', False) else "❌"
                models = ", ".join(data.get('models', []))
                print(f"{status} {provider.capitalize()}: {models}")
            
            print(f"\nTotal conversations: {self.conversation_count}")
            print(f"Strategy: {stats.get('strategy', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def show_history(self):
        """Show conversation history."""
        try:
            history = self.invoker.get_history()
            entries = history.entries if hasattr(history, 'entries') else []
            
            if not entries:
                print("\n📝 No conversation history yet.")
                return
            
            print(f"\n📝 Conversation History ({len(entries)} entries):")
            print("-" * 50)
            
            for i, entry in enumerate(entries[-10:], 1):  # Show last 10
                timestamp = entry.get('timestamp', 'Unknown time')
                prompt = entry.get('prompt', 'Unknown prompt')[:100]
                provider = entry.get('provider', 'Unknown')
                
                print(f"{i}. [{timestamp}] via {provider}")
                print(f"   Q: {prompt}...")
                
        except Exception as e:
            print(f"❌ Error getting history: {e}")
    
    def clear_history(self):
        """Clear conversation history."""
        try:
            # Reset the invoker to clear history
            self.invoker = llmInvoker(strategy="parallel", enable_history=True)
            self.setup_providers()
            self.conversation_count = 0
            print("✅ Conversation history cleared!")
        except Exception as e:
            print(f"❌ Error clearing history: {e}")
    
    async def chat_loop(self):
        """Main chat interaction loop."""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command == '/help':
                        self.print_help()
                    elif command == '/stats':
                        self.show_stats()
                    elif command == '/history':
                        self.show_history()
                    elif command == '/clear':
                        self.clear_history()
                    elif command in ['/quit', '/exit']:
                        print("\n👋 Goodbye! Thanks for using the parallel chatbot!")
                        break
                    else:
                        print(f"❌ Unknown command: {command}")
                        print("💡 Type /help for available commands")
                    
                    continue
                
                # Process regular message
                print("\n⏳ Querying both providers in parallel...")
                start_time = asyncio.get_event_loop().time()
                
                try:
                    # Get parallel responses
                    result = await self.invoker.invoke(
                        message=user_input,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    end_time = asyncio.get_event_loop().time()
                    total_time = end_time - start_time
                    
                    # Display results
                    self.display_parallel_responses(result)
                    print(f"⚡ Total response time: {total_time:.2f}s")
                    
                    self.conversation_count += 1
                    
                except Exception as e:
                    print(f"\n❌ Error during parallel invocation: {e}")
                    print("💡 Make sure your API keys are valid and providers are available")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the parallel chatbot!")
                break
            except EOFError:
                print("\n\n👋 Goodbye! Thanks for using the parallel chatbot!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("💡 Continuing... Type /quit to exit")

def check_environment():
    """Check if required environment variables are set."""
    required_vars = {
        'GITHUB_TOKEN': 'GitHub Models access',
        'GOOGLE_API_KEY': 'Google AI access'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var} ({description})")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\n💡 Add these to your .env file:")
        print("GITHUB_TOKEN=ghp_your-github-token")
        print("GOOGLE_API_KEY=your-google-api-key")
        return False
    
    return True

async def main():
    """Main function to run the parallel chatbot."""
    print("🔄 Starting Parallel LLM Chatbot...")
    
    # Check environment
    if not check_environment():
        return
    
    # Create and run chatbot
    chatbot = ParallelChatbot()
    await chatbot.chat_loop()

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Chatbot interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
