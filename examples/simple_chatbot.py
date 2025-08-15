#!/usr/bin/env python3
"""
Simple Chatbot using llm-invoker

A simple command-line chatbot that demonstrates how to use llm-invoker
for interactive conversations with automatic provider failover.
"""

import asyncio
import sys
import os
from typing import Optional
from dotenv import load_dotenv

# Add the parent directory to the path to import llmInvoker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llmInvoker import llmInvoker

# Load environment variables
load_dotenv()

class SimpleChatBot:
    """A simple chatbot using llm-invoker with failover capabilities."""
    
    def __init__(self):
        """Initialize the chatbot with available providers."""
        self.invoker = llmInvoker(strategy="failover", enable_history=True)
        self.setup_providers()
        self.conversation_active = True
        
    def setup_providers(self):
        """Setup providers based on available API keys."""
        available_providers = self.get_available_providers()
        
        if not available_providers:
            print("❌ No API keys found! Please configure your .env file.")
            print("\nRequired environment variables:")
            print("- GITHUB_TOKEN (for GitHub Models - Free tier available)")
            print("- OPENAI_API_KEY (for OpenAI models)")
            print("- ANTHROPIC_API_KEY (for Anthropic Claude)")
            print("- GOOGLE_API_KEY (for Google Gemini)")
            print("- OPENROUTER_API_KEY (for OpenRouter)")
            print("- HUGGINGFACE_API_KEY (for Hugging Face)")
            sys.exit(1)
            
        print(f"🔧 Configuring providers: {', '.join(available_providers.keys())}")
        self.invoker.configure_providers(**available_providers)
        
    def get_available_providers(self) -> dict:
        """Detect available providers based on environment variables."""
        providers = {}
        
        # GitHub Models (Free tier)
        if os.getenv('GITHUB_TOKEN'):
            providers['github'] = ['gpt-4o', 'gpt-4o-mini']
            
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            providers['openai'] = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
            
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            providers['anthropic'] = ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307']
            
        # Google AI
        if os.getenv('GOOGLE_API_KEY'):
            providers['google'] = ['gemini-2.0-flash-exp', 'gemini-1.5-pro']
            
        # OpenRouter
        if os.getenv('OPENROUTER_API_KEY'):
            providers['openrouter'] = ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o']
            
        # Hugging Face
        if os.getenv('HUGGINGFACE_API_KEY'):
            providers['huggingface'] = ['microsoft/DialoGPT-medium']
            
        return providers
    
    def print_welcome_message(self):
        """Print the welcome message and instructions."""
        print("\n" + "=" * 60)
        print("🤖 llm-invoker Simple Chatbot")
        print("=" * 60)
        print("Features:")
        print("✅ Automatic provider failover")
        print("✅ Conversation history maintained")
        print("✅ Multiple LLM providers supported")
        print("\nCommands:")
        print("• Type your message and press Enter")
        print("• '/help' - Show this help message")
        print("• '/history' - Show conversation history")
        print("• '/stats' - Show provider statistics")
        print("• '/clear' - Clear conversation history")
        print("• '/quit' or '/exit' - Exit the chatbot")
        print("• Ctrl+C - Force exit")
        print("=" * 60)
        
    def print_help(self):
        """Print help information."""
        print("\n📖 Help:")
        print("• Just type your message and press Enter to chat")
        print("• The bot will automatically try different providers if one fails")
        print("• Your conversation history is preserved across provider switches")
        print("• Use '/stats' to see which providers were used")
        
    def print_history(self):
        """Print conversation history summary."""
        history = self.invoker.get_history()
        if not history or not history.entries:
            print("\n📝 No conversation history yet.")
            return
            
        print(f"\n📝 Conversation History ({len(history.entries)} messages):")
        print(f"First message: {history.first_interaction}")
        print(f"Last message: {history.last_interaction}")
        
        # Show last few messages
        recent_entries = history.entries[-3:] if len(history.entries) > 3 else history.entries
        print("\n🔸 Recent messages:")
        for entry in recent_entries:
            timestamp = entry.timestamp.strftime("%H:%M:%S")
            provider = entry.response.get('provider', 'unknown')
            user_msg = entry.message[:50] + "..." if len(entry.message) > 50 else entry.message
            print(f"  [{timestamp}] You: {user_msg}")
            print(f"  [{timestamp}] AI ({provider}): Response received")
            
    def print_stats(self):
        """Print provider statistics."""
        stats = self.invoker.get_provider_stats()
        print(f"\n📊 Provider Statistics:")
        print(f"Strategy: {stats.get('strategy', 'unknown')}")
        print(f"Total providers configured: {stats.get('total_providers', 0)}")
        print(f"Total models available: {stats.get('total_models', 0)}")
        
        providers_info = stats.get('providers', {})
        for provider_name, info in providers_info.items():
            status = "✅" if info.get('configured') else "❌"
            models = ', '.join(info.get('models', []))
            print(f"  {status} {provider_name}: {models}")
            
    def clear_history(self):
        """Clear conversation history."""
        # Recreate invoker to clear history
        self.invoker = llmInvoker(strategy="failover", enable_history=True)
        self.setup_providers()
        print("\n🧹 Conversation history cleared!")
        
    async def process_command(self, user_input: str) -> bool:
        """Process special commands. Returns True if command was processed."""
        command = user_input.lower().strip()
        
        if command in ['/quit', '/exit']:
            print("\n👋 Thanks for using llm-invoker chatbot! Goodbye!")
            return True
            
        elif command == '/help':
            self.print_help()
            return True
            
        elif command == '/history':
            self.print_history()
            return True
            
        elif command == '/stats':
            self.print_stats()
            return True
            
        elif command == '/clear':
            self.clear_history()
            return True
            
        return False
        
    async def get_ai_response(self, user_input: str) -> Optional[str]:
        """Get AI response using llm-invoker."""
        try:
            print("🤔 Thinking...", end="", flush=True)
            
            response = await self.invoker.invoke(
                message=user_input,
                temperature=0.7,
                max_tokens=500  # Reasonable limit for chat
            )
            
            print("\r" + " " * 15 + "\r", end="")  # Clear "Thinking..." message
            
            if response['success']:
                # Extract the actual text response
                ai_response = response.get('response', '')
                provider = response.get('provider', 'unknown')
                model = response.get('model', 'unknown')
                
                # Handle different response formats
                if isinstance(ai_response, dict):
                    if 'choices' in ai_response and ai_response['choices']:
                        ai_response = ai_response['choices'][0]['message']['content']
                    elif 'candidates' in ai_response and ai_response['candidates']:
                        parts = ai_response['candidates'][0]['content']['parts']
                        ai_response = parts[0]['text'] if parts else str(ai_response)
                    else:
                        ai_response = str(ai_response)
                        
                print(f"🤖 AI ({provider}:{model}):")
                return ai_response
            else:
                error_msg = response.get('error', 'Unknown error')
                print(f"❌ All providers failed: {error_msg}")
                return None
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None
            
    async def run(self):
        """Main chatbot loop."""
        self.print_welcome_message()
        
        try:
            while self.conversation_active:
                # Get user input
                try:
                    user_input = input("\n💬 You: ").strip()
                except EOFError:
                    break
                    
                if not user_input:
                    continue
                    
                # Process commands
                if await self.process_command(user_input):
                    if user_input.lower() in ['/quit', '/exit']:
                        break
                    continue
                    
                # Get AI response
                ai_response = await self.get_ai_response(user_input)
                if ai_response:
                    print(ai_response)
                    
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot interrupted. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            
        finally:
            # Show final stats
            print("\n" + "=" * 40)
            print("📊 Session Summary:")
            self.print_stats()
            print("=" * 40)


async def main():
    """Main function."""
    chatbot = SimpleChatBot()
    await chatbot.run()


if __name__ == "__main__":
    asyncio.run(main())
