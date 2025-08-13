"""Conversation history management for maintaining context across provider switches."""
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class ConversationEntry:
    """Represents a single conversation entry."""
    
    def __init__(
        self, 
        user_message: Union[str, List[Dict]], 
        assistant_response: Any, 
        provider: str, 
        model: str,
        timestamp: Optional[datetime] = None
    ):
        self.user_message = user_message
        self.assistant_response = assistant_response
        self.provider = provider
        self.model = model
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary format."""
        return {
            'user_message': self.user_message,
            'assistant_response': self.assistant_response,
            'provider': self.provider,
            'model': self.model,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_langchain_messages(self) -> List[Dict[str, str]]:
        """Convert entry to LangChain message format."""
        messages = []
        
        # Add user message
        if isinstance(self.user_message, str):
            messages.append({"role": "user", "content": self.user_message})
        elif isinstance(self.user_message, list):
            messages.extend(self.user_message)
        
        # Add assistant response
        if isinstance(self.assistant_response, dict):
            # Extract content from response
            content = self._extract_content_from_response(self.assistant_response)
            if content:
                messages.append({"role": "assistant", "content": content})
        elif isinstance(self.assistant_response, str):
            messages.append({"role": "assistant", "content": self.assistant_response})
        
        return messages
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from various response formats."""
        # OpenAI format
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice:
                return choice['message'].get('content', '')
            elif 'text' in choice:
                return choice['text']
        
        # Anthropic format
        if 'content' in response:
            if isinstance(response['content'], list) and len(response['content']) > 0:
                return response['content'][0].get('text', '')
            elif isinstance(response['content'], str):
                return response['content']
        
        # Google format
        if 'candidates' in response and len(response['candidates']) > 0:
            candidate = response['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text']
        
        # Fallback to string representation
        return str(response)


class ConversationHistory:
    """Manages conversation history to maintain context across provider switches."""
    
    def __init__(self, max_entries: int = 50, max_context_tokens: int = 4000):
        self.entries: List[ConversationEntry] = []
        self.max_entries = max_entries
        self.max_context_tokens = max_context_tokens
    
    def add_interaction(
        self, 
        user_message: Union[str, List[Dict]], 
        assistant_response: Any, 
        provider: str, 
        model: str
    ) -> None:
        """Add a new conversation interaction."""
        entry = ConversationEntry(user_message, assistant_response, provider, model)
        self.entries.append(entry)
        
        # Maintain maximum entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_context_messages(self, include_last_n: int = 10) -> List[Dict[str, str]]:
        """Get conversation context as LangChain-compatible messages."""
        if not self.entries:
            return []
        
        # Get last N entries
        recent_entries = self.entries[-include_last_n:] if include_last_n > 0 else self.entries
        
        messages = []
        for entry in recent_entries:
            messages.extend(entry.to_langchain_messages())
        
        return messages
    
    def get_enhanced_messages(
        self, 
        new_message: Union[str, List[Dict], Dict], 
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Get enhanced messages with conversation context."""
        if not include_context or not self.entries:
            if isinstance(new_message, str):
                return [{"role": "user", "content": new_message}]
            elif isinstance(new_message, dict):
                return [new_message]
            elif isinstance(new_message, list):
                return new_message
        
        # Get context and add new message
        context_messages = self.get_context_messages()
        
        if isinstance(new_message, str):
            context_messages.append({"role": "user", "content": new_message})
        elif isinstance(new_message, dict):
            context_messages.append(new_message)
        elif isinstance(new_message, list):
            context_messages.extend(new_message)
        
        return context_messages
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation history."""
        if not self.entries:
            return {
                'total_entries': 0,
                'providers_used': [],
                'models_used': [],
                'first_interaction': None,
                'last_interaction': None
            }
        
        providers_used = list(set(entry.provider for entry in self.entries))
        models_used = list(set(f"{entry.provider}:{entry.model}" for entry in self.entries))
        
        return {
            'total_entries': len(self.entries),
            'providers_used': providers_used,
            'models_used': models_used,
            'first_interaction': self.entries[0].timestamp.isoformat(),
            'last_interaction': self.entries[-1].timestamp.isoformat()
        }
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.entries = []
    
    def export_to_json(self, file_path: str) -> None:
        """Export conversation history to JSON file."""
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_entries': len(self.entries)
            },
            'entries': [entry.to_dict() for entry in self.entries]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, file_path: str) -> None:
        """Import conversation history from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entries = []
        for entry_data in data.get('entries', []):
            entry = ConversationEntry(
                user_message=entry_data['user_message'],
                assistant_response=entry_data['assistant_response'],
                provider=entry_data['provider'],
                model=entry_data['model'],
                timestamp=datetime.fromisoformat(entry_data['timestamp'])
            )
            self.entries.append(entry)
    
    def get_provider_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about provider and model usage."""
        stats = {}
        
        for entry in self.entries:
            provider_key = entry.provider
            model_key = entry.model
            
            if provider_key not in stats:
                stats[provider_key] = {}
            
            if model_key not in stats[provider_key]:
                stats[provider_key][model_key] = 0
            
            stats[provider_key][model_key] += 1
        
        return stats
    
    def get_recent_failures(self, provider: str, model: str, hours: int = 1) -> int:
        """Get count of recent failures for a specific provider/model combination."""
        # This would be enhanced to track failures separately
        # For now, return 0 as a placeholder
        return 0