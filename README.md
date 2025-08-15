# llm-invoker


[![PyPI version](https://badge.fury.io/py/llm-invoker.svg)](https://badge.fury.io/py/llm-invoker)
[![Python Support](https://img.shields.io/pypi/pyversions/llm-invoker.svg)](https://pypi.org/project/llm-invoker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for managing multi-agent model invocation with automatic failover strategies, designed for POC development with seamless provider switching and conversation history management.

## 🎯 Why This Project Exists

### The Problem
During the development of multi-agent systems and proof-of-concept projects, developers face several recurring challenges:

1. **Rate Limiting**: Free and low-cost LLM providers impose strict rate limits, causing interruptions during active development
2. **Provider Reliability**: Individual providers can experience downtime or temporary service issues
3. **Model Comparison**: Developers need to test the same prompts across different models and providers to find the best fit
4. **Context Loss**: When switching between providers manually, conversation history and context are often lost
5. **Configuration Complexity**: Managing multiple API keys and provider configurations becomes cumbersome

### The Solution
`llmInvoker` was created to solve these exact problems by providing:

- **Automatic Provider Switching**: When one provider hits rate limits or fails, automatically switch to the next available provider
- **Context Preservation**: Maintain conversation history across provider switches, ensuring continuity
- **Unified Interface**: Single API to interact with multiple LLM providers (GitHub Models, OpenRouter, Google, OpenAI, Anthropic, etc.)
- **Development-Focused**: Optimized for rapid prototyping and POC development workflows
- **Zero Configuration**: Works out of the box with sensible defaults, but fully customizable when needed

This library was born from real-world frustration during multi-agent system development, where hitting rate limits would halt development flow and require manual intervention to switch providers.

## ✨ Features

- **🔄 Automatic Failover**: Seamlessly switch between providers when rate limits or errors occur
- **⚡ Parallel Invocation**: Compare responses from multiple models simultaneously  
- **💭 Conversation History**: Maintain context across provider switches
- **🔌 Multi-Provider Support**: GitHub Models, OpenRouter, Google Generative AI, Hugging Face, OpenAI, Anthropic
- **🔍 LangSmith Integration**: Monitor token usage and trace executions
- **🛠️ LangChain Compatible**: Easy integration with existing multi-agent frameworks
- **⚙️ Simple Configuration**: Environment-based API key management with code-level provider setup

## 🚀 Installation

```bash
# Using uv (recommended for modern Python projects)
uv add llm-invoker

# Using pip
pip install llm-invoker

# For development/contribution
git clone https://github.com/RaedJlassi/llm-invoker.git
cd llm-invoker
uv sync --dev
```

## ⚙️ Environment Setup

Create a `.env` file in your project root with your API keys (add only the providers you plan to use):

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# GitHub Models API Key (free tier available)
GITHUB_TOKEN=your_github_token_here

# Google Generative AI API Key
GOOGLE_API_KEY=your_google_api_key_here

# Hugging Face API Key
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# OpenRouter API Key (aggregates multiple providers)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# LangSmith Configuration (optional - for monitoring)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=multiagent_failover_poc
```

> **Note**: You don't need all API keys. The library will automatically detect which providers are available based on your environment variables.

## 🎯 Use Cases

This library is particularly useful for:

### 🔬 Research & Prototyping
- **Multi-agent system development** where different agents might use different models
- **POC development** where you need reliable access to LLMs without manual intervention
- **Comparing model outputs** across different providers for research purposes

### 🏗️ Development Workflows
- **Rate limit management** during intensive development sessions
- **Provider redundancy** for production applications that can't afford downtime
- **Cost optimization** by utilizing free tiers across multiple providers

### 🤖 Multi-Agent Applications
- **Agent swarms** where different agents can use different models
- **Fallback strategies** for critical agent communications
- **Context preservation** when agents switch between conversation partners

### 📊 Model Evaluation
- **A/B testing** different models on the same prompts
- **Performance benchmarking** across providers
- **Response quality comparison** for specific use cases

## 🚀 Quick Start

### 1. Installation & Setup

```python
# Convenience function (recommended for simple use cases)
from llmInvoker import invoke_failover

response = invoke_failover(
    message="Explain quantum computing in simple terms",
    providers={
        "github": ["gpt-4o", "gpt-4o-mini"],
        "google": ["gemini-2.0-flash-exp"]
    }
)

if response['success']:
    print(response['response'])
```

### 2. Class-based Usage

```python
from llmInvoker import llmInvoker

# Initialize with custom configuration
invoker = llmInvoker(
    strategy="failover",
    max_retries=3,
    timeout=30,
    enable_history=True
)

### Convenience Functions

```python
from llmInvoker import invoke_failover, invoke_parallel

# Quick failover
response = invoke_failover(
    "What are the benefits of renewable energy?",
    providers={
        "github": ["gpt-4o"],
        "google": ["gemini-2.0-flash-exp"]
    }
)

# Parallel comparison
response = invoke_parallel(
    "Explain machine learning in one sentence",
    providers={
        "github": ["gpt-4o"],
        "openrouter": ["deepseek/deepseek-r1"],
        "google": ["gemini-2.0-flash-exp"]
    }
)

# Compare responses from all providers
for result in response['successful_responses']:
    print(f"{result['provider']}: {result['response']}")
```

## 📋 Strategies

### 1. Failover Strategy
Tries providers in order until one succeeds:

```python
invoker = llmInvoker(strategy="failover")
invoker.configure_providers(
    github=["gpt-4o", "gpt-4o-mini"],
    google=["gemini-2.0-flash-exp"],
    openrouter=["deepseek/deepseek-r1"]
)
```

### 2. Parallel Strategy  
Invokes all providers simultaneously for comparison:

```python
invoker = llmInvoker(strategy="parallel")
# Same configuration as above
response = invoker.invoke_sync("Your question here")
# Get multiple responses to compare
```

## 📋 Response Structure

All methods return a standardized response structure for consistency across different strategies and providers:

### Successful Response
```python
{
    'success': True,
    'response': str,           # The actual LLM response content
    'provider': str,           # Provider name (e.g., 'github', 'google', 'openrouter')
    'model': str,              # Model name (e.g., 'gpt-4o', 'gemini-2.0-flash-exp')
    'timestamp': str,          # ISO format timestamp
    'attempt': int,            # Number of attempts made (for failover strategy)
    'all_responses': list,     # All responses (for parallel strategy)
    'metadata': dict           # Additional metadata (tokens, latency, etc.)
}
```

### Failed Response
```python
{
    'success': False,
    'error': str,              # Error message describing what went wrong
    'provider': str,           # Last attempted provider (if any)
    'model': str,              # Last attempted model (if any)
    'timestamp': str,          # ISO format timestamp
    'attempt': int,            # Number of attempts made
    'metadata': dict           # Error metadata and debugging info
}
```

### Parallel Strategy Response
When using parallel strategy, successful responses include additional fields:
```python
{
    'success': True,
    'response': str,           # Response from the first successful provider
    'all_responses': [         # List of all provider responses
        {
            'provider': str,
            'model': str,
            'response': str,   # Individual provider response
            'success': bool,   # Whether this specific provider succeeded
            'error': str,      # Error message if failed
            'metadata': dict   # Provider-specific metadata
        },
        # ... more provider responses
    ],
    'provider': str,           # First successful provider
    'model': str,              # First successful model
    'timestamp': str,
    'metadata': dict
}
```

### Metadata Fields
The `metadata` field may contain:
```python
{
    'tokens': {
        'prompt': int,         # Tokens used for prompt
        'completion': int,     # Tokens used for completion
        'total': int          # Total tokens used
    },
    'latency': float,         # Response time in seconds
    'rate_limit_info': dict,  # Rate limiting information
    'provider_config': dict,  # Provider-specific configuration used
    'langsmith_run_id': str,  # LangSmith tracking ID (if enabled)
    'conversation_id': str    # Conversation tracking ID (if history enabled)
}
```

## 🔧 Advanced Configuration

### String-based Configuration

```python
# Configure providers using string format
invoker.configure_from_string(
    "github['gpt-4o','gpt-4o-mini'],google['gemini-2.0-flash-exp'],openrouter['deepseek/deepseek-r1']"
)
```

### Default Configurations

```python
# Use default configurations for all available providers
invoker.use_defaults()
```

### Custom Parameters

```python
# Add model parameters
response = invoker.invoke_sync(
    "Your question",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9
)
```

## 🤖 LangChain Integration

Seamlessly integrate with existing LangChain workflows:

```python
from llmInvoker import llmInvoker

class LangChainWrapper:
    def __init__(self):
        self.invoker = llmInvoker(strategy="failover")
        self.invoker.use_defaults()
    
    async def __call__(self, prompt: str) -> str:
        response = await self.invoker.invoke(prompt)
        if response['success']:
            return self._extract_content(response['response'])
        raise Exception(f"All providers failed: {response['error']}")

# Use in LangChain chains
llm_wrapper = LangChainWrapper()
```

## 📊 Monitoring & History

### Conversation History

```python
# Enable history (default)
invoker = llmInvoker(enable_history=True)

# Get conversation summary
history = invoker.get_history()
summary = history.get_summary()
print(f"Total interactions: {summary['total_entries']}")
print(f"Providers used: {summary['providers_used']}")

# Export/import history
invoker.export_history("conversation_history.json")
invoker.import_history("conversation_history.json")
```

### Provider Statistics

```python
stats = invoker.get_provider_stats()
print(f"Total providers: {stats['total_providers']}")
print(f"Total models: {stats['total_models']}")
print(f"Provider details: {stats['providers']}")
```

### LangSmith Integration

Automatic token usage tracking and execution tracing when LangSmith is configured:

```bash
# In .env file
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
```

## 🔍 Examples

Check the `examples/` directory for complete examples:

- `failover_example.py` - Comprehensive failover strategy examples
- `parallel_invoke_example.py` - Parallel invocation and response comparison
- `langchain_integration.py` - Integration with LangChain and multi-agent frameworks

## 🛠️ Development

### Project Structure

```
multiagent_failover_invoke/
├── multiagent_failover_invoke/
│   ├── __init__.py          # Main exports
│   ├── core.py              # Core llmInvoker class  
│   ├── providers.py         # Provider implementations
│   ├── strategies.py        # Strategy implementations
│   ├── history.py           # Conversation history management
│   ├── config.py            # Configuration management
│   └── utils.py             # Utility functions
├── examples/                # Usage examples
├── tests/                   # Test suite
├── .env.example            # Environment template
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 🔧 Supported Providers

llm-invoker supports **6 major AI providers** with automatic failover between them:

### **Providers & Models**

| Provider | Models Supported | API Key Required | Status |
|----------|------------------|------------------|--------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` | `OPENAI_API_KEY` | ✅ Active |
| **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-3-haiku`, `claude-3-opus` | `ANTHROPIC_API_KEY` | ✅ Active |
| **GitHub Models** | `gpt-4o`, `gpt-4o-mini` | `GITHUB_TOKEN` | ✅ Active |
| **Google AI** | `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash` | `GOOGLE_API_KEY` | ✅ Active |
| **OpenRouter** | `anthropic/claude-3.5-sonnet`, `openai/gpt-4o` | `OPENROUTER_API_KEY` | ✅ Active |
| **Hugging Face** | Any text-generation model | `HUGGINGFACE_API_KEY` | ✅ Active |

### **Configuration Example**

```python
from llmInvoker import llmInvoker

# Configure multiple providers for failover
invoker = llmInvoker(strategy="failover")
invoker.configure_providers(
    openai=["gpt-4o", "gpt-4o-mini"],
    anthropic=["claude-3-5-sonnet-20241022"],
    github=["gpt-4o"],
    google=["gemini-2.0-flash-exp"],
    openrouter=["anthropic/claude-3.5-sonnet"],
    huggingface=["microsoft/DialoGPT-medium"]
)
```

## 🤝 Use Cases

Perfect for:

- **POC Development**: Rapid prototyping without worrying about rate limits
- **Multi-Agent Systems**: LangGraph, CrewAI, AutoGen integration
- **Model Comparison**: A/B testing different models on same tasks  
- **Reliability**: Production backup strategies for mission-critical applications
- **Cost Optimization**: Prefer free models, fallback to paid when needed

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! This project was created to solve real-world development challenges, and we'd love to hear about your use cases and improvements.

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/yourusername/multiagent-failover-invoke.git
cd multiagent-failover-invoke
uv sync --dev
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=llmInvoker

# Run specific test file
pytest tests/test_core.py
```

## 📚 Examples

The `examples/` directory contains comprehensive examples:

- **`failover_example.py`** - Basic and advanced failover strategies
- **`parallel_invoke_example.py`** - Parallel model invocation
- **`multimodal_example.py`** - Working with images and multimodal content
- **`langchain_integration.py`** - Integration with LangChain workflows
- **`quickstart.py`** - Quick start guide examples

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/JlassiRAed/llm-invoker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JlassiRAed/llm-invoker/discussions)
- **Documentation**: Comprehensive examples in the `examples/` directory

## 👨‍💻 Author

**Jlassi Raed**
- Email: raed.jlassi@etudiant-enit.utm.tn
- GitHub: [@RaedJlassi](https://github.com/JlassiRAed)

*Created during multi-agent system development at ENIT (École Nationale d'Ingénieurs de Tunis) to solve real-world rate limiting and provider reliability challenges in POC phase.*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the LLM providers for their APIs and free tiers that make development accessible
- Inspired by real-world challenges in multi-agent system development
- Built for the developer community facing similar rate limiting and reliability issues

---

**⭐ If this project helps you in your development workflow, please consider giving it a star!**