"""
Multimodal Content Example

This example demonstrates how to use the library with multimodal content
(text + images, text + files, etc.) for providers that support it.
"""

import asyncio
import base64
from llmInvoker import llmInvoker


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        # Return a dummy base64 for demonstration
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


async def example_1_image_analysis():
    """Example of analyzing an image with text prompt."""
    print("=== Example 1: Image Analysis ===")
    
    invoker = llmInvoker(strategy="failover")
    
    # Configure providers that support vision (OpenAI GPT-4V, Google Gemini)
    invoker.configure_providers(
        openai=["gpt-4o", "gpt-4o-mini"], 
        google=["gemini-2.0-flash-exp"],  
        anthropic=["claude-3-5-sonnet-20241022"]  
    )
    
    # Create multimodal message with image
    image_base64 = encode_image_to_base64("example_image.jpg")  # You would provide a real image
    
    multimodal_message = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this image? Describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    try:
        response = await invoker.invoke(message=multimodal_message)
        
        if response['success']:
            print(f"✓ Image analyzed by {response['provider']}:")
            print(response['response'])
        else:
            print(f"✗ Failed: {response['error']}")
            
    except Exception as e:
        print(f"Error: {e}")


async def example_2_document_analysis():
    """Example of analyzing a document with multiple providers."""
    print("\n=== Example 2: Document Analysis ===")
    
    invoker = llmInvoker(strategy="parallel")  # Compare responses
    
    # Use providers that can handle long text
    invoker.configure_providers(
        openai=["gpt-4o"],
        anthropic=["claude-3-5-sonnet-20241022"],
        google=["gemini-2.0-flash-exp"]
    )
    
    # Simulate a long document
    document_content = """
    This is a sample business proposal document.
    
    Executive Summary:
    Our company proposes to develop an AI-powered customer service solution
    that will reduce response times by 60% and improve customer satisfaction.
    
    Key Benefits:
    1. 24/7 availability
    2. Consistent responses
    3. Cost reduction
    4. Scalability
    
    Implementation Timeline:
    Phase 1: Research and Development (3 months)
    Phase 2: Prototype Development (2 months)  
    Phase 3: Testing and Optimization (2 months)
    Phase 4: Deployment (1 month)
    
    Budget: $500,000
    Expected ROI: 200% within 18 months
    """
    
    message = [
        {
            "role": "user",
            "content": f"Please analyze this business proposal and provide:\n1. Key strengths\n2. Potential risks\n3. Recommendations for improvement\n\nDocument:\n{document_content}"
        }
    ]
    
    response = await invoker.invoke(message=message, max_tokens=1000)
    
    if response['success']:
        print(f"Received {len(response['successful_responses'])} analyses:")
        
        for i, result in enumerate(response['successful_responses'], 1):
            print(f"\n--- Analysis {i} from {result['provider']} ---")
            print(result['response'])
            print("-" * 50)
    else:
        print("No successful analyses received")


async def example_3_code_review():
    """Example of code review with multimodal content."""
    print("\n=== Example 3: Code Review ===")
    
    invoker = llmInvoker(strategy="failover")
    invoker.use_defaults()
    
    # Code to review
    code_content = '''
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fibonacci({i}) = {fibonacci(i)}")
    '''
    
    message = [
        {
            "role": "user",
            "content": f"""Please review this Python code and provide:
1. Code quality assessment
2. Performance issues
3. Suggestions for improvement
4. Alternative implementations

Code:
```python
{code_content}
```"""
        }
    ]
    
    response = await invoker.invoke(message=message, temperature=0.3)
    
    if response['success']:
        print(f"Code review from {response['provider']}:")
        print(response['response'])
    else:
        print(f"Code review failed: {response['error']}")


async def example_4_mixed_content_conversation():
    """Example of a conversation with mixed content types."""
    print("\n=== Example 4: Mixed Content Conversation ===")
    
    invoker = llmInvoker(strategy="failover", enable_history=True)
    invoker.use_defaults()
    
    # First: Text-only message
    response1 = await invoker.invoke(
        message="I'm working on a data science project. Can you help me choose between Python and R?"
    )
    
    if response1['success']:
        print("First response (text):", response1['response'][:200] + "...")
    
    # Second: Follow-up with code example
    code_example = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2']]
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
'''
    
    response2 = await invoker.invoke(
        message=f"Here's my current Python code. Can you suggest improvements?\n\n```python\n{code_example}\n```"
    )
    
    if response2['success']:
        print("\nSecond response (code review):", response2['response'][:200] + "...")
    
    # Third: Request for visualization
    response3 = await invoker.invoke(
        message="Can you suggest some data visualization techniques for this type of analysis?"
    )
    
    if response3['success']:
        print("\nThird response (visualization):", response3['response'][:200] + "...")
    
    # Show conversation history
    history = invoker.get_history()
    if history:
        summary = history.get_summary()
        print(f"\nConversation summary: {summary['total_entries']} interactions")
        print(f"Providers used: {summary['providers_used']}")


async def example_5_file_content_analysis():
    """Example of analyzing file content with different formats."""
    print("\n=== Example 5: File Content Analysis ===")
    
    invoker = llmInvoker(strategy="parallel")
    invoker.use_defaults()
    
    # Simulate different file types
    files_content = {
        "csv_data": """
Name,Age,Department,Salary
Alice,28,Engineering,75000
Bob,32,Marketing,65000
Charlie,45,Engineering,95000
Diana,29,Sales,55000
""",
        "json_config": """
{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "company_db"
    },
    "api": {
        "version": "v1",
        "rate_limit": 1000
    }
}
""",
        "log_excerpt": """
2024-01-15 10:23:45 INFO: User login successful - user_id: 12345
2024-01-15 10:24:12 ERROR: Database connection failed - timeout after 30s
2024-01-15 10:24:15 INFO: Retrying database connection
2024-01-15 10:24:20 INFO: Database connection restored
2024-01-15 10:25:01 WARNING: High memory usage detected - 85%
"""
    }
    
    message = f"""
Please analyze these different file contents and provide insights:

1. CSV Data:
{files_content['csv_data']}

2. JSON Configuration:
{files_content['json_config']}

3. Log Excerpt:
{files_content['log_excerpt']}

For each file, provide:
- Summary of contents
- Potential issues or concerns
- Recommendations
"""
    
    response = await invoker.invoke(message=message, max_tokens=1500)
    
    if response['success']:
        print("File analysis results:")
        for i, result in enumerate(response['successful_responses'], 1):
            print(f"\n=== Analysis {i} from {result['provider']} ===")
            print(result['response'])
    else:
        print("File analysis failed")


def main():
    """Run all multimodal examples."""
    print("MultiAgent Failover Invoke - Multimodal Content Examples")
    print("=" * 60)
    
    try:
        asyncio.run(run_examples())
    except RuntimeError:
        # If already in async context
        asyncio.run(run_examples())


async def run_examples():
    """Run all async examples."""
    await example_1_image_analysis()
    await example_2_document_analysis()
    await example_3_code_review()
    await example_4_mixed_content_conversation()
    await example_5_file_content_analysis()
    
    print("\n" + "=" * 60)
    print("All multimodal examples completed!")
    print("\nKey Multimodal Features:")
    print("- Support for text + image content")
    print("- Document and code analysis")
    print("- Mixed content conversations with history")
    print("- File content analysis in various formats")
    print("- Automatic provider switching for multimodal content")


if __name__ == "__main__":
    main()
