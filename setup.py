import os
from setuptools import setup, find_packages

def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and hashes
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Remove hashes and everything after them
                    if ' --hash=' in line:
                        line = line.split(' --hash=')[0]
                    # Skip editable installs
                    if not line.startswith('-e'):
                        requirements.append(line)
            return requirements
    return []

def read_long_description():
    """Read long description from README"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="llm-invoker",
    version="0.1.1",
    author="Raed Jlassi",
    author_email="raed.jlassi@etudiant-enit.utm.tn", 
    description="llm-invoker - Robust LLM invocation with failover strategies",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/JlassiRAed/llm-invoker",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-invoke=llmInvoker.cli:main", 
        ],
    },
    include_package_data=True,
    keywords="llm, ai, failover, multiagent, openai, anthropic, langchain",
)