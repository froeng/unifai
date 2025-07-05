from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="unifai",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A unified interface for multiple AI providers with automatic fallback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/unifai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.30.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=1.0",
        ],
    },
    keywords="ai, openai, anthropic, llm, chat, completion, unified, interface",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/unifai/issues",
        "Source": "https://github.com/yourusername/unifai",
        "Documentation": "https://github.com/yourusername/unifai/blob/main/README.md",
    },
)