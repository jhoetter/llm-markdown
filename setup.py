from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-markdown",
    version="0.3.0",
    author="Johannes Hötter",
    author_email="johannes.hoetter@kern.ai",
    description="Turn Python functions into typed LLM calls using docstrings as prompts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jhoetter/llm-markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pydantic",
        "requests",
    ],
    extras_require={
        "openai": ["openai"],
        "langfuse": ["langfuse"],
        "test": ["pytest", "pytest-asyncio"],
        "all": ["openai", "langfuse", "python-dotenv"],
    },
)
