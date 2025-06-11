from setuptools import setup, find_packages
import shutil
import os
from pathlib import Path

def copy_env_file():
    """Copy .env file to user's config directory during install."""
    try:
        source_env = Path(__file__).parent / ".env"
        config_dir = Path.home() / ".config" / "code-qna"
        config_dir.mkdir(parents=True, exist_ok=True)
        target_env = config_dir / ".env"
        
        if source_env.exists() and not target_env.exists():
            shutil.copy2(source_env, target_env)
            print(f"Copied .env file to {target_env}")
    except Exception as e:
        print(f"Warning: Could not copy .env file: {e}")

# Copy .env file during installation
copy_env_file()

setup(
    name="code-qna",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "google-genai>=0.1.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "tree-sitter>=0.20.0",
        "tree-sitter-language-pack>=0.5.0",
        "pygments>=2.0.0",
        "networkx>=2.5",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "code-qna=code_qna.cli:main",
        ],
    },
    python_requires=">=3.8",
)