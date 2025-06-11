"""Configuration management for Code Q&A using Pydantic Settings."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchConfig(BaseModel):
    """Search configuration."""
    max_results_per_keyword: int = Field(default=50, description="Maximum results per keyword search")
    max_context_lines: int = Field(default=15, description="Maximum context lines around matches")
    enable_fuzzy_search: bool = Field(default=True, description="Enable fuzzy searching")
    case_sensitive: bool = Field(default=False, description="Case sensitive search")
    search_timeout: int = Field(default=30, description="Search timeout in seconds")
    enable_semantic_analysis: bool = Field(default=False, description="Enable expensive semantic analysis")
    ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            ".git", "node_modules", "__pycache__", "*.pyc",
            "dist", "build", ".venv", "venv", ".idea", ".vscode"
        ],
        description="Patterns to ignore during search"
    )


class ContextConfig(BaseModel):
    """Context optimization configuration."""
    max_context_size: int = Field(default=900000, gt=0, description="Maximum context size in characters")
    context_buffer_percentage: float = Field(default=0.9, gt=0.0, le=1.0, description="Context buffer percentage for low priority files")
    partial_content_window_size: int = Field(default=10, gt=0, description="Lines of context around matches in partial content")
    max_search_results: int = Field(default=100, gt=0, description="Maximum total search results to process")
    max_results_per_keyword: int = Field(default=50, gt=0, description="Maximum results per keyword search")
    max_related_files: int = Field(default=10, gt=0, description="Maximum related files to include")
    max_keywords_to_process: int = Field(default=5, gt=0, description="Maximum keywords to process")
    max_file_size_mb: float = Field(default=1.0, gt=0, description="Maximum file size in MB to process")
    cache_validation_sample_size: int = Field(default=50, gt=0, description="Sample size for cache validation")
    token_estimation_chars_per_token: int = Field(default=4, gt=0, description="Characters per token for estimation")


class AIConfig(BaseModel):
    """AI model configuration."""
    model: str = Field(default="gemini-2.0-flash-001", description="AI model to use")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Model temperature")
    max_output_tokens: int = Field(default=4000, gt=0, description="Maximum output tokens")
    stream_responses: bool = Field(default=False, description="Enable streaming responses")
    api_key: Optional[str] = Field(default=None, description="API key for the AI service")
    enable_thinking: bool = Field(default=False, description="Enable thinking mode for 2.5 series models")
    thinking_budget: Optional[int] = Field(default=None, description="Thinking token budget (128-32768 for Pro, 0-24576 for Flash)")
    include_thoughts: bool = Field(default=False, description="Include thought summaries in responses")
    # Vertex AI specific settings
    use_vertexai: bool = Field(default=False, description="Use Vertex AI instead of Gemini Developer API")
    vertexai_project: Optional[str] = Field(default=None, description="Google Cloud Project ID for Vertex AI")
    vertexai_location: Optional[str] = Field(default="us-central1", description="Vertex AI location")
    api_version: Optional[str] = Field(default=None, description="API version (v1, v1alpha, etc.)")


class UIConfig(BaseModel):
    """UI configuration."""
    show_debug_info: bool = Field(default=False, description="Show debug information")
    show_file_paths: bool = Field(default=True, description="Show file paths in output")
    highlight_matches: bool = Field(default=True, description="Highlight search matches")
    max_display_files: int = Field(default=10, gt=0, description="Maximum files to display")
    use_colors: bool = Field(default=True, description="Use colors in output")
    show_line_numbers: bool = Field(default=True, description="Show line numbers")


class Config(BaseSettings):
    """Main configuration class using Pydantic Settings."""
    
    # Nested configurations
    search: SearchConfig = Field(default_factory=SearchConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    
    # Environment variable configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Direct environment mappings for common settings
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    code_qna_model: Optional[str] = Field(default=None, alias="CODE_QNA_MODEL")
    code_qna_temperature: Optional[float] = Field(default=None, alias="CODE_QNA_TEMPERATURE")
    code_qna_max_output_tokens: Optional[int] = Field(default=None, alias="CODE_QNA_MAX_OUTPUT_TOKENS")
    code_qna_max_context: Optional[int] = Field(default=None, alias="CODE_QNA_MAX_CONTEXT")
    code_qna_stream: Optional[bool] = Field(default=None, alias="CODE_QNA_STREAM")
    code_qna_debug: Optional[bool] = Field(default=None, alias="CODE_QNA_DEBUG")
    enable_semantic_analysis: Optional[bool] = Field(default=None, alias="ENABLE_SEMANTIC_ANALYSIS")
    no_color: Optional[bool] = Field(default=None, alias="NO_COLOR")
    
    # Thinking configuration environment mappings
    code_qna_enable_thinking: Optional[bool] = Field(default=None, alias="CODE_QNA_ENABLE_THINKING")
    code_qna_thinking_budget: Optional[int] = Field(default=None, alias="CODE_QNA_THINKING_BUDGET")
    code_qna_include_thoughts: Optional[bool] = Field(default=None, alias="CODE_QNA_INCLUDE_THOUGHTS")
    
    # Vertex AI environment mappings
    google_genai_use_vertexai: Optional[bool] = Field(default=None, alias="GOOGLE_GENAI_USE_VERTEXAI")
    google_cloud_project: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_LOCATION")
    code_qna_api_version: Optional[str] = Field(default=None, alias="CODE_QNA_API_VERSION")
    
    # Context optimization environment mappings
    context_buffer_percentage: Optional[float] = Field(default=None, alias="CONTEXT_BUFFER_PERCENTAGE")
    partial_content_window_size: Optional[int] = Field(default=None, alias="PARTIAL_CONTENT_WINDOW_SIZE")
    max_search_results: Optional[int] = Field(default=None, alias="MAX_SEARCH_RESULTS")
    max_related_files: Optional[int] = Field(default=None, alias="MAX_RELATED_FILES")
    max_keywords_to_process: Optional[int] = Field(default=None, alias="MAX_KEYWORDS_TO_PROCESS")
    max_file_size_mb: Optional[float] = Field(default=None, alias="MAX_FILE_SIZE_MB")
    cache_validation_sample_size: Optional[int] = Field(default=None, alias="CACHE_VALIDATION_SAMPLE_SIZE")
    token_estimation_chars_per_token: Optional[int] = Field(default=None, alias="TOKEN_ESTIMATION_CHARS_PER_TOKEN")
    
    def model_post_init(self, __context) -> None:
        """Apply environment variable overrides after model initialization."""
        # Apply API key (prefer GEMINI_API_KEY over GOOGLE_API_KEY)
        if self.gemini_api_key:
            self.ai.api_key = self.gemini_api_key
        elif self.google_api_key and not self.gemini_api_key:
            self.ai.api_key = self.google_api_key
        
        # Apply AI settings
        if self.code_qna_model:
            self.ai.model = self.code_qna_model
        if self.code_qna_temperature is not None:
            self.ai.temperature = self.code_qna_temperature
        if self.code_qna_max_output_tokens:
            self.ai.max_output_tokens = self.code_qna_max_output_tokens
        if self.code_qna_stream is not None:
            self.ai.stream_responses = self.code_qna_stream
        if self.code_qna_enable_thinking is not None:
            self.ai.enable_thinking = self.code_qna_enable_thinking
        if self.code_qna_thinking_budget is not None:
            self.ai.thinking_budget = self.code_qna_thinking_budget
        if self.code_qna_include_thoughts is not None:
            self.ai.include_thoughts = self.code_qna_include_thoughts
        
        # Apply Vertex AI settings
        if self.google_genai_use_vertexai is not None:
            self.ai.use_vertexai = self.google_genai_use_vertexai
        if self.google_cloud_project:
            self.ai.vertexai_project = self.google_cloud_project
        if self.google_cloud_location:
            self.ai.vertexai_location = self.google_cloud_location
        if self.code_qna_api_version:
            self.ai.api_version = self.code_qna_api_version
        
        # Apply context settings
        if self.code_qna_max_context:
            self.context.max_context_size = self.code_qna_max_context
        if self.context_buffer_percentage is not None:
            self.context.context_buffer_percentage = self.context_buffer_percentage
        if self.partial_content_window_size is not None:
            self.context.partial_content_window_size = self.partial_content_window_size
        if self.max_search_results is not None:
            self.context.max_search_results = self.max_search_results
        if self.max_related_files is not None:
            self.context.max_related_files = self.max_related_files
        if self.max_keywords_to_process is not None:
            self.context.max_keywords_to_process = self.max_keywords_to_process
        if self.max_file_size_mb is not None:
            self.context.max_file_size_mb = self.max_file_size_mb
        if self.cache_validation_sample_size is not None:
            self.context.cache_validation_sample_size = self.cache_validation_sample_size
        if self.token_estimation_chars_per_token is not None:
            self.context.token_estimation_chars_per_token = self.token_estimation_chars_per_token
        
        # Apply UI settings
        if self.code_qna_debug is not None:
            self.ui.show_debug_info = self.code_qna_debug
        if self.no_color is not None:
            self.ui.use_colors = not self.no_color
        
        # Apply search settings
        if self.enable_semantic_analysis is not None:
            self.search.enable_semantic_analysis = self.enable_semantic_analysis


class ConfigManager:
    """Manage configuration loading and saving."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "code-qna"
        self.config_file = self.config_dir / "config.yaml"
        self.project_config_file = Path.cwd() / ".code-qna.yaml"
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Config:
        """Load configuration using Pydantic Settings."""
        # Pydantic Settings automatically loads .env files
        return Config()
    
    def create_sample_env(self, project_specific: bool = False):
        """Create a sample .env file."""
        sample_env_content = """# Code Q&A Configuration
# AI Model Settings
# For Gemini Developer API:
GEMINI_API_KEY=your_api_key_here
# or GOOGLE_API_KEY=your_api_key_here

# For Vertex AI:
# GOOGLE_GENAI_USE_VERTEXAI=true
# GOOGLE_CLOUD_PROJECT=your-project-id
# GOOGLE_CLOUD_LOCATION=us-central1

# Model Configuration
CODE_QNA_MODEL=gemini-2.0-flash-001
CODE_QNA_TEMPERATURE=0.3
CODE_QNA_MAX_OUTPUT_TOKENS=4000
CODE_QNA_STREAM=false
# CODE_QNA_API_VERSION=v1  # Optional: API version (v1, v1alpha)

# Thinking Mode Settings (for Gemini 2.5 series models)
CODE_QNA_ENABLE_THINKING=false
CODE_QNA_THINKING_BUDGET=1024
CODE_QNA_INCLUDE_THOUGHTS=false

# Context Optimization Settings
CODE_QNA_MAX_CONTEXT=900000
CONTEXT_BUFFER_PERCENTAGE=0.9
PARTIAL_CONTENT_WINDOW_SIZE=10
MAX_SEARCH_RESULTS=100
MAX_RELATED_FILES=10
MAX_KEYWORDS_TO_PROCESS=5
MAX_FILE_SIZE_MB=1.0
CACHE_VALIDATION_SAMPLE_SIZE=50
TOKEN_ESTIMATION_CHARS_PER_TOKEN=4

# Or use nested format for context settings:
# CONTEXT__MAX_CONTEXT_SIZE=900000
# CONTEXT__CONTEXT_BUFFER_PERCENTAGE=0.9
# CONTEXT__PARTIAL_CONTENT_WINDOW_SIZE=10
# CONTEXT__MAX_SEARCH_RESULTS=100
# CONTEXT__MAX_RESULTS_PER_KEYWORD=50
# CONTEXT__MAX_RELATED_FILES=10
# CONTEXT__MAX_KEYWORDS_TO_PROCESS=5
# CONTEXT__MAX_FILE_SIZE_MB=1.0
# CONTEXT__CACHE_VALIDATION_SAMPLE_SIZE=50
# CONTEXT__TOKEN_ESTIMATION_CHARS_PER_TOKEN=4

# Search Settings
SEARCH__MAX_RESULTS_PER_KEYWORD=50
SEARCH__MAX_CONTEXT_LINES=15
SEARCH__ENABLE_FUZZY_SEARCH=true
SEARCH__CASE_SENSITIVE=false
SEARCH__SEARCH_TIMEOUT=30
ENABLE_SEMANTIC_ANALYSIS=false

# UI Settings
CODE_QNA_DEBUG=false
UI__SHOW_FILE_PATHS=true
UI__HIGHLIGHT_MATCHES=true
UI__MAX_DISPLAY_FILES=10
UI__USE_COLORS=true
UI__SHOW_LINE_NUMBERS=true
NO_COLOR=false
"""
        
        target_file = Path.cwd() / ".env" if project_specific else self.config_dir / ".env"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_file, 'w') as f:
            f.write(sample_env_content)
        
        print(f"Sample .env file created at: {target_file}")


# Global config manager instance
config_manager = ConfigManager()


def load_config() -> Config:
    """Load configuration using the global config manager."""
    return config_manager.load_config()


def create_sample_env(project_specific: bool = False):
    """Create a sample .env file."""
    config_manager.create_sample_env(project_specific)


def save_config(config: Config, global_config: bool = True):
    """Save configuration - placeholder for backward compatibility."""
    # With Pydantic Settings, configuration is primarily managed through .env files
    # This function is kept for backward compatibility but doesn't do anything
    # since Pydantic handles configuration automatically
    pass