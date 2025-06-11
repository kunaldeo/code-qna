# 🤖 Code Q&A - Intelligent Codebase Question Answering

Ask questions about any codebase and get intelligent answers. No RAG, no vector databases - just smart context optimization that feeds the right code to AI models for accurate responses.

## 🌟 Why Code Q&A is Different

### 🚫 **No RAG Complexity**
- **No vector databases** - No need to maintain embeddings or indices
- **No preprocessing** - Ask questions on any codebase immediately  
- **No stale data** - Always analyzes your current code, not cached representations
- **No embedding costs** - Efficient search without expensive vector operations

### 🎯 **Optimized Context Utilization**
- **Intelligent file ranking** - Prioritizes source files, core modules, and relevant matches
- **Smart content extraction** - Preserves function boundaries and class structures
- **Context window maximization** - Fits 900K+ characters of the most relevant code
- **Progressive analysis** - Builds context incrementally for optimal AI understanding

### 🔍 **Right Context, Right Answer**
- **Multi-strategy search** - Combines text search, symbol lookup, and dependency analysis
- **Relevance scoring** - Advanced algorithm considers file importance, match quality, and code structure
- **Language-aware parsing** - Tree-sitter integration for precise Python and Java syntax understanding
- **Relationship mapping** - Understands how code components connect and interact

## 🛠️ How It Works

### Smart Context Pipeline
```
Question → Keywords → Multi-Search → Syntax Parse → Semantic Analysis → Relevance Score → Context Optimize → AI
```

1. **Extract Keywords** - Identify relevant terms from your question
2. **Multi-Strategy Search** - Find code using text search, symbol lookup, and file discovery
3. **Parse Syntax** - Use Tree-sitter for precise code structure understanding  
4. **Semantic Analysis** - Map dependencies, imports, and code relationships
5. **Score Relevance** - Rank files by importance, match quality, and code relationships
6. **Optimize Context** - Fit the most relevant code within AI token limits
7. **Generate Answer** - Feed optimized context to Gemini for accurate responses

Instead of vector embeddings, we use intelligent ranking:
- **File importance** - Source files > config files > documentation
- **Match quality** - Keyword density, function/class boundaries, import relationships
- **Code structure** - Preserve complete functions, avoid truncating mid-scope
- **Progressive inclusion** - Add files by relevance until context limit reached

## 🚀 Quick Start

### Installation
```bash
# Install with pipx (recommended)
pip install pipx
pipx install git+https://github.com/kunaldeo/code-qna.git

# Set your API credentials

# Option 1: Gemini Developer API
export GEMINI_API_KEY="your-api-key-here"

# Option 2: Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Basic Usage
```bash
# Ask a question about your codebase
code-qna "How does user authentication work?"

# Start interactive mode for multi-turn conversations
code-qna -i

# Analyze a specific directory with debug info
code-qna -p /path/to/project -d "What are the main API endpoints?"

# Enable debug mode for detailed analysis
code-qna -d "Explain the database schema"
```

### Command Line Options
```bash
code-qna [OPTIONS] [QUESTION]

Options:
  -p, --path PATH           Path to analyze (default: current directory)
  -i, --interactive        Start interactive mode with chat history
  -d, --debug             Show detailed debug information and analysis
  --config                Show current configuration and settings
  --help                  Show help message and exit
```

## ⚙️ Configuration

### Environment Variables
Copy `.env.sample` to `.env` and customize your settings:

```bash
# API Credentials (choose one option)

# Option 1: Gemini Developer API
GEMINI_API_KEY=your_api_key_here                    # Your Gemini API key from Google AI Studio
# or GOOGLE_API_KEY=your_api_key_here               # Alternative environment variable

# Option 2: Vertex AI
# GOOGLE_GENAI_USE_VERTEXAI=true                    # Use Vertex AI instead of Developer API
# GOOGLE_CLOUD_PROJECT=your-project-id              # Your Google Cloud Project ID
# GOOGLE_CLOUD_LOCATION=us-central1                 # Vertex AI location

# AI Model Settings
CODE_QNA_MODEL=gemini-2.0-flash-001                 # AI model to use (gemini-2.0-flash-001, gemini-1.5-pro, etc.)
CODE_QNA_TEMPERATURE=0.3                            # Model temperature (0.0-2.0): Lower = focused, Higher = creative
CODE_QNA_MAX_OUTPUT_TOKENS=4000                     # Maximum tokens in AI response (1-8192)
CODE_QNA_STREAM=false                               # Enable streaming responses (true/false)
# CODE_QNA_API_VERSION=v1                           # API version (optional: v1, v1alpha)

# Thinking Mode Settings (for Gemini 2.5 series models)
CODE_QNA_ENABLE_THINKING=false                         # Enable thinking mode for complex reasoning tasks (true/false)
CODE_QNA_THINKING_BUDGET=1024                          # Thinking token budget (128-32768 for Pro, 0-24576 for Flash)
CODE_QNA_INCLUDE_THOUGHTS=false                        # Include thought summaries in responses (true/false)

# Context Optimization
CODE_QNA_MAX_CONTEXT=900000                         # Maximum context size in characters (100000-2000000)
CONTEXT_BUFFER_PERCENTAGE=0.9                       # Buffer percentage for low priority files (0.1-1.0)
MAX_SEARCH_RESULTS=100                              # Maximum total search results to process (50-500)
MAX_RELATED_FILES=10                                # Maximum related files via imports (5-50)
MAX_FILE_SIZE_MB=1.0                                # Maximum file size in MB to analyze (0.1-10.0)

# Search Settings
SEARCH__MAX_RESULTS_PER_KEYWORD=50                  # Maximum results per keyword search (10-200)
SEARCH__MAX_CONTEXT_LINES=15                        # Lines of context around search matches (5-50)
SEARCH__ENABLE_FUZZY_SEARCH=true                    # Enable fuzzy/approximate matching (true/false)
SEARCH__CASE_SENSITIVE=false                        # Case sensitive search (true/false)
SEARCH__SEARCH_TIMEOUT=30                           # Search timeout in seconds (10-120)
ENABLE_SEMANTIC_ANALYSIS=false                      # Enable semantic code analysis (slower but more thorough)

# UI Settings
CODE_QNA_DEBUG=false                                # Show debug information (true/false)
UI__SHOW_FILE_PATHS=true                            # Show file paths in output (true/false)
UI__USE_COLORS=true                                 # Use colors in terminal output (true/false)
UI__MAX_DISPLAY_FILES=10                            # Maximum files to display in results (5-50)
```

### Configuration Files
Create `.env` in your project root or `~/.config/code-qna/.env` for global settings.

View current configuration: `code-qna --config`

## 🔧 Advanced Features

### Thinking Mode (Gemini 2.5 Series)
Enable advanced reasoning capabilities for complex code analysis tasks:

```bash
# Enable thinking mode for complex reasoning
export CODE_QNA_ENABLE_THINKING=true

# Set thinking budget for detailed analysis (higher = more thorough)
export CODE_QNA_THINKING_BUDGET=2048

# Include thought summaries in responses to see reasoning process
export CODE_QNA_INCLUDE_THOUGHTS=true

# Use a 2.5 series model that supports thinking
export CODE_QNA_MODEL=gemini-2.5-flash-preview-06-05
```

**Thinking Budget Guidelines:**
- **Gemini 2.5 Pro**: 128-32768 tokens (minimum 128, cannot be disabled)
- **Gemini 2.5 Flash**: 0-24576 tokens (set to 0 to disable thinking)
- **Higher budgets** = more detailed reasoning for complex tasks
- **Lower budgets** = faster responses for simpler questions

**Best Use Cases for Thinking Mode:**
- Complex architectural analysis and design questions
- Multi-step debugging and troubleshooting
- Performance optimization recommendations
- Security vulnerability analysis
- Advanced refactoring suggestions

### Project Type Detection
Automatically adapts to Python and Java projects by detecting manifest files (setup.py, pyproject.toml, pom.xml, build.gradle).

### Semantic Code Analysis
```bash
export CODE_QNA_ENABLE_SEMANTIC_ANALYSIS=true
```
- **Dependency mapping** - Build NetworkX graphs of code relationships
- **Function call tracing** - Trace execution paths across files  
- **Impact analysis** - Find code affected by changes

### Performance & Caching
- **Concurrent processing** - Multi-threaded search operations
- **Intelligent caching** - SHA256-keyed storage with file timestamp validation
- **Memory optimization** - Efficient processing of large codebases

## 💡 Example Use Cases

```bash
# Code Understanding
code-qna "How does user authentication work in this codebase?"
code-qna "What are all the REST API endpoints and what do they do?"

# Code Analysis & Reviews
code-qna "Are there any potential security vulnerabilities in the auth code?"
code-qna "What parts of the code might have performance bottlenecks?"

# Development & Debugging
code-qna "Show me all functions that process user payments"
code-qna "How does user data flow from the frontend to the database?"
```

## 📊 Example Output

<img width="1190" alt="in-action" src="https://github.com/user-attachments/assets/7d3cc393-b52b-4e78-acf2-8a88b6102856" />

---

**Built with ❤️ for developers who want to understand code better**
