import click
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
import time

from .context import ContextExtractor
from .ai import GeminiClient
from .utils.config import load_config, save_config, config_manager

console = Console()

# Load config early to use as defaults
_config = load_config()

@click.command()
@click.argument('question', required=False)
@click.option('--path', '-p', default='.', help='Path to the codebase to analyze')
@click.option('--model', '-m', default=_config.ai.model, help='Gemini model to use')
@click.option('--api-key', '-k', default=_config.ai.api_key, envvar='GEMINI_API_KEY', help='Gemini API key')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mode')
@click.option('--stream', '-s', default=_config.ai.stream_responses, is_flag=True, help='Stream the response')
@click.option('--max-context', '-c', default=_config.context.max_context_size, help='Maximum context size in characters')
@click.option('--debug', '-d', default=_config.ui.show_debug_info, is_flag=True, help='Show debug information')
@click.option('--show-progress', default=True, is_flag=True, help='Show detailed progress tracking')
@click.option('--config', is_flag=True, help='Show current configuration and exit')
def main(question, path, model, api_key, interactive, stream, max_context, debug, show_progress, config):
    """Code Q&A - Ask questions about your codebase using AI."""
    
    # Show config if requested
    if config:
        show_config()
        return
    
    # Validate path
    code_path = Path(path).resolve()
    if not code_path.exists():
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        sys.exit(1)
    
    if not code_path.is_dir():
        console.print(f"[red]Error: Path '{path}' is not a directory[/red]")
        sys.exit(1)
    
    # Load fresh config to respect any environment changes
    config_obj = load_config()
    
    # Use CLI args if provided, otherwise fall back to config
    final_model = model if model != _config.ai.model else config_obj.ai.model
    final_api_key = api_key if api_key else config_obj.ai.api_key
    final_max_context = max_context if max_context != _config.context.max_context_size else config_obj.context.max_context_size
    final_stream = stream if stream != _config.ai.stream_responses else config_obj.ai.stream_responses
    final_debug = debug if debug else config_obj.ui.show_debug_info
    
    # Initialize components
    try:
        # Prepare HTTP options if API version is specified
        http_options = None
        if config_obj.ai.api_version:
            from google.genai import types
            http_options = types.HttpOptions(api_version=config_obj.ai.api_version)
        
        gemini = GeminiClient(
            api_key=final_api_key, 
            model=final_model,
            temperature=config_obj.ai.temperature,
            max_output_tokens=config_obj.ai.max_output_tokens,
            enable_thinking=config_obj.ai.enable_thinking,
            thinking_budget=config_obj.ai.thinking_budget,
            include_thoughts=config_obj.ai.include_thoughts,
            vertexai=config_obj.ai.use_vertexai,
            project=config_obj.ai.vertexai_project,
            location=config_obj.ai.vertexai_location,
            http_options=http_options
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        if "Project ID" in str(e):
            console.print("\nFor Vertex AI, please set GOOGLE_CLOUD_PROJECT environment variable or configure in .env")
        else:
            console.print("\nPlease set GEMINI_API_KEY environment variable or use --api-key option")
        sys.exit(1)
    
    context_extractor = ContextExtractor(str(code_path), max_context_size=final_max_context, show_progress=show_progress)
    
    # Header
    console.print(Panel.fit(
        f"[bold blue]Code Q&A[/bold blue]\n"
        f"Analyzing: [green]{code_path}[/green]\n"
        f"Model: [yellow]{final_model}[/yellow]",
        border_style="blue"
    ))
    
    if interactive or not question:
        # Interactive mode
        console.print("\n[bold]Interactive mode[/bold] - Type 'exit' or 'quit' to leave\n")
        chat_session = gemini.create_chat_session()
        
        while True:
            try:
                question = console.input("[bold green]❯[/bold green] ")
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not question.strip():
                    continue
                
                # Process question
                process_question(
                    question, context_extractor, gemini, 
                    stream=final_stream, debug=final_debug, chat_session=chat_session,
                    show_progress=show_progress
                )
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
    else:
        # Single question mode
        process_question(question, context_extractor, gemini, stream=final_stream, debug=final_debug, show_progress=show_progress)
    
    console.print("\n[dim]Thank you for using Code Q&A![/dim]")


def process_question(question, context_extractor, gemini, stream=False, debug=False, chat_session=None, show_progress=True):
    """Process a single question."""
    console.print(f"\n[bold]Question:[/bold] {question}")
    
    # Extract context
    if not show_progress:
        # Use simple progress spinner when detailed progress is disabled
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=None)
            start_time = time.time()
            context = context_extractor.extract_relevant_context(question)
            extract_time = time.time() - start_time
            progress.update(task, description="Generating answer...")
    else:
        # Detailed progress tracking is handled inside context_extractor
        start_time = time.time()
        context = context_extractor.extract_relevant_context(question)
        extract_time = time.time() - start_time
    
    # Show debug info or progress summary
    if debug or show_progress:
        if not show_progress:  # Only show basic timing if detailed progress was disabled
            console.print(f"\n[dim]Extraction time: {extract_time:.2f}s[/dim]")
        
        console.print(f"[dim]Keywords: {', '.join(context['keywords'])}[/dim]")
        console.print(f"[dim]Files found: {context['context_files']['total_files']}[/dim]")
        console.print(f"[dim]Context size: {context['context_files']['total_size']:,} chars[/dim]")
        
        # Show token info if available
        if 'context_utilization' in context and context['context_utilization']['context_tokens'] > 0:
            tokens = context['context_utilization']['context_tokens']
            max_tokens = context['context_utilization']['max_context_tokens'] 
            utilization = context['context_utilization']['utilization_percent']
            console.print(f"[dim]Context tokens: {tokens:,} / {max_tokens:,} ({utilization:.1f}%)[/dim]")
        
        # Show files table in debug mode
        if debug and context['context_files']['files']:
            table = Table(title="Files in context")
            table.add_column("File", style="cyan")
            table.add_column("Matches", style="green")
            table.add_column("Partial", style="yellow")
            table.add_column("Size", style="blue")
            
            for file_info in context['context_files']['files']:
                table.add_row(
                    file_info['path'].replace(str(context_extractor.root_path) + '/', ''),
                    str(file_info.get('matches', 0)),
                    "Yes" if file_info.get('is_partial') else "No",
                    f"{len(file_info.get('content', '')):,} chars"
                )
            
            console.print(table)
    
    # Generate answer
    console.print("\n[bold]Answer:[/bold]")
    
    if stream:
        # Streaming response
        response_parts = []
        if chat_session:
            generator = chat_session.ask_stream(question, context)
        else:
            generator = gemini.generate_answer_stream(question, context)
        
        for chunk in generator:
            console.print(chunk, end="")
            response_parts.append(chunk)
        
        console.print()  # New line after streaming
    else:
        # Non-streaming response
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            
            if chat_session:
                answer = chat_session.ask(question, context)
            else:
                answer = gemini.generate_answer(question, context)
        
        # Display formatted answer
        console.print(Markdown(answer))


def show_config():
    """Display current configuration."""
    config = load_config()
    
    console.print("\n[bold blue]⚙️  Current Configuration[/bold blue]\n")
    
    # Create configuration tree
    tree = Tree("[bold]Configuration[/bold]")
    
    # AI Settings
    ai_branch = tree.add("🧠 AI Settings")
    ai_branch.add(f"Model: [yellow]{config.ai.model}[/yellow]")
    ai_branch.add(f"Temperature: [cyan]{config.ai.temperature}[/cyan]")
    ai_branch.add(f"Max Output Tokens: [green]{config.ai.max_output_tokens:,}[/green]")
    ai_branch.add(f"Stream Responses: [magenta]{config.ai.stream_responses}[/magenta]")
    ai_branch.add(f"API Key: [red]{'Set' if config.ai.api_key else 'Not Set'}[/red]")
    
    # Context Optimization Settings
    context_branch = tree.add("⚡ Context Optimization")
    context_branch.add(f"Max Context Size: [green]{config.context.max_context_size:,}[/green] chars")
    context_branch.add(f"Buffer Percentage: [cyan]{config.context.context_buffer_percentage:.1%}[/cyan]")
    context_branch.add(f"Partial Content Window: [cyan]{config.context.partial_content_window_size}[/cyan] lines")
    context_branch.add(f"Max Search Results: [yellow]{config.context.max_search_results}[/yellow]")
    context_branch.add(f"Max Related Files: [yellow]{config.context.max_related_files}[/yellow]")
    context_branch.add(f"Max Keywords Processed: [yellow]{config.context.max_keywords_to_process}[/yellow]")
    context_branch.add(f"Max File Size: [red]{config.context.max_file_size_mb}[/red] MB")
    context_branch.add(f"Cache Sample Size: [dim]{config.context.cache_validation_sample_size}[/dim]")
    context_branch.add(f"Token Estimation: [dim]{config.context.token_estimation_chars_per_token}[/dim] chars/token")
    
    # Search Settings
    search_branch = tree.add("🔍 Search Settings")
    search_branch.add(f"Max Results per Keyword: [cyan]{config.search.max_results_per_keyword}[/cyan]")
    search_branch.add(f"Context Lines: [cyan]{config.search.max_context_lines}[/cyan]")
    search_branch.add(f"Fuzzy Search: [magenta]{config.search.enable_fuzzy_search}[/magenta]")
    search_branch.add(f"Case Sensitive: [magenta]{config.search.case_sensitive}[/magenta]")
    search_branch.add(f"Search Timeout: [yellow]{config.search.search_timeout}s[/yellow]")
    search_branch.add(f"Semantic Analysis: [magenta]{config.search.enable_semantic_analysis}[/magenta]")
    
    # UI Settings
    ui_branch = tree.add("🎨 UI Settings")
    ui_branch.add(f"Show Debug Info: [magenta]{config.ui.show_debug_info}[/magenta]")
    ui_branch.add(f"Show File Paths: [magenta]{config.ui.show_file_paths}[/magenta]")
    ui_branch.add(f"Highlight Matches: [magenta]{config.ui.highlight_matches}[/magenta]")
    ui_branch.add(f"Max Display Files: [cyan]{config.ui.max_display_files}[/cyan]")
    ui_branch.add(f"Use Colors: [magenta]{config.ui.use_colors}[/magenta]")
    ui_branch.add(f"Show Line Numbers: [magenta]{config.ui.show_line_numbers}[/magenta]")
    
    console.print(tree)
    
    # Show config file locations
    console.print(f"\n[dim]Global .env: {config_manager.config_dir / '.env'}[/dim]")
    console.print(f"[dim]Project .env: {Path.cwd() / '.env'}[/dim]")
    console.print(f"\n[dim]Environment variables take precedence over .env files[/dim]")


def get_project_info(path):
    """Get basic project information."""
    project_files = {
        'package.json': ('Node.js', 'JavaScript'),
        'setup.py': ('Python', 'Python'),
        'pyproject.toml': ('Python', 'Python'),
        'Cargo.toml': ('Rust', 'Rust'),
        'go.mod': ('Go', 'Go'),
        'pom.xml': ('Java', 'Java'),
        'build.gradle': ('Java', 'Java'),
        'composer.json': ('PHP', 'PHP'),
        'Gemfile': ('Ruby', 'Ruby'),
    }
    
    for file, (ptype, lang) in project_files.items():
        if (path / file).exists():
            return {'type': ptype, 'language': lang}
    
    # Fallback: detect by file extensions
    try:
        common_files = list(path.rglob('*'))[:100]  # Check first 100 files
        if any(f.suffix == '.py' for f in common_files):
            return {'type': 'Python', 'language': 'Python'}
        elif any(f.suffix in ['.js', '.ts'] for f in common_files):
            return {'type': 'JavaScript', 'language': 'JavaScript'}
        elif any(f.suffix == '.java' for f in common_files):
            return {'type': 'Java', 'language': 'Java'}
        elif any(f.suffix == '.go' for f in common_files):
            return {'type': 'Go', 'language': 'Go'}
    except:
        pass
    
    return {'type': 'Unknown', 'language': 'Mixed'}


if __name__ == "__main__":
    main()