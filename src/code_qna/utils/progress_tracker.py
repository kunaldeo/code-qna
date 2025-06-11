from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, track
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
import time
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from .config import load_config

def estimate_tokens(text: str, config=None) -> int:
    """
    Estimate token count using a simple approximation.
    Uses configurable characters per token ratio.
    """
    if not text:
        return 0
    
    if config is None:
        config = load_config()
    
    # Remove excessive whitespace and normalize
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    # Use configurable approximation
    chars_per_token = config.context.token_estimation_chars_per_token
    return max(1, len(normalized) // chars_per_token)


def estimate_max_tokens(max_chars: int, config=None) -> int:
    """Estimate max tokens from max character limit."""
    if config is None:
        config = load_config()
    
    chars_per_token = config.context.token_estimation_chars_per_token
    return max(1, max_chars // chars_per_token)


@dataclass
class ProgressStep:
    """Track individual progress steps."""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    substeps: List['ProgressStep'] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None
    
    def start(self):
        """Mark step as started."""
        self.status = "running"
        self.start_time = time.time()
    
    def complete(self, **details):
        """Mark step as completed."""
        self.status = "completed"
        self.end_time = time.time()
        self.details.update(details)
    
    def fail(self, error: str):
        """Mark step as failed."""
        self.status = "failed"
        self.end_time = time.time()
        self.details["error"] = error


class ProgressTracker:
    """Comprehensive progress tracking for the Q&A system."""
    
    def __init__(self, console: Optional[Console] = None, show_details: bool = True):
        self.console = console or Console()
        self.show_details = show_details
        self.steps: List[ProgressStep] = []
        self.current_step: Optional[ProgressStep] = None
        self.context_stats = {
            "total_files_found": 0,
            "files_included": 0,
            "context_tokens": 0,
            "max_context_tokens": 0,
            "context_chars": 0,
            "utilization_percent": 0.0,
            "search_tools_used": [],
            "parsing_tools_used": []
        }
        self.live_display: Optional[Live] = None
    
    def add_step(self, name: str, **details) -> ProgressStep:
        """Add a new progress step."""
        step = ProgressStep(name=name, details=details)
        self.steps.append(step)
        return step
    
    def start_step(self, name: str, **details) -> ProgressStep:
        """Start a new progress step."""
        if self.current_step and self.current_step.status == "running":
            self.current_step.complete()
        
        step = self.add_step(name, **details)
        step.start()
        self.current_step = step
        
        if self.show_details:
            self._update_live_display()
        
        return step
    
    def complete_current_step(self, **details):
        """Complete the current step."""
        if self.current_step:
            self.current_step.complete(**details)
            if self.show_details:
                self._update_live_display()
    
    def fail_current_step(self, error: str):
        """Fail the current step."""
        if self.current_step:
            self.current_step.fail(error)
            if self.show_details:
                self._update_live_display()
    
    def update_context_stats(self, **stats):
        """Update context utilization statistics."""
        self.context_stats.update(stats)
        if self.context_stats["max_context_tokens"] > 0:
            self.context_stats["utilization_percent"] = (
                self.context_stats["context_tokens"] / self.context_stats["max_context_tokens"] * 100
            )
    
    def show_summary(self):
        """Show final progress summary."""
        if self.live_display:
            self.live_display.stop()
        
        # Create summary panel
        summary_table = Table(title="🔍 Search & Analysis Summary")
        summary_table.add_column("Phase", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Duration", style="yellow")
        summary_table.add_column("Details", style="dim")
        
        for step in self.steps:
            status_icon = {
                "completed": "✅",
                "failed": "❌", 
                "running": "🔄",
                "pending": "⏳"
            }.get(step.status, "❓")
            
            duration_text = f"{step.duration:.2f}s" if step.duration else "-"
            
            details_parts = []
            
            # Keywords extraction phase
            if "keywords_found" in step.details:
                details_parts.append(f"Keywords: {step.details['keywords_found']}")
            
            # Search phase
            if "files_found" in step.details:
                details_parts.append(f"Files: {step.details['files_found']}")
            if "matches" in step.details:
                details_parts.append(f"Matches: {step.details['matches']}")
            if "tool" in step.details:
                details_parts.append(f"Tool: {step.details['tool']}")
            
            # Syntax analysis phase
            if "functions_found" in step.details:
                details_parts.append(f"Functions: {step.details['functions_found']}")
            if "classes_found" in step.details:
                details_parts.append(f"Classes: {step.details['classes_found']}")
            if "files_analyzed" in step.details:
                details_parts.append(f"Files: {step.details['files_analyzed']}")
            
            # Related files phase
            if "related_files" in step.details:
                details_parts.append(f"Related: {step.details['related_files']}")
            
            # Optimization phase
            if "final_size" in step.details:
                details_parts.append(f"Size: {step.details['final_size']:,} chars")
            if "files_included" in step.details:
                details_parts.append(f"Included: {step.details['files_included']}")
            if "reduction_percent" in step.details:
                reduction = step.details['reduction_percent']
                if reduction > 0:
                    details_parts.append(f"Reduced: {reduction:.1f}%")
            
            # General size field
            if "size" in step.details and "final_size" not in step.details:
                details_parts.append(f"Size: {step.details['size']:,} chars")
            
            details_text = " | ".join(details_parts) if details_parts else "-"
            
            summary_table.add_row(
                step.name,
                f"{status_icon} {step.status.title()}",
                duration_text,
                details_text
            )
        
        self.console.print(summary_table)
        
        # Show context utilization
        if self.context_stats["max_context_tokens"] > 0:
            self._show_context_utilization()
    
    def _show_context_utilization(self):
        """Show context utilization details."""
        util_panel = Panel(
            self._create_context_display(),
            title="📊 Context Utilization",
            border_style="blue"
        )
        self.console.print(util_panel)
    
    def _create_context_display(self) -> str:
        """Create context utilization display."""
        stats = self.context_stats
        
        # Calculate utilization bar
        utilization = stats["utilization_percent"]
        bar_length = 40
        filled_length = int(bar_length * utilization / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Color code based on utilization
        if utilization < 50:
            util_color = "green"
        elif utilization < 80:
            util_color = "yellow"
        else:
            util_color = "red"
        
        display_parts = [
            f"Context Tokens: {stats['context_tokens']:,} / {stats['max_context_tokens']:,} tokens",
            f"Context Size: {stats['context_chars']:,} chars",
            f"Utilization: [{util_color}]{utilization:.1f}%[/{util_color}] [{util_color}]{bar}[/{util_color}]",
            f"Files Processed: {stats['total_files_found']} found → {stats['files_included']} included",
        ]
        
        if stats["search_tools_used"]:
            display_parts.append(f"Search Tools: {', '.join(stats['search_tools_used'])}")
        
        if stats["parsing_tools_used"]:
            display_parts.append(f"Parsing Tools: {', '.join(stats['parsing_tools_used'])}")
        
        return "\n".join(display_parts)
    
    def _update_live_display(self):
        """Update the live progress display."""
        if not self.live_display:
            return
        
        # Create current progress display
        progress_text = Text()
        
        if self.current_step:
            if self.current_step.status == "running":
                progress_text.append("🔄 ", style="cyan")
            elif self.current_step.status == "completed":
                progress_text.append("✅ ", style="green")
            elif self.current_step.status == "failed":
                progress_text.append("❌ ", style="red")
            
            progress_text.append(f"{self.current_step.name}", style="bold")
            
            if self.current_step.duration:
                progress_text.append(f" ({self.current_step.duration:.1f}s)", style="dim")
        
        # Show context stats if available
        if self.context_stats["context_tokens"] > 0:
            progress_text.append("\n")
            progress_text.append(f"Context: {self.context_stats['context_tokens']:,} tokens ", style="blue")
            progress_text.append(f"({self.context_stats['utilization_percent']:.1f}%)", style="dim")
        
        self.live_display.update(Panel(progress_text, title="Progress", border_style="blue"))
    
    def start_live_tracking(self):
        """Start live progress tracking display."""
        if self.show_details:
            self.live_display = Live(console=self.console, refresh_per_second=2, transient=True)
            self.live_display.start()
    
    def stop_live_tracking(self):
        """Stop live progress tracking display."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
    
    def reset(self):
        """Reset progress tracker for a new operation."""
        self.steps.clear()
        self.current_step = None
        self.context_stats = {
            "total_files_found": 0,
            "files_included": 0,
            "context_tokens": 0,
            "max_context_tokens": 0,
            "context_chars": 0,
            "utilization_percent": 0.0,
            "search_tools_used": [],
            "parsing_tools_used": []
        }

    def track_search_phase(self, keywords: List[str]) -> ProgressStep:
        """Track search phase progress."""
        return self.start_step(
            "🔍 Searching codebase",
            phase="search",
            keywords=keywords,
            keyword_count=len(keywords)
        )
    
    def track_syntax_analysis(self, files_count: int) -> ProgressStep:
        """Track syntax analysis phase."""
        return self.start_step(
            "🌳 Analyzing syntax trees",
            phase="syntax",
            files_to_analyze=files_count
        )
    
    def track_context_building(self, total_files: int) -> ProgressStep:
        """Track context building phase."""
        return self.start_step(
            "📝 Building context",
            phase="context",
            total_files=total_files
        )
    
    def track_optimization(self, context_size: int, max_size: int) -> ProgressStep:
        """Track context optimization phase."""
        return self.start_step(
            "⚡ Optimizing context size",
            phase="optimization",
            initial_size=context_size,
            max_size=max_size
        )
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing summary for all completed steps."""
        timing = {}
        total_time = 0
        
        for step in self.steps:
            if step.duration:
                timing[step.name] = step.duration
                total_time += step.duration
        
        timing["total"] = total_time
        return timing

# Enhanced progress tracker for code mapping operations
class CodeMapProgressTracker:
    """Specialized progress tracker for code mapping operations."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.current_operation: Optional[str] = None
        self.progress: Optional[Progress] = None
        self.current_task_id: Optional[int] = None

    def start_code_mapping(self, total_files: int):
        """Start code mapping operation tracking."""
        self.current_operation = "code_mapping"
        self.operations["code_mapping"] = {
            "total_files": total_files,
            "processed_files": 0,
            "start_time": time.time(),
            "phases": []
        }

        self.console.print("🗺️  [bold blue]Starting Code Map Generation[/bold blue]")
        self.console.print(f"📁 Found {total_files} files to analyze")

    def start_phase(self, phase_name: str, total_items: int, description: str = ""):
        """Start a new phase of the mapping process."""
        if not self.current_operation:
            return

        phase_info = {
            "name": phase_name,
            "description": description,
            "total_items": total_items,
            "processed_items": 0,
            "start_time": time.time(),
            "end_time": None
        }

        self.operations[self.current_operation]["phases"].append(phase_info)

        # Close previous progress bar if exists
        if self.progress:
            self.progress.stop()

        # Create Rich progress bar with separate console to avoid conflicts
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "({task.completed}/{task.total})",
            TimeElapsedColumn(),
            console=Console(file=sys.stderr),  # Use stderr to avoid conflicts
            transient=False
        )
        
        self.progress.start()
        task_description = f"{phase_name}"
        if description:
            task_description += f" - {description}"
        
        self.current_task_id = self.progress.add_task(task_description, total=total_items)
    
    def update_progress(self, items_processed: int = 1, postfix: str = ""):
        """Update progress for current phase."""
        if self.progress and self.current_task_id is not None:
            self.progress.update(self.current_task_id, advance=items_processed)
        
        # Update operation data
        if self.current_operation and self.operations[self.current_operation]["phases"]:
            current_phase = self.operations[self.current_operation]["phases"][-1]
            current_phase["processed_items"] += items_processed
    
    def complete_phase(self):
        """Complete the current phase."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.current_task_id = None
        
        if self.current_operation and self.operations[self.current_operation]["phases"]:
            current_phase = self.operations[self.current_operation]["phases"][-1]
            current_phase["end_time"] = time.time()
            
            # Show completion message
            duration = current_phase["end_time"] - current_phase["start_time"]
            self.console.print(f"✅ Completed: {current_phase['name']} ({duration:.1f}s)", style="green")
    
    def complete_code_mapping(self, graph_stats: Dict[str, Any]):
        """Complete code mapping operation."""
        if not self.current_operation:
            return
        
        self.complete_phase()  # Complete any remaining phase
        
        operation = self.operations[self.current_operation]
        operation["end_time"] = time.time()
        operation["graph_stats"] = graph_stats
        
        total_time = operation["end_time"] - operation["start_time"]
        
        # Show completion summary
        self.console.print("\n🎉 [bold green]Code Map Generation Complete![/bold green]")
        self.console.print(Panel(
            self._create_summary_text(operation),
            title="📈 Code Mapping Summary",
            border_style="green"
        ))
        
        self.current_operation = None
    
    def _create_summary_text(self, operation: Dict[str, Any]) -> str:
        """Create summary text for completed operation."""
        total_time = operation["end_time"] - operation["start_time"]
        stats = operation.get("graph_stats", {})
        
        summary_lines = [
            f"⏱️  Total Time: {total_time:.2f} seconds",
            f"📁 Files Analyzed: {operation['total_files']}",
            f"🔗 Graph Nodes: {stats.get('total_nodes', 'N/A')}",
            f"➡️  Graph Edges: {stats.get('total_edges', 'N/A')}",
            f"🏗️  Components: {stats.get('connected_components', 'N/A')}"
        ]
        
        # Add phase timing
        if operation["phases"]:
            summary_lines.append("\n📊 Phase Breakdown:")
            for phase in operation["phases"]:
                if phase.get("end_time"):
                    phase_time = phase["end_time"] - phase["start_time"]
                    summary_lines.append(f"  • {phase['name']}: {phase_time:.2f}s")
        
        # Add node type breakdown if available
        if "nodes_by_type" in stats:
            summary_lines.append("\n🏷️  Node Types:")
            for node_type, count in stats["nodes_by_type"].items():
                if count > 0:
                    summary_lines.append(f"  • {node_type}: {count}")
        
        return "\n".join(summary_lines)

# Global code map progress tracker
code_map_progress = CodeMapProgressTracker()