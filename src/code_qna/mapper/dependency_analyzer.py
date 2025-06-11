from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
import re
import ast
from tqdm import tqdm
from .semantic_graph import SemanticGraph, CodeNode, CodeEdge, NodeType, EdgeType
from ..parser import SyntaxParser
from ..utils.progress_tracker import code_map_progress

class DependencyAnalyzer:
    """Analyzes code dependencies and builds semantic relationships."""
    
    def __init__(self, root_path: str, parser: SyntaxParser):
        self.root_path = Path(root_path)
        self.parser = parser
        self.graph = SemanticGraph()
        
    def analyze_codebase(self, file_patterns: List[str] = None) -> SemanticGraph:
        """Analyze the entire codebase and build a dependency graph."""
        if file_patterns is None:
            file_patterns = ['**/*.py', '**/*.java', '**/*.js', '**/*.ts']
        
        # Find all relevant files
        files_to_analyze = []
        print("Discovering files...")
        for pattern in file_patterns:
            files_to_analyze.extend(self.root_path.glob(pattern))
        
        # Filter out common ignore patterns (enhanced)
        ignore_patterns = [
            '__pycache__', '.git', 'node_modules', 'dist', 'build',
            '.pytest_cache', '.mypy_cache', 'target', '.venv', 'venv',
            'site-packages', '.env', 'env', '.tox', 'eggs', '*.egg-info',
            '.DS_Store', 'Thumbs.db', '.idea', '.vscode', '.coverage'
        ]
        
        files_to_analyze = [
            f for f in files_to_analyze 
            if not any(ignore in str(f) for ignore in ignore_patterns)
        ]
        
        # Phase 1: Extract structure
        code_map_progress.start_phase(
            "Structure", 
            len(files_to_analyze),
            "Extracting functions, classes, and modules"
        )
        
        for file_path in files_to_analyze:
            self._analyze_file_structure(file_path)
            code_map_progress.update_progress(1)
        
        code_map_progress.complete_phase()
        
        # Phase 2: Analyze dependencies
        code_map_progress.start_phase(
            "Dependency Analysis",
            len(files_to_analyze), 
            "Analyzing imports and function calls"
        )
        
        for file_path in files_to_analyze:
            self._analyze_file_dependencies(file_path)
            code_map_progress.update_progress(1)
        
        code_map_progress.complete_phase()
        return self.graph
    
    def _analyze_file_structure(self, file_path: Path):
        """Analyze the structure of a single file and create nodes."""
        # Skip files that are too large or non-source files
        try:
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                return
        except (OSError, IOError):
            return
        
        file_id = f"file:{file_path}"
        
        # Add file node
        file_node = CodeNode(
            id=file_id,
            type=NodeType.FILE,
            name=file_path.name,
            file_path=str(file_path),
            metadata={'absolute_path': str(file_path.absolute())}
        )
        self.graph.add_node(file_node)
        
        # Extract functions and classes with error handling
        try:
            functions = self.parser.extract_functions(str(file_path))
            classes = self.parser.extract_classes(str(file_path))
        except Exception:
            # If parsing fails, continue with empty lists
            functions = []
            classes = []
        
        # Add function nodes
        for func in functions:
            func_id = f"func:{file_path}:{func['name']}:{func['start_line']}"
            func_node = CodeNode(
                id=func_id,
                type=NodeType.FUNCTION,
                name=func['name'],
                file_path=str(file_path),
                line_start=func['start_line'],
                line_end=func['end_line'],
                metadata={
                    'function_type': func.get('type', 'function'),
                    'code': func.get('code', '')
                }
            )
            self.graph.add_node(func_node)
            
            # Add edge from file to function
            self.graph.add_edge(CodeEdge(
                source=file_id,
                target=func_id,
                type=EdgeType.CONTAINS
            ))
        
        # Add class nodes
        for cls in classes:
            cls_id = f"class:{file_path}:{cls['name']}:{cls['start_line']}"
            cls_node = CodeNode(
                id=cls_id,
                type=NodeType.CLASS,
                name=cls['name'],
                file_path=str(file_path),
                line_start=cls['start_line'],
                line_end=cls['end_line'],
                metadata={'code': cls.get('code', '')}
            )
            self.graph.add_node(cls_node)
            
            # Add edge from file to class
            self.graph.add_edge(CodeEdge(
                source=file_id,
                target=cls_id,
                type=EdgeType.CONTAINS
            ))
    
    def _analyze_file_dependencies(self, file_path: Path):
        """Analyze dependencies within and from a file."""
        # Skip large files for performance
        try:
            if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                return
        except (OSError, IOError):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read in chunks for large files
                content = f.read(512 * 1024)  # Read max 512KB for dependency analysis
        except Exception:
            return
        
        file_id = f"file:{file_path}"
        
        # Analyze based on file type
        if file_path.suffix == '.py':
            self._analyze_python_dependencies(file_path, content, file_id)
        elif file_path.suffix == '.java':
            self._analyze_java_dependencies(file_path, content, file_id)
        elif file_path.suffix in ['.js', '.ts']:
            self._analyze_javascript_dependencies(file_path, content, file_id)
    
    def _analyze_python_dependencies(self, file_path: Path, content: str, file_id: str):
        """Analyze Python-specific dependencies."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return
        
        # Analyze imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._add_import_dependency(file_id, alias.name, file_path)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._add_import_dependency(file_id, node.module, file_path)
            
            elif isinstance(node, ast.Call):
                # Analyze function calls
                if isinstance(node.func, ast.Name):
                    self._add_function_call(file_id, node.func.id, node.lineno, file_path)
                elif isinstance(node.func, ast.Attribute):
                    # Method calls or module.function calls
                    call_name = self._get_full_call_name(node.func)
                    if call_name:
                        self._add_function_call(file_id, call_name, node.lineno, file_path)
    
    def _analyze_java_dependencies(self, file_path: Path, content: str, file_id: str):
        """Analyze Java-specific dependencies."""
        # Java import analysis
        import_pattern = r'import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*;'
        imports = re.findall(import_pattern, content)
        
        for import_name in imports:
            self._add_import_dependency(file_id, import_name, file_path)
        
        # Function/method call analysis
        call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        calls = re.findall(call_pattern, content)
        
        for call_name in calls:
            # Filter out Java keywords and common method names
            java_keywords = {'if', 'for', 'while', 'switch', 'catch', 'throw', 'return'}
            if call_name not in java_keywords and len(call_name) > 1:
                self._add_function_call(file_id, call_name, 0, file_path)
    
    def _analyze_javascript_dependencies(self, file_path: Path, content: str, file_id: str):
        """Analyze JavaScript/TypeScript dependencies."""
        # ES6 imports
        import_patterns = [
            r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
            r'import\s+["\']([^"\']+)["\']',
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'
        ]
        
        for pattern in import_patterns:
            imports = re.findall(pattern, content)
            for import_name in imports:
                self._add_import_dependency(file_id, import_name, file_path)
        
        # Function calls
        call_pattern = r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\('
        calls = re.findall(call_pattern, content)
        
        for call_name in calls:
            # Filter out JS keywords and common names
            js_keywords = {'if', 'for', 'while', 'switch', 'catch', 'throw', 'return', 'function'}
            if call_name not in js_keywords and len(call_name) > 1:
                self._add_function_call(file_id, call_name, 0, file_path)
    
    def _add_import_dependency(self, file_id: str, import_name: str, file_path: Path):
        """Add an import dependency."""
        # Try to resolve the import to an actual file
        resolved_path = self._resolve_import(import_name, file_path)
        
        if resolved_path and resolved_path.exists():
            target_file_id = f"file:{resolved_path}"
            
            # Create target file node if it doesn't exist
            if target_file_id not in self.graph.nodes:
                target_node = CodeNode(
                    id=target_file_id,
                    type=NodeType.FILE,
                    name=resolved_path.name,
                    file_path=str(resolved_path)
                )
                self.graph.add_node(target_node)
            
            # Add import edge
            self.graph.add_edge(CodeEdge(
                source=file_id,
                target=target_file_id,
                type=EdgeType.IMPORTS,
                metadata={'import_name': import_name}
            ))
        else:
            # External dependency - create a placeholder node
            import_id = f"import:{import_name}"
            if import_id not in self.graph.nodes:
                import_node = CodeNode(
                    id=import_id,
                    type=NodeType.IMPORT,
                    name=import_name,
                    metadata={'external': True}
                )
                self.graph.add_node(import_node)
            
            self.graph.add_edge(CodeEdge(
                source=file_id,
                target=import_id,
                type=EdgeType.IMPORTS,
                metadata={'import_name': import_name}
            ))
    
    def _add_function_call(self, file_id: str, call_name: str, line_number: int, file_path: Path):
        """Add a function call dependency."""
        # Look for functions with this name in the current file first
        file_functions = [
            node for node in self.graph.get_nodes_in_file(str(file_path))
            if node.type == NodeType.FUNCTION and node.name == call_name
        ]
        
        if file_functions:
            # Call to function in same file
            for func_node in file_functions:
                self.graph.add_edge(CodeEdge(
                    source=file_id,
                    target=func_node.id,
                    type=EdgeType.CALLS,
                    metadata={'line_number': line_number}
                ))
        else:
            # Look for functions in other files
            all_functions = self.graph.get_nodes_by_type(NodeType.FUNCTION)
            matching_functions = [f for f in all_functions if f.name == call_name]
            
            for func_node in matching_functions:
                self.graph.add_edge(CodeEdge(
                    source=file_id,
                    target=func_node.id,
                    type=EdgeType.CALLS,
                    metadata={'line_number': line_number, 'cross_file': True}
                ))
    
    def _resolve_import(self, import_name: str, from_file: Path) -> Optional[Path]:
        """Try to resolve an import to an actual file path."""
        # Handle relative imports
        if import_name.startswith('.'):
            current_dir = from_file.parent
            # Count leading dots
            dots = 0
            for char in import_name:
                if char == '.':
                    dots += 1
                else:
                    break
            
            # Navigate up directories
            target_dir = current_dir
            for _ in range(dots - 1):
                target_dir = target_dir.parent
            
            # Get module path after dots
            module_part = import_name[dots:]
            if module_part:
                potential_paths = [
                    target_dir / module_part.replace('.', '/') / "__init__.py",
                    target_dir / (module_part.replace('.', '/') + ".py")
                ]
            else:
                potential_paths = [target_dir / "__init__.py"]
            
            for path in potential_paths:
                if path.exists():
                    return path
        
        # Handle absolute imports
        else:
            # Try different common Python project structures
            potential_paths = [
                self.root_path / import_name.replace('.', '/') / "__init__.py",
                self.root_path / (import_name.replace('.', '/') + ".py"),
                self.root_path / "src" / import_name.replace('.', '/') / "__init__.py",
                self.root_path / "src" / (import_name.replace('.', '/') + ".py")
            ]
            
            for path in potential_paths:
                if path.exists():
                    return path
        
        return None
    
    def _get_full_call_name(self, node: ast.Attribute) -> str:
        """Get the full name of a method/attribute call."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts)) if parts else ""
    
    def find_data_flow_paths(self, start_function: str, end_function: str) -> List[List[str]]:
        """Find potential data flow paths between two functions."""
        start_nodes = self.graph.find_node(start_function, NodeType.FUNCTION)
        end_nodes = self.graph.find_node(end_function, NodeType.FUNCTION)
        
        paths = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                path = self.graph.get_shortest_path(start_node.id, end_node.id)
                if path:
                    paths.append(path)
        
        return paths
    
    def get_function_impact_analysis(self, function_name: str) -> Dict[str, Any]:
        """Analyze the impact of a function change."""
        function_nodes = self.graph.find_node(function_name, NodeType.FUNCTION)
        
        if not function_nodes:
            return {'error': 'Function not found'}
        
        impact = {
            'direct_callers': [],
            'indirect_callers': [],
            'affected_files': set(),
            'call_chains': []
        }
        
        for func_node in function_nodes:
            # Find direct callers
            callers = self.graph.get_dependents(func_node.id, [EdgeType.CALLS])
            impact['direct_callers'].extend(callers)
            
            # Find all related nodes within distance 3
            related = self.graph.find_related_nodes(func_node.id, max_distance=3)
            impact['indirect_callers'].extend(related.keys())
            
            # Find affected files
            for node_id in callers + list(related.keys()):
                if node_id in self.graph.nodes:
                    node = self.graph.nodes[node_id]
                    if node.file_path:
                        impact['affected_files'].add(node.file_path)
            
            # Find call chains
            chains = self.graph.find_call_chains(func_node.id, max_depth=4)
            impact['call_chains'].extend(chains)
        
        impact['affected_files'] = list(impact['affected_files'])
        return impact