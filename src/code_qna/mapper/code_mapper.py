from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from .semantic_graph import SemanticGraph, NodeType, EdgeType
from .dependency_analyzer import DependencyAnalyzer
from ..parser import SyntaxParser
from ..cache.cache_manager import cache_manager
from ..utils.progress_tracker import code_map_progress
from ..utils.config import load_config

class CodeMapper:
    """Main class for building and maintaining a comprehensive code map."""
    
    def __init__(self, root_path: str, cache_mgr=None):
        self.root_path = Path(root_path)
        self.cache_manager = cache_mgr or cache_manager
        self.graph: Optional[SemanticGraph] = None
        self.config = load_config()
        self.last_analysis_time: Optional[float] = None
        
        # Detect active languages
        self.active_languages = self._detect_active_languages()
        self.parser = SyntaxParser(active_languages=self.active_languages)
        self.dependency_analyzer = DependencyAnalyzer(str(root_path), self.parser)
        
    def build_code_map(self, force_rebuild: bool = False) -> SemanticGraph:
        """Build or load the complete code map for the repository."""
        cache_key = f"code_map_{hashlib.sha256(str(self.root_path).encode()).hexdigest()[:16]}"
        
        # Try to load from cache first
        if not force_rebuild and self.cache_manager:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data and self._is_cache_valid(cached_data):
                self.graph = SemanticGraph()
                self.graph.import_from_json(cached_data['graph_path'])
                self.last_analysis_time = cached_data['timestamp']
                return self.graph
        
        # Build new code map with progress tracking
        start_time = time.time()
        
        # Use optimized analysis with progress tracking
        self.graph = self._build_optimized_graph()
        self.last_analysis_time = time.time()
        
        # Cache the results
        if self.cache_manager:
            cache_path = self.cache_manager.cache_dir / f"graph_{hashlib.sha256(str(self.root_path).encode()).hexdigest()[:16]}.json"
            self.graph.export_to_json(str(cache_path))
            
            cache_data = {
                'timestamp': self.last_analysis_time,
                'graph_path': str(cache_path),
                'stats': self.graph.get_statistics()
            }
            self.cache_manager.set(cache_key, cache_data)
        
        analysis_time = time.time() - start_time
        
        # Complete progress tracking with summary
        code_map_progress.complete_code_mapping(self.graph.get_statistics())
        
        return self.graph
    
    def semantic_search(self, query: str, context_type: str = "all") -> List[Dict[str, Any]]:
        """Perform semantic search using the code map."""
        if not self.graph:
            self.build_code_map()
        
        results = []
        query_lower = query.lower()
        
        # Direct name matches with high relevance
        for node in self.graph.nodes.values():
            if query_lower in node.name.lower():
                relevance = self._calculate_relevance(node.name, query)
                results.append({
                    'node': node,
                    'relevance': relevance,
                    'match_type': 'name',
                    'related_nodes': self._get_context_nodes(node.id, context_type)
                })
        
        # Code content matches
        for node in self.graph.nodes.values():
            if node.metadata and 'code' in node.metadata:
                code = node.metadata['code'].lower()
                if query_lower in code:
                    relevance = self._calculate_code_relevance(code, query)
                    results.append({
                        'node': node,
                        'relevance': relevance,
                        'match_type': 'code',
                        'related_nodes': self._get_context_nodes(node.id, context_type)
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Remove duplicates
        seen_nodes = set()
        unique_results = []
        for result in results:
            if result['node'].id not in seen_nodes:
                seen_nodes.add(result['node'].id)
                unique_results.append(result)
        
        return unique_results[:50]  # Return top 50 results
    
    def find_related_concepts(self, concept: str, max_distance: int = 2) -> List[Dict[str, Any]]:
        """Find concepts related to the given concept through the graph."""
        if not self.graph:
            self.build_code_map()
        
        related_concepts = []
        
        # Find nodes matching the concept
        matching_nodes = []
        for node in self.graph.nodes.values():
            if concept.lower() in node.name.lower():
                matching_nodes.append(node)
        
        # For each matching node, find related nodes
        for node in matching_nodes:
            related_node_ids = self.graph.find_related_nodes(node.id, max_distance)
            
            for related_id, distance in related_node_ids.items():
                related_node = self.graph.nodes[related_id]
                
                # Calculate relationship strength
                relationship_strength = 1.0 / (distance + 1)
                
                related_concepts.append({
                    'concept': related_node.name,
                    'node': related_node,
                    'distance': distance,
                    'strength': relationship_strength,
                    'relationship_type': self._get_relationship_type(node.id, related_id)
                })
        
        # Sort by relationship strength
        related_concepts.sort(key=lambda x: x['strength'], reverse=True)
        
        return related_concepts[:20]  # Return top 20 related concepts
    
    def get_context_for_question(self, question: str) -> Dict[str, Any]:
        """Get comprehensive context for answering a question."""
        if not self.graph:
            self.build_code_map()
        
        # Extract key terms from the question
        key_terms = self._extract_key_terms(question)
        
        context = {
            'question': question,
            'key_terms': key_terms,
            'relevant_nodes': [],
            'related_files': set(),
            'function_chains': [],
            'data_flows': [],
            'impact_analysis': {}
        }
        
        # Find relevant nodes for each key term
        for term in key_terms:
            search_results = self.semantic_search(term, context_type="comprehensive")
            context['relevant_nodes'].extend(search_results[:10])  # Top 10 per term
            
            # Add related files
            for result in search_results[:5]:
                if result['node'].file_path:
                    context['related_files'].add(result['node'].file_path)
        
        # Find function call chains if question involves functions
        function_terms = self._extract_function_terms(question)
        for term in function_terms:
            func_nodes = self.graph.find_node(term, NodeType.FUNCTION)
            for func_node in func_nodes:
                chains = self.graph.find_call_chains(func_node.id, max_depth=3)
                context['function_chains'].extend(chains)
        
        # Analyze potential data flows
        if len(function_terms) >= 2:
            for i, start_func in enumerate(function_terms[:-1]):
                for end_func in function_terms[i+1:]:
                    flows = self.dependency_analyzer.find_data_flow_paths(start_func, end_func)
                    context['data_flows'].extend(flows)
        
        # Impact analysis for modification questions
        if any(word in question.lower() for word in ['change', 'modify', 'update', 'fix']):
            for term in function_terms:
                impact = self.dependency_analyzer.get_function_impact_analysis(term)
                context['impact_analysis'][term] = impact
        
        context['related_files'] = list(context['related_files'])
        
        return context
    
    def _detect_active_languages(self) -> Set[str]:
        """Detect which programming languages are used in the repository."""
        active_languages = set()
        
        # File extension to language mapping
        ext_to_lang = {
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript"
        }
        
        file_count = 0
        for file_path in self.root_path.rglob("*"):
            if file_path.is_file() and file_count < 1000:
                suffix = file_path.suffix.lower()
                if suffix in ext_to_lang:
                    active_languages.add(ext_to_lang[suffix])
                    file_count += 1
        
        return active_languages or {"python"}
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid (optimized version)."""
        if not cached_data.get('timestamp'):
            return False
        
        cache_time = cached_data['timestamp']
        
        # Quick check: if cache is very recent (< 5 minutes), assume valid
        if time.time() - cache_time < 300:  # 5 minutes
            return True
        
        # Sample-based validation: check only a subset of files for performance
        source_extensions = {'.py', '.java', '.js', '.ts'}
        files_checked = 0
        max_files_to_check = self.config.context.cache_validation_sample_size
        
        for file_path in self.root_path.rglob("*"):
            if files_checked >= max_files_to_check:
                break
                
            if file_path.is_file() and file_path.suffix in source_extensions:
                try:
                    if file_path.stat().st_mtime > cache_time:
                        return False
                    files_checked += 1
                except (OSError, IOError):
                    # If we can't stat the file, continue checking others
                    continue
        
        return True
    
    def _calculate_relevance(self, name: str, query: str) -> float:
        """Calculate relevance score for a name match."""
        name_lower = name.lower()
        query_lower = query.lower()
        
        if name_lower == query_lower:
            return 1.0
        elif name_lower.startswith(query_lower):
            return 0.9
        elif query_lower in name_lower:
            return 0.7
        else:
            # Use simple string similarity
            common_chars = len(set(name_lower) & set(query_lower))
            total_chars = len(set(name_lower) | set(query_lower))
            return common_chars / total_chars if total_chars > 0 else 0.0
    
    def _calculate_code_relevance(self, code: str, query: str) -> float:
        """Calculate relevance score for code content match."""
        query_lower = query.lower()
        code_lower = code.lower()
        
        # Count occurrences
        occurrences = code_lower.count(query_lower)
        if occurrences == 0:
            return 0.0
        
        # Factor in code length and position
        relevance = min(occurrences * 0.1, 0.6)  # Max 0.6 for code matches
        
        # Boost if query appears in function/class names or comments
        if f"def {query_lower}" in code_lower or f"class {query_lower}" in code_lower:
            relevance += 0.3
        
        return min(relevance, 1.0)
    
    def _get_context_nodes(self, node_id: str, context_type: str) -> List[str]:
        """Get context nodes for a given node based on context type."""
        if context_type == "minimal":
            return self.graph.get_dependencies(node_id)[:5]
        elif context_type == "comprehensive":
            related = self.graph.find_related_nodes(node_id, max_distance=2)
            return list(related.keys())[:15]
        else:  # "all"
            return self.graph.get_dependencies(node_id) + self.graph.get_dependents(node_id)
    
    def _get_relationship_type(self, source_id: str, target_id: str) -> str:
        """Get the type of relationship between two nodes."""
        if source_id not in self.graph.graph or target_id not in self.graph.graph:
            return "unknown"
        
        edges = self.graph.graph.get_edge_data(source_id, target_id)
        if edges:
            edge_types = [edge_data.get('type', 'unknown') for edge_data in edges.values()]
            return edge_types[0].value if edge_types else "unknown"
        
        # Check reverse direction
        edges = self.graph.graph.get_edge_data(target_id, source_id)
        if edges:
            edge_types = [edge_data.get('type', 'unknown') for edge_data in edges.values()]
            return f"reverse_{edge_types[0].value}" if edge_types else "unknown"
        
        return "indirect"
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from a question."""
        import re
        
        # Remove common question words
        stop_words = {
            'what', 'where', 'how', 'when', 'why', 'who', 'which', 'does', 'do',
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that'
        }
        
        # Extract potential identifiers and quoted terms
        words = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', question)
        quoted_terms = re.findall(r'["\']([^"\']+)["\']', question)
        
        key_terms = []
        for word in words:
            if word.lower() not in stop_words and len(word) > 2:
                key_terms.append(word)
        
        key_terms.extend(quoted_terms)
        
        return list(set(key_terms))
    
    def _extract_function_terms(self, question: str) -> List[str]:
        """Extract terms that are likely function names."""
        key_terms = self._extract_key_terms(question)
        
        # Look for terms that match functions in our graph
        function_terms = []
        for term in key_terms:
            if self.graph.find_node(term, NodeType.FUNCTION):
                function_terms.append(term)
        
        return function_terms
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the code map."""
        if not self.graph:
            return {'error': 'Code map not built yet'}
        
        stats = self.graph.get_statistics()
        stats['last_analysis_time'] = self.last_analysis_time
        stats['active_languages'] = list(self.active_languages)
        
        return stats
    
    def _build_optimized_graph(self) -> SemanticGraph:
        """Build code map with performance optimizations and progress tracking."""
        # Use fast file discovery
        file_patterns = ['**/*.py', '**/*.java', '**/*.js', '**/*.ts']
        ignore_patterns = [
            '__pycache__', '.git', 'node_modules', 'dist', 'build',
            '.pytest_cache', '.mypy_cache', 'target', '*.pyc'
        ]
        
        all_files = []
        for pattern in file_patterns:
            pattern_files = list(self.root_path.glob(pattern))
            filtered_files = [
                f for f in pattern_files
                if f.is_file() and not any(ignore in str(f) for ignore in ignore_patterns)
            ]
            all_files.extend(filtered_files)
        
        # Remove duplicates and sort by size
        unique_files = list(set(all_files))
        unique_files.sort(key=lambda f: f.stat().st_size if f.exists() else 0)
        
        if not unique_files:
            return self.dependency_analyzer.analyze_codebase()
        
        # Start progress tracking
        code_map_progress.start_code_mapping(len(unique_files))
        
        # Use dependency analyzer which will handle its own progress tracking
        return self.dependency_analyzer.analyze_codebase()
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Quick check if file should be analyzed."""
        try:
            # Skip very large files that might slow down analysis
            if file_path.stat().st_size > self.config.context.max_file_size_mb * 1024 * 1024:
                return False
            
            # Skip binary files and known non-source extensions
            skip_extensions = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.log'}
            if file_path.suffix.lower() in skip_extensions:
                return False
            
            return True
        except (OSError, IOError):
            return False