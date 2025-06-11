from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import os
import re
from ..search import CodeSearcher
from ..parser import SyntaxParser
from ..mapper import CodeMapper
from ..utils.progress_tracker import ProgressTracker, estimate_tokens, estimate_max_tokens
from ..utils.config import load_config


class ContextExtractor:
    def __init__(self, root_path: str, max_context_size: int = None, show_progress: bool = True):
        self.root_path = Path(root_path).resolve()
        self.searcher = CodeSearcher(root_path)
        self.show_progress = show_progress
        self.progress_tracker = ProgressTracker(show_details=show_progress)
        
        # Load configuration
        self.config = load_config()
        
        # Use config values for context optimization
        self.max_context_size = max_context_size or self.config.context.max_context_size
        
        # Detect active languages in the repository
        active_languages = self._detect_active_languages()
        self.parser = SyntaxParser(active_languages=active_languages)
        
        # Initialize the semantic code mapper (builds on top of existing tree-sitter)
        # Only initialize if semantic analysis is enabled
        if self.config.search.enable_semantic_analysis:
            self.code_mapper = CodeMapper(root_path)
        else:
            self.code_mapper = None

    def extract_relevant_context(self, question: str) -> Dict[str, any]:
        """Extract relevant context for a question using both keyword and semantic search."""
        # Reset progress tracker for new operation
        self.progress_tracker.reset()
        
        if self.show_progress:
            self.progress_tracker.start_live_tracking()
        
        try:
            # Phase 1: Extract keywords (existing approach)
            self.progress_tracker.start_step("🔍 Extracting keywords", phase="keywords")
            keywords = self._extract_keywords(question)
            self.progress_tracker.complete_current_step(keywords_found=len(keywords), keywords=keywords)
            
            # Phase 1.5: Semantic search using code map (NEW!) - Only if enabled
            semantic_context = {}
            semantic_results = []
            if self.config.search.enable_semantic_analysis and self.code_mapper:
                self.progress_tracker.start_step("🧠 Semantic analysis", phase="semantic")
                semantic_context = self.code_mapper.get_context_for_question(question)
                semantic_results = semantic_context.get('relevant_nodes', [])
                self.progress_tracker.complete_current_step(
                    semantic_matches=len(semantic_results),
                    related_concepts=len(semantic_context.get('related_files', []))
                )
            
            # Phase 2: Initial search (combine keyword + semantic)
            search_step = self.progress_tracker.track_search_phase(keywords)
            keyword_results = self._perform_initial_search(keywords)
            
            # Merge semantic results with keyword results
            combined_results = self._merge_search_results(keyword_results, semantic_results)
            unique_files = len(set(r["file"] for r in combined_results))
            search_tool_description = self.searcher.primary_search_tool
            if self.config.search.enable_semantic_analysis and semantic_results:
                search_tool_description += "+semantic"
            
            self.progress_tracker.complete_current_step(
                files_found=unique_files, 
                matches=len(combined_results),
                tool=search_tool_description
            )
            
            # Phase 3: Expand context using syntax tree
            syntax_step = self.progress_tracker.track_syntax_analysis(unique_files)
            expanded_context = self._expand_context_syntactically(combined_results)
            functions_found = sum(len(info.get("functions", set())) for info in expanded_context.values())
            classes_found = sum(len(info.get("classes", set())) for info in expanded_context.values())
            self.progress_tracker.complete_current_step(
                functions_found=functions_found,
                classes_found=classes_found,
                files_analyzed=len(expanded_context)
            )
            
            # Phase 4: Add related files
            self.progress_tracker.start_step("🔗 Finding related files", phase="relations")
            final_context = self._add_related_files(expanded_context)
            related_files = sum(1 for info in final_context.values() if info.get("is_related"))
            self.progress_tracker.complete_current_step(related_files=related_files)
            
            # Phase 5: Optimize context size
            initial_size = sum(len(self._get_file_content_with_context(path, info)["content"]) 
                             for path, info in final_context.items())
            optimize_step = self.progress_tracker.track_optimization(initial_size, self.max_context_size)
            optimized_context = self._optimize_context_size(final_context)
            final_size = optimized_context["total_size"]
            
            # Update context stats
            context_tokens = estimate_tokens(''.join(f["content"] for f in optimized_context["files"]))
            max_tokens = estimate_max_tokens(self.max_context_size)
            
            self.progress_tracker.update_context_stats(
                total_files_found=len(final_context),
                files_included=optimized_context["total_files"],
                context_tokens=context_tokens,
                max_context_tokens=max_tokens,
                context_chars=final_size,
                search_tools_used=[self.searcher.primary_search_tool],
                parsing_tools_used=list(self.parser.languages.keys()) if hasattr(self.parser, 'languages') else []
            )
            
            self.progress_tracker.complete_current_step(
                final_size=final_size,
                files_included=optimized_context["total_files"],
                reduction_percent=((initial_size - final_size) / initial_size * 100) if initial_size > 0 else 0
            )
            
            return {
                "question": question,
                "keywords": keywords,
                "context_files": optimized_context,
                "total_size": final_size,
                "progress_summary": self.progress_tracker.get_timing_summary(),
                "context_utilization": self.progress_tracker.context_stats,
                "semantic_context": semantic_context,  # Add semantic information if enabled
                "search_strategy": "hybrid_keyword_semantic" if self.config.search.enable_semantic_analysis else "keyword_only"
            }
        
        finally:
            if self.show_progress:
                self.progress_tracker.stop_live_tracking()
                self.progress_tracker.show_summary()

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from the question."""
        # Remove common words
        stop_words = {
            "what", "where", "how", "does", "the", "is", "are", "can",
            "in", "of", "to", "for", "with", "and", "or", "a", "an"
        }
        
        # Extract potential code identifiers
        code_pattern = r'[A-Za-z_][A-Za-z0-9_]*'
        potential_identifiers = re.findall(code_pattern, question)
        
        # Filter keywords
        keywords = []
        for word in potential_identifiers:
            if word.lower() not in stop_words and len(word) > 2:
                keywords.append(word)
        
        # Also look for quoted strings
        quoted = re.findall(r'["\']([^"\'\']+)["\']', question)
        keywords.extend(quoted)
        
        return list(set(keywords))

    def _merge_search_results(self, keyword_results: List[Dict[str, any]], 
                             semantic_results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Merge keyword search results with semantic search results."""
        merged_results = keyword_results.copy()
        
        # Convert semantic results to keyword result format
        for semantic_result in semantic_results:
            node = semantic_result['node']
            if node.file_path and node.line_start:
                # Create a keyword-style result from semantic node
                keyword_style_result = {
                    "file": node.file_path,
                    "line_number": node.line_start,
                    "line": node.metadata.get('code', node.name)[:100],  # Truncate long code
                    "column": 0,
                    "score": semantic_result['relevance'],
                    "match_type": "semantic",
                    "semantic_info": {
                        "node_type": node.type.value,
                        "node_name": node.name,
                        "related_nodes": semantic_result.get('related_nodes', [])
                    }
                }
                merged_results.append(keyword_style_result)
        
        # Remove duplicates based on file and line number
        seen_locations = set()
        unique_results = []
        
        for result in merged_results:
            location = (result["file"], result["line_number"])
            if location not in seen_locations:
                seen_locations.add(location)
                unique_results.append(result)
        
        # Sort by score (keyword results have score, semantic results have relevance)
        unique_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        
        return unique_results

    def _perform_initial_search(self, keywords: List[str]) -> List[Dict[str, any]]:
        """Perform initial search using keywords."""
        all_results = []
        seen_locations = set()
        
        for keyword in keywords[:self.config.context.max_keywords_to_process]:
            results = self.searcher.search_text(keyword, max_results=self.config.context.max_results_per_keyword)
            
            for result in results:
                location = (result["file"], result["line_number"])
                if location not in seen_locations:
                    seen_locations.add(location)
                    all_results.append(result)
        
        # Sort by relevance (number of keyword matches)
        relevance_scores = {}
        for result in all_results:
            key = (result["file"], result["line_number"])
            relevance_scores[key] = sum(
                1 for kw in keywords 
                if kw.lower() in result["line"].lower()
            )
        
        all_results.sort(
            key=lambda r: relevance_scores[(r["file"], r["line_number"])],
            reverse=True
        )
        
        return all_results[:self.config.context.max_search_results]

    def _expand_context_syntactically(self, search_results: List[Dict[str, any]]) -> Dict[str, any]:
        """Expand context using syntax tree analysis."""
        expanded_files = {}
        processed_files = set()
        
        for result in search_results:
            file_path = result["file"]
            if file_path not in expanded_files:
                expanded_files[file_path] = {
                    "path": file_path,
                    "matches": [],
                    "functions": set(),
                    "classes": set(),
                    "imports": set()
                }
            
            expanded_files[file_path]["matches"].append(result)
            
            # Only process each file once for syntax analysis
            if file_path not in processed_files:
                processed_files.add(file_path)
                
                # Extract ALL functions and classes from files with matches
                functions = self.parser.extract_functions(file_path)
                classes = self.parser.extract_classes(file_path)
                
                # Add all functions to the file info
                for func in functions:
                    expanded_files[file_path]["functions"].add(
                        (func.get("name", "anonymous"), func["start_line"], func["end_line"])
                    )
                
                # Add all classes to the file info
                for cls in classes:
                    expanded_files[file_path]["classes"].add(
                        (cls.get("name", "anonymous"), cls["start_line"], cls["end_line"])
                    )
        
        return expanded_files

    def _add_related_files(self, expanded_context: Dict[str, any]) -> Dict[str, any]:
        """Add related files based on imports and references."""
        related_files = set()
        total_imports_found = 0
        
        for file_path, file_info in expanded_context.items():
            # Parse imports from the file
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Python imports
                if file_path.endswith('.py'):
                    import_patterns = [
                        r'from\s+([\w\.]+)\s+import',
                        r'import\s+([\w\.]+)'
                    ]
                    for pattern in import_patterns:
                        matches = re.findall(pattern, content)
                        total_imports_found += len(matches)
                        for match in matches:
                            potential_paths = []
                            
                            if match.startswith('.'):
                                # Relative import - resolve from current file's directory
                                current_file_path = Path(file_path)
                                current_dir = current_file_path.parent
                                
                                # Count leading dots to determine how many levels up
                                dots = 0
                                for char in match:
                                    if char == '.':
                                        dots += 1
                                    else:
                                        break
                                
                                # Navigate up the directory tree
                                target_dir = current_dir
                                for _ in range(dots - 1):
                                    target_dir = target_dir.parent
                                
                                # Get the module path after the dots
                                module_part = match[dots:]
                                if module_part:
                                    potential_paths = [
                                        target_dir / module_part.replace('.', '/') / "__init__.py",
                                        target_dir / (module_part.replace('.', '/') + ".py")
                                    ]
                                else:
                                    # Just dots, import the package itself
                                    potential_paths = [target_dir / "__init__.py"]
                            else:
                                # Absolute import - skip built-in modules and third-party imports
                                if not self._is_project_module(match):
                                    continue
                                    
                                # Convert module path to file path
                                potential_paths = [
                                    self.root_path / match.replace('.', '/') / "__init__.py",
                                    self.root_path / (match.replace('.', '/') + ".py")
                                ]
                                
                                # Also check within src/ directory
                                src_paths = [
                                    self.root_path / "src" / match.replace('.', '/') / "__init__.py",
                                    self.root_path / "src" / (match.replace('.', '/') + ".py")
                                ]
                                potential_paths.extend(src_paths)
                            
                            for path in potential_paths:
                                if path.exists() and str(path) not in expanded_context:
                                    related_files.add(str(path))
                
                # JavaScript/TypeScript imports
                elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    import_pattern = r'import.*from\s+["\']([^"\']+)["\']'
                    matches = re.findall(import_pattern, content)
                    total_imports_found += len(matches)
                    for match in matches:
                        if match.startswith('.'):
                            # Relative import
                            base_dir = Path(file_path).parent
                            potential_paths = [
                                base_dir / match,
                                base_dir / (match + ".js"),
                                base_dir / (match + ".ts"),
                                base_dir / match / "index.js",
                                base_dir / match / "index.ts"
                            ]
                            for path in potential_paths:
                                if path.exists() and str(path) not in expanded_context:
                                    related_files.add(str(path.resolve()))
                        else:
                            # Absolute import - check within project root
                            potential_paths = [
                                self.root_path / match,
                                self.root_path / (match + ".js"),
                                self.root_path / (match + ".ts"),
                                self.root_path / match / "index.js",
                                self.root_path / match / "index.ts"
                            ]
                            for path in potential_paths:
                                if path.exists() and str(path) not in expanded_context:
                                    related_files.add(str(path.resolve()))
                
                # Java imports
                elif file_path.endswith('.java'):
                    java_import_patterns = [
                        r'import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*;',
                        r'import\s+static\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*;'
                    ]
                    for pattern in java_import_patterns:
                        matches = re.findall(pattern, content)
                        total_imports_found += len(matches)
                        for match in matches:
                            # Skip standard library imports
                            if self._is_java_standard_library(match):
                                continue
                            
                            # Convert Java package to file path
                            # e.g., com.example.MyClass -> com/example/MyClass.java
                            class_path = match.replace('.', '/')
                            
                            # Look for the class in common Java source directories
                            potential_paths = [
                                self.root_path / "src" / "main" / "java" / (class_path + ".java"),
                                self.root_path / "src" / "test" / "java" / (class_path + ".java"),
                                self.root_path / "src" / (class_path + ".java"),
                                self.root_path / (class_path + ".java")
                            ]
                            
                            for path in potential_paths:
                                if path.exists() and str(path) not in expanded_context:
                                    related_files.add(str(path))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Debug information
        if total_imports_found > 0 and len(related_files) == 0:
            print(f"Debug: Found {total_imports_found} imports but 0 related files. Files analyzed: {len(expanded_context)}")
        
        # Add related files to context
        for related_file in list(related_files)[:self.config.context.max_related_files]:
            if related_file not in expanded_context:
                expanded_context[related_file] = {
                    "path": related_file,
                    "matches": [],
                    "functions": set(),
                    "classes": set(),
                    "imports": set(),
                    "is_related": True
                }
        
        return expanded_context

    def _is_project_module(self, module_name: str) -> bool:
        """Check if a module is likely a project module vs built-in/third-party."""
        # Skip standard library modules
        builtin_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'collections', 'typing', 'pathlib',
            're', 'math', 'random', 'itertools', 'functools', 'hashlib', 'uuid',
            'logging', 'argparse', 'subprocess', 'threading', 'multiprocessing',
            'urllib', 'http', 'socket', 'ssl', 'email', 'csv', 'xml', 'html'
        }
        
        # Check if it's a built-in module or starts with built-in prefix
        root_module = module_name.split('.')[0]
        if root_module in builtin_modules:
            return False
            
        # Skip common third-party packages
        third_party_prefixes = {
            'numpy', 'pandas', 'matplotlib', 'requests', 'flask', 'django',
            'sqlalchemy', 'pytest', 'click', 'rich', 'pydantic', 'fastapi',
            'aiohttp', 'asyncio', 'setuptools', 'pip', 'wheel'
        }
        
        if root_module in third_party_prefixes:
            return False
            
        return True

    def _is_java_standard_library(self, import_name: str) -> bool:
        """Check if a Java import is from the standard library."""
        java_stdlib_prefixes = {
            'java.', 'javax.', 'javafx.', 'jdk.', 'com.sun.', 'sun.',
            'org.w3c.', 'org.xml.', 'org.ietf.', 'org.omg.'
        }
        
        # Check common third-party libraries
        third_party_prefixes = {
            'org.springframework.', 'org.apache.', 'org.junit.', 'org.mockito.',
            'org.slf4j.', 'org.hibernate.', 'com.google.', 'com.fasterxml.',
            'org.jetbrains.', 'io.github.', 'org.testng.'
        }
        
        for prefix in java_stdlib_prefixes:
            if import_name.startswith(prefix):
                return True
                
        for prefix in third_party_prefixes:
            if import_name.startswith(prefix):
                return True
                
        return False

    def _optimize_context_size(self, expanded_context: Dict[str, any]) -> Dict[str, any]:
        """Optimize context to fit within size limits."""
        files_to_include = []
        current_size = 0
        
        # Categorize files by importance
        high_priority_files = []  # Files with direct matches
        medium_priority_files = []  # Files with functions/classes but no matches
        low_priority_files = []  # Related files (imports)
        
        for file_path, file_info in expanded_context.items():
            if file_info["matches"]:
                # Files with direct keyword matches - highest priority
                high_priority_files.append((file_path, file_info))
            elif file_info.get("functions") or file_info.get("classes"):
                # Files with relevant functions/classes but no direct matches - medium priority
                medium_priority_files.append((file_path, file_info))
            elif file_info.get("is_related"):
                # Files found through imports - lower priority
                low_priority_files.append((file_path, file_info))
        
        # Sort high priority files by number of matches
        high_priority_files.sort(key=lambda x: len(x[1]["matches"]), reverse=True)
        
        # Sort medium priority by number of functions + classes
        medium_priority_files.sort(
            key=lambda x: len(x[1].get("functions", set())) + len(x[1].get("classes", set())), 
            reverse=True
        )
        
        # Process files in priority order
        all_priority_files = [
            (high_priority_files, "high"),
            (medium_priority_files, "medium"), 
            (low_priority_files, "low")
        ]
        
        for priority_group, priority_level in all_priority_files:
            for file_path, file_info in priority_group:
                # Stop if we're getting close to the limit (use configurable buffer for low priority)
                if priority_level == "low" and current_size >= self.max_context_size * self.config.context.context_buffer_percentage:
                    break
                
                file_content = self._get_file_content_with_context(file_path, file_info)
                file_size = len(file_content["content"])
                
                if current_size + file_size <= self.max_context_size:
                    files_to_include.append(file_content)
                    current_size += file_size
                else:
                    # Try to include partial content
                    partial_content = self._get_partial_file_content(file_path, file_info)
                    partial_size = len(partial_content["content"])
                    if current_size + partial_size <= self.max_context_size:
                        files_to_include.append(partial_content)
                        current_size += partial_size
        
        return {
            "files": files_to_include,
            "total_files": len(files_to_include),
            "total_size": current_size
        }

    def _get_file_content_with_context(self, file_path: str, file_info: Dict[str, any]) -> Dict[str, any]:
        """Get file content with highlighted matches."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Mark important lines
            important_lines = set()
            for match in file_info["matches"]:
                important_lines.add(match["line_number"])
            
            for func_name, start, end in file_info["functions"]:
                for i in range(start, min(end + 1, len(lines) + 1)):
                    important_lines.add(i)
            
            for cls_name, start, end in file_info["classes"]:
                for i in range(start, min(end + 1, len(lines) + 1)):
                    important_lines.add(i)
            
            content_lines = []
            for i, line in enumerate(lines, 1):
                if i in important_lines:
                    content_lines.append(f"{i:4d} >>> {line.rstrip()}")
                else:
                    content_lines.append(f"{i:4d}     {line.rstrip()}")
            
            return {
                "path": file_path,
                "language": self._detect_language(file_path),
                "content": "\n".join(content_lines),
                "matches": len(file_info["matches"]),
                "important_lines": list(important_lines)
            }
        except Exception as e:
            return {
                "path": file_path,
                "error": str(e),
                "content": ""
            }

    def _get_partial_file_content(self, file_path: str, file_info: Dict[str, any]) -> Dict[str, any]:
        """Get partial file content focusing on matches."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Collect line ranges to include
            line_ranges = []
            context_size = self.config.context.partial_content_window_size
            
            for match in file_info["matches"]:
                start = max(1, match["line_number"] - context_size)
                end = min(len(lines), match["line_number"] + context_size)
                line_ranges.append((start, end))
            
            # Merge overlapping ranges
            merged_ranges = self._merge_ranges(line_ranges)
            
            content_parts = []
            for start, end in merged_ranges:
                content_parts.append(f"\n# Lines {start}-{end}:\n")
                for i in range(start - 1, end):
                    if i < len(lines):
                        line_num = i + 1
                        if any(m["line_number"] == line_num for m in file_info["matches"]):
                            content_parts.append(f"{line_num:4d} >>> {lines[i].rstrip()}")
                        else:
                            content_parts.append(f"{line_num:4d}     {lines[i].rstrip()}")
            
            return {
                "path": file_path,
                "language": self._detect_language(file_path),
                "content": "\n".join(content_parts),
                "matches": len(file_info["matches"]),
                "is_partial": True
            }
        except Exception as e:
            return {
                "path": file_path,
                "error": str(e),
                "content": ""
            }

    def _merge_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping line ranges."""
        if not ranges:
            return []
        
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        
        for start, end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged

    def _detect_active_languages(self) -> Set[str]:
        """Detect which programming languages are used in the repository."""
        active_languages = set()
        
        # File extension to language mapping
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript", 
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript", 
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c"
        }
        
        try:
            # Sample files to avoid checking every file in large repos
            file_count = 0
            for file_path in self.root_path.rglob("*"):
                if file_path.is_file() and file_count < 1000:  # Limit to first 1000 files
                    suffix = file_path.suffix.lower()
                    if suffix in ext_to_lang:
                        active_languages.add(ext_to_lang[suffix])
                        file_count += 1
                    
                    # Stop early if we've found common languages
                    if len(active_languages) >= 3 and file_count > 100:
                        break
        except Exception as e:
            print(f"Warning: Could not detect active languages: {e}")
            # Fallback to common languages
            active_languages = {"python", "javascript"}
        
        if not active_languages:
            # Default fallback
            active_languages = {"python"}
            
        return active_languages

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "objective-c",
            ".mm": "objective-c++",
        }
        
        suffix = Path(file_path).suffix.lower()
        return ext_map.get(suffix, "text")
    
    def _merge_and_prioritize_regions(self, regions: List[Dict[str, any]], budget: int,
                                    lines: List[str]) -> List[Dict[str, any]]:
        """Merge overlapping regions and select best ones within budget."""
        if not regions:
            return []
        
        # Sort by start position
        regions.sort(key=lambda x: (x["start"], -x["priority"]))
        
        # Merge overlapping regions
        merged = []
        current = regions[0].copy()
        
        for region in regions[1:]:
            if region["start"] <= current["end"] + 5:  # Allow small gaps
                # Merge regions
                current["end"] = max(current["end"], region["end"])
                current["priority"] = max(current["priority"], region["priority"])
                if region.get("reason") and region["priority"] >= current["priority"]:
                    current["reason"] = region["reason"]
            else:
                merged.append(current)
                current = region.copy()
        
        merged.append(current)
        
        # Calculate sizes and select regions within budget
        for region in merged:
            region["size"] = sum(len(lines[i]) for i in range(
                region["start"] - 1, min(region["end"], len(lines))
            ))
        
        # Sort by priority and select
        merged.sort(key=lambda x: x["priority"], reverse=True)
        
        selected = []
        used_budget = 0
        
        for region in merged:
            if used_budget + region["size"] <= budget * 0.9:  # Leave some buffer
                selected.append(region)
                used_budget += region["size"]
        
        # Re-sort by position for output
        selected.sort(key=lambda x: x["start"])
        return selected
    
    def _load_project_metadata(self):
        """Load project metadata for better context understanding."""
        self.project_metadata = {
            "type": "unknown",
            "main_language": "unknown",
            "frameworks": [],
            "has_tests": False,
            "structure": {}
        }
        
        # Detect project type
        if (self.root_path / "package.json").exists():
            self.project_metadata["type"] = "node"
            self.project_metadata["main_language"] = "javascript"
        elif (self.root_path / "setup.py").exists() or (self.root_path / "pyproject.toml").exists():
            self.project_metadata["type"] = "python"
            self.project_metadata["main_language"] = "python"
        elif (self.root_path / "pom.xml").exists():
            self.project_metadata["type"] = "maven"
            self.project_metadata["main_language"] = "java"
        elif (self.root_path / "go.mod").exists():
            self.project_metadata["type"] = "go"
            self.project_metadata["main_language"] = "go"
        elif (self.root_path / "Cargo.toml").exists():
            self.project_metadata["type"] = "rust"
            self.project_metadata["main_language"] = "rust"
        
        # Check for test directories
        test_dirs = ["test", "tests", "spec", "__tests__"]
        for test_dir in test_dirs:
            if (self.root_path / test_dir).exists():
                self.project_metadata["has_tests"] = True
                break
    
    def _enrich_context_metadata(self, context: Dict[str, any]) -> Dict[str, any]:
        """Add metadata to help LLM understand the context better."""
        context["metadata"] = {
            "project_type": self.project_metadata["type"],
            "main_language": self.project_metadata["main_language"],
            "file_count": context.get("total_files", 0),
            "has_partial_files": any(f.get("is_partial") for f in context.get("files", [])),
            "search_strategies_used": list(set(
                result.get("search_strategy", "unknown")
                for file_data in context.get("files", [])
                for result in file_data.get("matches", [])
            ))
        }
        return context
    
    def _get_search_strategy_summary(self, results: List[Dict[str, any]]) -> Dict[str, int]:
        """Summarize which search strategies found results."""
        strategy_counts = {}
        for result in results:
            strategy = result.get("search_strategy", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        return strategy_counts