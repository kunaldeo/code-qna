import subprocess
import os
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import re
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


class CodeSearcher:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path).resolve()
        self._check_tools()
        self._init_ignore_patterns()

    def _check_tools(self):
        """Check if required search tools are available."""
        tools = ["rg", "ag", "grep", "find", "fd", "ctags"]
        self.available_tools = {}
        
        for tool in tools:
            self.available_tools[tool] = shutil.which(tool) is not None
        
        # Prefer ripgrep, then silver searcher, then grep
        if self.available_tools.get("rg"):
            self.primary_search_tool = "rg"
        elif self.available_tools.get("ag"):
            self.primary_search_tool = "ag"
        else:
            self.primary_search_tool = "grep"
    
    def _init_ignore_patterns(self):
        """Initialize common ignore patterns."""
        self.ignore_patterns = [
            ".git", ".svn", ".hg", "node_modules", "__pycache__",
            "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.exe",
            "dist", "build", "target", ".venv", "venv", "env",
            ".tox", ".pytest_cache", ".mypy_cache", "*.egg-info",
            "coverage", ".coverage", "htmlcov", ".idea", ".vscode"
        ]

    def search_text(self, pattern: str, file_types: Optional[List[str]] = None,
                    max_results: int = 100, case_sensitive: bool = False,
                    whole_word: bool = False, regex: bool = True) -> List[Dict[str, any]]:
        """Search for text pattern in codebase."""
        results = []
        if self.primary_search_tool == "rg":
            results = self._ripgrep_search(pattern, file_types, max_results, case_sensitive, whole_word, regex)
        elif self.primary_search_tool == "ag":
            results = self._ag_search(pattern, file_types, max_results, case_sensitive, whole_word, regex)
        else:
            results = self._grep_search(pattern, file_types, max_results, case_sensitive, whole_word, regex)
        
        # Add search tool info to each result
        for result in results:
            result["search_tool"] = self.primary_search_tool
        
        return results
    
    def search_multiple_patterns(self, patterns: List[str], file_types: Optional[List[str]] = None,
                                max_results_per_pattern: int = 50) -> Dict[str, List[Dict[str, any]]]:
        """Search for multiple patterns concurrently."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(patterns), 5)) as executor:
            future_to_pattern = {
                executor.submit(self.search_text, pattern, file_types, max_results_per_pattern): pattern
                for pattern in patterns
            }
            
            for future in as_completed(future_to_pattern):
                pattern = future_to_pattern[future]
                try:
                    results[pattern] = future.result()
                except Exception as e:
                    print(f"Error searching for '{pattern}': {e}")
                    results[pattern] = []
        
        return results

    def _ripgrep_search(self, pattern: str, file_types: Optional[List[str]],
                        max_results: int, case_sensitive: bool, whole_word: bool, regex: bool) -> List[Dict[str, any]]:
        """Use ripgrep for fast searching."""
        cmd = ["rg", "-n", "--json", "-m", str(max_results)]
        
        # Add search options
        if not case_sensitive:
            cmd.append("-i")
        if whole_word:
            cmd.append("-w")
        if not regex:
            cmd.append("-F")  # Fixed string search
        
        # Add ignore patterns
        for ignore in self.ignore_patterns:
            cmd.extend(["--glob", f"!{ignore}"])
        
        # Add file type filters
        if file_types:
            for ft in file_types:
                # Map common extensions to ripgrep types
                type_map = {
                    "py": "python",
                    "js": "js",
                    "ts": "ts",
                    "java": "java",
                    "go": "go",
                    "rs": "rust",
                    "cpp": "cpp",
                    "c": "c",
                    "cs": "csharp",
                    "rb": "ruby",
                    "php": "php",
                    "swift": "swift",
                    "kt": "kotlin",
                }
                rg_type = type_map.get(ft, ft)
                cmd.extend(["-t", rg_type])
        
        cmd.extend([pattern, str(self.root_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return self._parse_ripgrep_output(result.stdout)
        except subprocess.TimeoutExpired:
            print("Search timed out after 30 seconds")
            return []
        except Exception as e:
            print(f"Ripgrep search error: {e}")
            return []

    def _parse_ripgrep_output(self, output: str) -> List[Dict[str, any]]:
        """Parse ripgrep JSON output."""
        results = []
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "match":
                    match_data = data["data"]
                    file_path = match_data["path"]["text"]
                    
                    # Calculate match score based on various factors
                    score = 1.0
                    
                    # Boost score for matches in certain file types
                    if file_path.endswith(('.py', '.js', '.ts', '.java', '.go')):
                        score *= 1.2
                    
                    # Boost score for matches in source directories
                    if '/src/' in file_path or '/lib/' in file_path:
                        score *= 1.1
                    
                    # Reduce score for test files
                    if '/test/' in file_path or file_path.endswith(('_test.py', '.test.js', 'Test.java')):
                        score *= 0.8
                    
                    results.append({
                        "file": file_path,
                        "line_number": match_data["line_number"],
                        "line": match_data["lines"]["text"].strip(),
                        "column": match_data["submatches"][0]["start"] if match_data.get("submatches") else 0,
                        "score": score,
                        "match_count": len(match_data.get("submatches", []))
                    })
            except json.JSONDecodeError:
                continue
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _grep_search(self, pattern: str, file_types: Optional[List[str]],
                     max_results: int, case_sensitive: bool, whole_word: bool, regex: bool) -> List[Dict[str, any]]:
        """Fallback to grep for searching."""
        cmd = ["grep", "-rn", "-m", str(max_results)]
        
        if not case_sensitive:
            cmd.append("-i")
        if whole_word:
            cmd.append("-w")
        if not regex:
            cmd.append("-F")
        
        # Add exclude patterns
        for ignore in self.ignore_patterns:
            cmd.extend(["--exclude", ignore])
            cmd.extend(["--exclude-dir", ignore])
        
        if file_types:
            for ft in file_types:
                cmd.extend(["--include", f"*.{ft}"])
        
        cmd.extend([pattern, str(self.root_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return self._parse_grep_output(result.stdout)
        except subprocess.TimeoutExpired:
            print("Search timed out after 30 seconds")
            return []
        except Exception as e:
            print(f"Grep search error: {e}")
            return []

    def _parse_grep_output(self, output: str) -> List[Dict[str, any]]:
        """Parse grep output."""
        results = []
        pattern = re.compile(r'^(.+?):(\d+):(.*)$')
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            match = pattern.match(line)
            if match:
                results.append({
                    "file": match.group(1),
                    "line_number": int(match.group(2)),
                    "line": match.group(3).strip(),
                    "column": 0
                })
        
        return results

    def find_files(self, pattern: str, file_types: Optional[List[str]] = None,
                   exclude_dirs: Optional[List[str]] = None) -> List[str]:
        """Find files matching a pattern."""
        if self.available_tools.get("fd"):
            return self._fd_find_files(pattern, file_types, exclude_dirs)
        else:
            return self._find_files(pattern, file_types, exclude_dirs)
    
    def _fd_find_files(self, pattern: str, file_types: Optional[List[str]],
                       exclude_dirs: Optional[List[str]]) -> List[str]:
        """Use fd for fast file finding."""
        cmd = ["fd", "-t", "f", pattern, str(self.root_path)]
        
        if file_types:
            for ft in file_types:
                cmd.extend(["-e", ft])
        
        if exclude_dirs:
            for exclude in exclude_dirs:
                cmd.extend(["-E", exclude])
        
        # Add default excludes
        for ignore in self.ignore_patterns:
            cmd.extend(["-E", ignore])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception as e:
            print(f"fd error: {e}")
            return self._find_files(pattern, file_types, exclude_dirs)
    
    def _find_files(self, pattern: str, file_types: Optional[List[str]],
                    exclude_dirs: Optional[List[str]]) -> List[str]:
        """Use find command for file finding."""
        cmd = ["find", str(self.root_path)]
        
        # Exclude directories
        all_excludes = set(self.ignore_patterns)
        if exclude_dirs:
            all_excludes.update(exclude_dirs)
        
        for exclude in all_excludes:
            cmd.extend(["-path", f"*/{exclude}", "-prune", "-o"])
        
        cmd.extend(["-type", "f"])
        
        if pattern:
            cmd.extend(["-name", pattern])
        
        if file_types:
            cmd.append("(")
            for i, ft in enumerate(file_types):
                if i > 0:
                    cmd.append("-o")
                cmd.extend(["-name", f"*.{ft}"])
            cmd.append(")")
        
        cmd.append("-print")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            # Filter out any remaining unwanted paths
            return [f for f in files if not any(ignore in f for ignore in all_excludes)]
        except Exception as e:
            print(f"Find files error: {e}")
            return []

    def get_file_context(self, file_path: str, line_number: int,
                         context_lines: int = 10) -> Dict[str, any]:
        """Get context around a specific line in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            return {
                "file": file_path,
                "target_line": line_number,
                "context": {
                    "start": start + 1,
                    "end": end,
                    "lines": [
                        {
                            "number": i + 1,
                            "content": lines[i].rstrip(),
                            "is_target": i + 1 == line_number
                        }
                        for i in range(start, end)
                    ]
                }
            }
        except Exception as e:
            print(f"Error reading file context: {e}")
            return {}
    
    def _ag_search(self, pattern: str, file_types: Optional[List[str]],
                   max_results: int, case_sensitive: bool, whole_word: bool, regex: bool) -> List[Dict[str, any]]:
        """Use silver searcher (ag) for searching."""
        cmd = ["ag", "--numbers", "--noheading", "-m", str(max_results)]
        
        if not case_sensitive:
            cmd.append("-i")
        if whole_word:
            cmd.append("-w")
        if not regex:
            cmd.append("-Q")  # Literal search
        
        # Add file type filters
        if file_types:
            for ft in file_types:
                cmd.extend(["--file-search-regex", f"\\.{ft}$"])
        
        cmd.extend([pattern, str(self.root_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return self._parse_ag_output(result.stdout)
        except subprocess.TimeoutExpired:
            print("Search timed out after 30 seconds")
            return []
        except Exception as e:
            print(f"Ag search error: {e}")
            return []
    
    def _parse_ag_output(self, output: str) -> List[Dict[str, any]]:
        """Parse silver searcher output."""
        results = []
        pattern = re.compile(r'^(.+?):(\d+):(.*)$')
        
        for line in output.strip().split('\n'):
            if not line:
                continue
            match = pattern.match(line)
            if match:
                results.append({
                    "file": match.group(1),
                    "line_number": int(match.group(2)),
                    "line": match.group(3).strip(),
                    "column": 0,
                    "score": 1.0
                })
        
        return results
    
    def search_symbols(self, symbol_name: str, symbol_type: Optional[str] = None) -> List[Dict[str, any]]:
        """Search for symbols (functions, classes, variables) using ctags if available."""
        if not self.available_tools.get("ctags"):
            # Fallback to text search
            patterns = []
            if symbol_type in [None, "function"]:
                patterns.extend([f"def {symbol_name}", f"function {symbol_name}", f"{symbol_name}()"])
            if symbol_type in [None, "class"]:
                patterns.extend([f"class {symbol_name}", f"struct {symbol_name}", f"interface {symbol_name}"])
            
            all_results = []
            for pattern in patterns:
                results = self.search_text(pattern, max_results=20, regex=False)
                all_results.extend(results)
            
            return all_results
        
        # Use ctags
        try:
            # Generate tags file
            tags_file = self.root_path / ".code-qna-tags"
            subprocess.run(
                ["ctags", "-f", str(tags_file), "-R", "--fields=+n", str(self.root_path)],
                capture_output=True
            )
            
            # Search in tags file
            results = []
            with open(tags_file, 'r') as f:
                for line in f:
                    if line.startswith(symbol_name):
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            results.append({
                                "symbol": parts[0],
                                "file": parts[1],
                                "pattern": parts[2],
                                "type": parts[3] if len(parts) > 3 else "unknown"
                            })
            
            # Clean up
            tags_file.unlink(missing_ok=True)
            return results
            
        except Exception as e:
            print(f"Ctags error: {e}")
            return []