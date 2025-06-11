from tree_sitter import Language, Parser, Node, Tree
from tree_sitter_language_pack import get_language, get_parser
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Generator
import os


class SyntaxParser:
    def __init__(self, active_languages: Optional[Set[str]] = None):
        self.languages = {}
        self.parsers = {}
        self.queries = {}
        self.active_languages = active_languages
        self._setup_languages()
        self._setup_queries()

    def _setup_languages(self):
        """Setup tree-sitter languages using tree-sitter-language-pack."""
        # Supported languages - focusing on Python and Java as requested
        supported_languages = ["python", "java"]
        
        # Only load languages that are active in the repository
        languages_to_load = supported_languages
        if self.active_languages:
            languages_to_load = [lang for lang in supported_languages if lang in self.active_languages]
        
        for lang in languages_to_load:
            try:
                language = get_language(lang)
                parser = get_parser(lang)
                
                self.languages[lang] = language
                self.parsers[lang] = parser
            except Exception as e:
                print(f"Error loading language {lang}: {e}")

    
    def _setup_queries(self):
        """Setup tree-sitter queries for supported languages."""
        # Python queries
        if "python" in self.languages:
            self.queries["python"] = {
                "functions": self.languages["python"].query("""
                    (function_definition
                        name: (identifier) @function.name
                        parameters: (parameters) @function.params
                        body: (block) @function.body) @function.def
                """),
                "classes": self.languages["python"].query("""
                    (class_definition
                        name: (identifier) @class.name
                        body: (block) @class.body) @class.def
                """),
                "docstrings": self.languages["python"].query("""
                    (module . (comment)* . (expression_statement (string)) @module_doc_str)
                    (class_definition
                        body: (block . (expression_statement (string)) @class_doc_str))
                    (function_definition
                        body: (block . (expression_statement (string)) @function_doc_str))
                """)
            }
        
        # Java queries
        if "java" in self.languages:
            self.queries["java"] = {
                "functions": self.languages["java"].query("""
                    (method_declaration
                        name: (identifier) @function.name
                        parameters: (formal_parameters) @function.params
                        body: (block) @function.body) @function.def
                    
                    (constructor_declaration
                        name: (identifier) @function.name
                        parameters: (formal_parameters) @function.params
                        body: (constructor_body) @function.body) @function.def
                """),
                "classes": self.languages["java"].query("""
                    (class_declaration
                        name: (identifier) @class.name
                        body: (class_body) @class.body) @class.def
                    
                    (interface_declaration
                        name: (identifier) @class.name
                        body: (interface_body) @class.body) @class.def
                """),
                "javadoc": self.languages["java"].query("""
                    (block_comment) @javadoc
                """)
            }

    def parse_file(self, file_path: str) -> Optional[Dict[str, any]]:
        """Parse a file and extract its syntax tree."""
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        language = self._detect_language(file_path)
        if not language or language not in self.parsers:
            return None
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            parser = self.parsers[language]
            tree = parser.parse(content)
            
            # Check if tree has parsing errors
            if tree.root_node.has_error:
                print(f"Warning: Parsing errors in {file_path}")
            
            return {
                "file": str(file_path),
                "language": language,
                "tree": tree,
                "content": content.decode('utf-8', errors='replace')
            }
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file extension for supported languages."""
        ext_map = {
            ".py": "python",
            ".java": "java",
        }
        
        return ext_map.get(file_path.suffix.lower())

    def extract_functions(self, file_path: str) -> List[Dict[str, any]]:
        """Extract all function definitions from a file using tree-sitter queries."""
        parsed = self.parse_file(file_path)
        if not parsed:
            return []
        
        language = parsed["language"]
        tree = parsed["tree"]
        content = parsed["content"]
        lines = content.split('\n')
        
        functions = []
        
        # Use queries for precise extraction
        if language in self.queries and "functions" in self.queries[language]:
            query = self.queries[language]["functions"]
            captures = query.captures(tree.root_node)
            
            # Group captures by function definition
            function_groups = {}
            capture_dict = {}
            
            # Convert captures to a more usable format
            for capture_name, nodes in captures.items():
                capture_dict[capture_name] = nodes
            
            # Process function definitions first
            if "function.def" in capture_dict:
                for node in capture_dict["function.def"]:
                    function_groups[id(node)] = {"def": node, "name": None, "params": None, "body": None}
            
            # Then process other captures
            for capture_name, nodes in capture_dict.items():
                if capture_name == "function.name":
                    for node in nodes:
                        parent = self._find_parent_function_def(node, capture_dict)
                        if parent and id(parent) in function_groups:
                            function_groups[id(parent)]["name"] = node
                elif capture_name == "function.params":
                    for node in nodes:
                        parent = self._find_parent_function_def(node, capture_dict)
                        if parent and id(parent) in function_groups:
                            function_groups[id(parent)]["params"] = node
                elif capture_name == "function.body":
                    for node in nodes:
                        parent = self._find_parent_function_def(node, capture_dict)
                        if parent and id(parent) in function_groups:
                            function_groups[id(parent)]["body"] = node
            
            # Extract function information
            for func_id, func_data in function_groups.items():
                if func_data["def"]:
                    func_info = self._extract_function_info(func_data, lines, language)
                    if func_info:
                        functions.append(func_info)
        
        return functions
    
    def _find_parent_function_def(self, node: Node, capture_dict: Dict) -> Optional[Node]:
        """Find the parent function definition node for a given node."""
        if "function.def" in capture_dict:
            for capture_node in capture_dict["function.def"]:
                if self._is_ancestor(capture_node, node):
                    return capture_node
        return None
    
    def _is_ancestor(self, ancestor: Node, descendant: Node) -> bool:
        """Check if ancestor is an ancestor of descendant."""
        current = descendant.parent
        while current:
            if current == ancestor:
                return True
            current = current.parent
        return False
    
    def _extract_function_info(self, func_data: Dict, lines: List[str], language: str) -> Optional[Dict[str, any]]:
        """Extract function information from captured nodes."""
        try:
            def_node = func_data["def"]
            name_node = func_data["name"]
            
            # Get function name
            name = "anonymous"
            if name_node:
                name = name_node.text.decode('utf-8')
            elif language == "javascript" and def_node.type == "arrow_function":
                # For arrow functions, try to get name from assignment
                parent = def_node.parent
                if parent and parent.type == "variable_declarator":
                    name_child = parent.child_by_field_name("name")
                    if name_child:
                        name = name_child.text.decode('utf-8')
            
            # Determine function type
            func_type = "function"
            if def_node.type == "method_definition":
                func_type = "method"
            elif def_node.type == "constructor_declaration":
                func_type = "constructor"
            elif def_node.type == "arrow_function":
                func_type = "arrow_function"
            
            return {
                "name": name,
                "type": func_type,
                "start_line": def_node.start_point[0] + 1,
                "end_line": def_node.end_point[0] + 1,
                "code": "\n".join(lines[def_node.start_point[0]:def_node.end_point[0] + 1])
            }
        except Exception as e:
            return None

    def extract_classes(self, file_path: str) -> List[Dict[str, any]]:
        """Extract all class definitions from a file using tree-sitter queries."""
        parsed = self.parse_file(file_path)
        if not parsed:
            return []
        
        language = parsed["language"]
        tree = parsed["tree"]
        content = parsed["content"]
        lines = content.split('\n')
        
        classes = []
        
        # Use queries for precise extraction
        if language in self.queries and "classes" in self.queries[language]:
            query = self.queries[language]["classes"]
            captures = query.captures(tree.root_node)
            
            # Group captures by class definition
            class_groups = {}
            capture_dict = {}
            
            # Convert captures to a more usable format
            for capture_name, nodes in captures.items():
                capture_dict[capture_name] = nodes
            
            # Process class definitions first
            if "class.def" in capture_dict:
                for node in capture_dict["class.def"]:
                    class_groups[id(node)] = {"def": node, "name": None, "body": None}
            
            # Then process other captures
            for capture_name, nodes in capture_dict.items():
                if capture_name == "class.name":
                    for node in nodes:
                        parent = self._find_parent_class_def(node, capture_dict)
                        if parent and id(parent) in class_groups:
                            class_groups[id(parent)]["name"] = node
                elif capture_name == "class.body":
                    for node in nodes:
                        parent = self._find_parent_class_def(node, capture_dict)
                        if parent and id(parent) in class_groups:
                            class_groups[id(parent)]["body"] = node
            
            # Extract class information
            for class_id, class_data in class_groups.items():
                if class_data["def"]:
                    class_info = self._extract_class_info(class_data, lines)
                    if class_info:
                        classes.append(class_info)
        
        return classes
    
    def _find_parent_class_def(self, node: Node, capture_dict: Dict) -> Optional[Node]:
        """Find the parent class definition node for a given node."""
        if "class.def" in capture_dict:
            for capture_node in capture_dict["class.def"]:
                if self._is_ancestor(capture_node, node):
                    return capture_node
        return None
    
    def _extract_class_info(self, class_data: Dict, lines: List[str]) -> Optional[Dict[str, any]]:
        """Extract class information from captured nodes."""
        try:
            def_node = class_data["def"]
            name_node = class_data["name"]
            
            # Get class name
            name = "anonymous"
            if name_node:
                name = name_node.text.decode('utf-8')
            
            return {
                "name": name,
                "start_line": def_node.start_point[0] + 1,
                "end_line": def_node.end_point[0] + 1,
                "code": "\n".join(lines[def_node.start_point[0]:def_node.end_point[0] + 1])
            }
        except Exception as e:
            return None

    def get_node_at_position(self, file_path: str, line: int, column: int) -> Optional[Dict[str, any]]:
        """Get the syntax node at a specific position."""
        parsed = self.parse_file(file_path)
        if not parsed:
            return None
        
        tree = parsed["tree"]
        point = (line - 1, column - 1)  # Convert to 0-based
        
        node = tree.root_node.descendant_for_point_range(point, point)
        if not node:
            return None
        
        return {
            "type": node.type,
            "text": node.text.decode('utf-8'),
            "start": {"line": node.start_point[0] + 1, "column": node.start_point[1] + 1},
            "end": {"line": node.end_point[0] + 1, "column": node.end_point[1] + 1},
            "parent": node.parent.type if node.parent else None
        }
    
    def traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
        """Traverse all nodes in a tree using a cursor for efficiency."""
        cursor = tree.walk()
        
        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

    def extract_docstrings(self, file_path: str) -> List[Dict[str, any]]:
        """Extract docstrings from Python files or Javadoc from Java files."""
        parsed = self.parse_file(file_path)
        if not parsed:
            return []
        
        language = parsed["language"]
        tree = parsed["tree"]
        content = parsed["content"]
        lines = content.split('\n')
        
        docstrings = []
        
        if language == "python" and "docstrings" in self.queries.get(language, {}):
            query = self.queries[language]["docstrings"]
            captures = query.captures(tree.root_node)
            
            for capture_name, nodes in captures.items():
                for node in nodes:
                    docstring_text = node.text.decode('utf-8').strip('"\'')
                    docstrings.append({
                        "type": capture_name.replace("_doc_str", ""),
                        "content": docstring_text,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1
                    })
        
        elif language == "java" and "javadoc" in self.queries.get(language, {}):
            query = self.queries[language]["javadoc"]
            captures = query.captures(tree.root_node)
            
            for capture_name, nodes in captures.items():
                for node in nodes:
                    comment_text = node.text.decode('utf-8')
                    if comment_text.startswith("/**"):
                        docstrings.append({
                            "type": "javadoc",
                            "content": comment_text,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1
                        })
        
        return docstrings

    def query_code_patterns(self, file_path: str, query_string: str) -> List[Dict[str, any]]:
        """Execute a custom tree-sitter query on a file."""
        parsed = self.parse_file(file_path)
        if not parsed:
            return []
        
        language = parsed["language"]
        tree = parsed["tree"]
        
        try:
            query = self.languages[language].query(query_string)
            captures = query.captures(tree.root_node)
            
            results = []
            for capture_name, nodes in captures.items():
                for node in nodes:
                    results.append({
                        "capture": capture_name,
                        "text": node.text.decode('utf-8'),
                        "type": node.type,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "start_column": node.start_point[1] + 1,
                        "end_column": node.end_point[1] + 1
                    })
            
            return results
        except Exception as e:
            print(f"Error executing query: {e}")
            return []