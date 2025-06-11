"""Relevance scoring for code context extraction."""

import re
from typing import Dict, List, Set, Tuple
from pathlib import Path
import math


class RelevanceScorer:
    """Score the relevance of code snippets and files for a given query."""
    
    def __init__(self):
        # Common programming stop words
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "what", "where", "when", "how", "why", "which", "who", "whom", "whose"
        }
        
        # Technical terms that should be weighted higher
        self.technical_terms = {
            "function", "method", "class", "module", "package", "import", "export",
            "interface", "implementation", "algorithm", "data", "structure", "api",
            "service", "controller", "model", "view", "component", "library",
            "framework", "database", "query", "request", "response", "error",
            "exception", "handler", "middleware", "authentication", "authorization",
            "validation", "configuration", "initialization", "constructor", "destructor"
        }
        
        # File path importance weights
        self.path_weights = {
            "src": 1.2,
            "lib": 1.2,
            "core": 1.3,
            "main": 1.1,
            "app": 1.1,
            "test": 0.7,
            "tests": 0.7,
            "spec": 0.7,
            "example": 0.6,
            "examples": 0.6,
            "doc": 0.5,
            "docs": 0.5,
            "vendor": 0.4,
            "node_modules": 0.3,
            "dist": 0.3,
            "build": 0.3
        }
        
        # File extension importance
        self.extension_weights = {
            ".py": 1.0,
            ".js": 1.0,
            ".ts": 1.0,
            ".java": 1.0,
            ".go": 1.0,
            ".rs": 1.0,
            ".cpp": 1.0,
            ".c": 1.0,
            ".cs": 1.0,
            ".rb": 1.0,
            ".php": 1.0,
            ".swift": 1.0,
            ".kt": 1.0,
            ".scala": 1.0,
            ".md": 0.6,
            ".txt": 0.5,
            ".json": 0.7,
            ".yaml": 0.7,
            ".yml": 0.7,
            ".xml": 0.6,
            ".config": 0.7,
            ".ini": 0.6
        }
    
    def extract_query_features(self, query: str) -> Dict[str, any]:
        """Extract features from the query for scoring."""
        # Tokenize query
        tokens = re.findall(r'\b\w+\b', query.lower())
        
        # Remove stop words
        meaningful_tokens = [t for t in tokens if t not in self.stop_words]
        
        # Identify technical terms
        technical_tokens = [t for t in meaningful_tokens if t in self.technical_terms]
        
        # Extract potential identifiers (camelCase, snake_case, etc.)
        identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', query)
        
        # Extract quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        
        return {
            "tokens": meaningful_tokens,
            "technical_tokens": technical_tokens,
            "identifiers": identifiers,
            "quoted": quoted,
            "token_count": len(meaningful_tokens)
        }
    
    def score_file_relevance(self, file_path: str, query_features: Dict[str, any],
                           search_matches: List[Dict[str, any]] = None) -> float:
        """Score the relevance of a file based on various factors."""
        path = Path(file_path)
        score = 1.0
        
        # 1. File path relevance
        path_parts = path.parts
        for part in path_parts:
            part_lower = part.lower()
            for key, weight in self.path_weights.items():
                if key in part_lower:
                    score *= weight
        
        # 2. File extension relevance
        extension = path.suffix.lower()
        if extension in self.extension_weights:
            score *= self.extension_weights[extension]
        
        # 3. Filename relevance
        filename = path.stem.lower()
        for token in query_features["tokens"]:
            if token in filename:
                score *= 1.5
        
        # 4. Search match quality
        if search_matches:
            # More matches = higher relevance
            match_count = len(search_matches)
            score *= (1 + math.log(match_count + 1) * 0.3)
            
            # Consider match density (matches per line)
            if match_count > 0:
                unique_lines = len(set(m["line_number"] for m in search_matches))
                density = match_count / unique_lines
                score *= (1 + density * 0.2)
        
        return score
    
    def score_code_snippet(self, snippet: Dict[str, any], query_features: Dict[str, any]) -> float:
        """Score a code snippet based on its relevance to the query."""
        score = 1.0
        
        # Get the code content
        code = snippet.get("code", snippet.get("content", ""))
        code_lower = code.lower()
        
        # 1. Direct token matches
        for token in query_features["tokens"]:
            count = code_lower.count(token.lower())
            if count > 0:
                score *= (1 + math.log(count + 1) * 0.2)
        
        # 2. Technical term matches
        for term in query_features["technical_tokens"]:
            if term in code_lower:
                score *= 1.3
        
        # 3. Identifier matches (case-sensitive)
        for identifier in query_features["identifiers"]:
            if identifier in code:
                score *= 2.0  # Exact identifier match is very relevant
        
        # 4. Quoted string matches
        for quoted in query_features["quoted"]:
            if quoted in code:
                score *= 2.5  # Quoted strings are usually very specific
        
        # 5. Snippet type relevance
        snippet_type = snippet.get("type", "").lower()
        if snippet_type in ["function", "method", "class"]:
            score *= 1.2
        
        # 6. Line proximity (if we have match line numbers)
        if "line_number" in snippet and "matches" in snippet:
            # Check how close the matches are to each other
            match_lines = [m.get("line_number", 0) for m in snippet["matches"]]
            if len(match_lines) > 1:
                # Calculate average distance between matches
                distances = []
                for i in range(1, len(match_lines)):
                    distances.append(abs(match_lines[i] - match_lines[i-1]))
                avg_distance = sum(distances) / len(distances)
                # Closer matches = higher relevance
                proximity_score = 1 / (1 + avg_distance * 0.1)
                score *= (1 + proximity_score * 0.3)
        
        return score
    
    def rank_results(self, results: List[Dict[str, any]], query: str,
                    max_results: int = None) -> List[Dict[str, any]]:
        """Rank search results by relevance."""
        query_features = self.extract_query_features(query)
        
        # Score each result
        for result in results:
            file_score = self.score_file_relevance(
                result.get("file", ""),
                query_features,
                [result]  # Pass the result as a match
            )
            
            snippet_score = self.score_code_snippet(result, query_features)
            
            # Combine scores
            result["relevance_score"] = file_score * snippet_score
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Return top results
        if max_results:
            return results[:max_results]
        return results
    
    def group_results_by_file(self, results: List[Dict[str, any]]) -> Dict[str, List[Dict[str, any]]]:
        """Group search results by file."""
        grouped = {}
        for result in results:
            file_path = result.get("file", "")
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].append(result)
        return grouped
    
    def calculate_file_importance(self, file_path: str, grouped_results: Dict[str, List[Dict[str, any]]],
                                query_features: Dict[str, any]) -> float:
        """Calculate overall importance of a file based on all matches."""
        matches = grouped_results.get(file_path, [])
        
        # Base file score
        file_score = self.score_file_relevance(file_path, query_features, matches)
        
        # Aggregate match scores
        if matches:
            match_scores = [m.get("relevance_score", 1.0) for m in matches]
            # Use a combination of sum and max to balance quantity and quality
            aggregate_score = (sum(match_scores) * 0.3 + max(match_scores) * 0.7)
            return file_score * aggregate_score
        
        return file_score