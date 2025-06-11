import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class NodeType(Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    CALL = "call"
    PARAMETER = "parameter"
    RETURN_TYPE = "return_type"

class EdgeType(Enum):
    CONTAINS = "contains"          # File contains function/class
    CALLS = "calls"               # Function calls another function
    IMPORTS = "imports"           # File imports another file/module
    INHERITS = "inherits"         # Class inherits from another
    REFERENCES = "references"     # Variable/function references another
    DEFINES = "defines"           # Function defines a variable
    USES = "uses"                # Function uses a variable
    IMPLEMENTS = "implements"     # Class implements interface
    DEPENDS_ON = "depends_on"     # General dependency

@dataclass
class CodeNode:
    id: str
    type: NodeType
    name: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class CodeEdge:
    source: str
    target: str
    type: EdgeType
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SemanticGraph:
    """A semantic graph representing the structure and relationships in a codebase."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, CodeNode] = {}
        self.edges: List[CodeEdge] = []
        self._index_by_type: Dict[NodeType, Set[str]] = {nt: set() for nt in NodeType}
        self._index_by_file: Dict[str, Set[str]] = {}
        
    def add_node(self, node: CodeNode) -> str:
        """Add a node to the graph and return its ID."""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)
        
        # Update indices
        self._index_by_type[node.type].add(node.id)
        if node.file_path:
            if node.file_path not in self._index_by_file:
                self._index_by_file[node.file_path] = set()
            self._index_by_file[node.file_path].add(node.id)
            
        return node.id
    
    def add_edge(self, edge: CodeEdge):
        """Add an edge to the graph."""
        if edge.source in self.nodes and edge.target in self.nodes:
            self.edges.append(edge)
            self.graph.add_edge(edge.source, edge.target, type=edge.type, **edge.metadata)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[CodeNode]:
        """Get all nodes of a specific type."""
        return [self.nodes[node_id] for node_id in self._index_by_type[node_type]]
    
    def get_nodes_in_file(self, file_path: str) -> List[CodeNode]:
        """Get all nodes defined in a specific file."""
        node_ids = self._index_by_file.get(file_path, set())
        return [self.nodes[node_id] for node_id in node_ids]
    
    def find_node(self, name: str, node_type: Optional[NodeType] = None, 
                  file_path: Optional[str] = None) -> List[CodeNode]:
        """Find nodes by name, optionally filtered by type and file."""
        results = []
        for node in self.nodes.values():
            if node.name == name:
                if node_type is None or node.type == node_type:
                    if file_path is None or node.file_path == file_path:
                        results.append(node)
        return results
    
    def get_dependencies(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[str]:
        """Get all nodes that this node depends on."""
        if node_id not in self.graph:
            return []
        
        dependencies = []
        for target in self.graph.successors(node_id):
            edges = self.graph[node_id][target]
            for edge_data in edges.values():
                if edge_types is None or EdgeType(edge_data['type']) in edge_types:
                    dependencies.append(target)
                    break
        
        return dependencies
    
    def get_dependents(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[str]:
        """Get all nodes that depend on this node."""
        if node_id not in self.graph:
            return []
        
        dependents = []
        for source in self.graph.predecessors(node_id):
            edges = self.graph[source][node_id]
            for edge_data in edges.values():
                if edge_types is None or EdgeType(edge_data['type']) in edge_types:
                    dependents.append(source)
                    break
        
        return dependents
    
    def find_call_chains(self, start_node: str, max_depth: int = 5) -> List[List[str]]:
        """Find function call chains starting from a given node."""
        if start_node not in self.graph:
            return []
        
        chains = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth >= max_depth:
                return
            
            for target in self.graph.successors(current):
                edges = self.graph[current][target]
                for edge_data in edges.values():
                    if EdgeType(edge_data['type']) == EdgeType.CALLS:
                        new_path = path + [target]
                        chains.append(new_path)
                        dfs(target, new_path, depth + 1)
                        break
        
        dfs(start_node, [start_node], 0)
        return chains
    
    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def find_related_nodes(self, node_id: str, max_distance: int = 2) -> Dict[str, int]:
        """Find all nodes within a certain distance and their distances."""
        if node_id not in self.graph:
            return {}
        
        distances = {}
        visited = {node_id}
        queue = [(node_id, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if dist >= max_distance:
                continue
                
            # Check both predecessors and successors
            neighbors = set(self.graph.predecessors(current)) | set(self.graph.successors(current))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        return distances
    
    def get_connected_components(self) -> List[Set[str]]:
        """Get all connected components in the graph."""
        return [set(component) for component in nx.weakly_connected_components(self.graph)]
    
    def analyze_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality for all nodes."""
        return nx.betweenness_centrality(self.graph)
    
    def export_to_json(self, file_path: str):
        """Export the graph to JSON format."""
        data = {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type.value,
                    'name': node.name,
                    'file_path': node.file_path,
                    'line_start': node.line_start,
                    'line_end': node.line_end,
                    'metadata': node.metadata
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.type.value,
                    'metadata': edge.metadata
                }
                for edge in self.edges
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, file_path: str):
        """Import the graph from JSON format."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        self._index_by_type = {nt: set() for nt in NodeType}
        self._index_by_file.clear()
        
        # Add nodes
        for node_data in data['nodes']:
            node = CodeNode(
                id=node_data['id'],
                type=NodeType(node_data['type']),
                name=node_data['name'],
                file_path=node_data.get('file_path'),
                line_start=node_data.get('line_start'),
                line_end=node_data.get('line_end'),
                metadata=node_data.get('metadata', {})
            )
            self.add_node(node)
        
        # Add edges
        for edge_data in data['edges']:
            edge = CodeEdge(
                source=edge_data['source'],
                target=edge_data['target'],
                type=EdgeType(edge_data['type']),
                metadata=edge_data.get('metadata', {})
            )
            self.add_edge(edge)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the graph."""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'nodes_by_type': {nt.value: len(nodes) for nt, nodes in self._index_by_type.items()},
            'files_analyzed': len(self._index_by_file),
            'connected_components': len(self.get_connected_components()),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0
        }