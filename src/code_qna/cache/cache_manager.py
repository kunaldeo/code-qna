"""Simple caching system for Code Q&A."""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pickle


class CacheManager:
    """Manage caching of search results and context extractions."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_age: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files (default: ~/.cache/code-qna)
            max_age: Maximum age of cache entries in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "code-qna" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        
        # Initialize cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"entries": {}, "stats": {"hits": 0, "misses": 0, "created": time.time()}}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache metadata: {e}")
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data."""
        # Create a stable hash from the data
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.cache"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self.metadata["entries"]:
            return False
        
        entry = self.metadata["entries"][key]
        age = time.time() - entry["timestamp"]
        return age < self.max_age
    
    def get_search_results(self, query: str, path: str, patterns: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        cache_data = {
            "type": "search",
            "query": query,
            "path": str(Path(path).resolve()),
            "patterns": sorted(patterns)
        }
        
        key = self._get_cache_key(cache_data)
        
        if not self._is_cache_valid(key):
            self.metadata["stats"]["misses"] += 1
            return None
        
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            self.metadata["stats"]["misses"] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            
            self.metadata["stats"]["hits"] += 1
            self._save_metadata()
            return results
        except Exception:
            self.metadata["stats"]["misses"] += 1
            return None
    
    def cache_search_results(self, query: str, path: str, patterns: List[str], results: List[Dict[str, Any]]):
        """Cache search results."""
        cache_data = {
            "type": "search",
            "query": query,
            "path": str(Path(path).resolve()),
            "patterns": sorted(patterns)
        }
        
        key = self._get_cache_key(cache_data)
        cache_file = self._get_cache_path(key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "timestamp": time.time(),
                "type": "search",
                "size": len(results)
            }
            self._save_metadata()
        except Exception as e:
            print(f"Warning: Could not cache search results: {e}")
    
    def get_context(self, question: str, path: str, file_timestamps: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Get cached context extraction."""
        cache_data = {
            "type": "context",
            "question": question,
            "path": str(Path(path).resolve()),
            "file_timestamps": file_timestamps
        }
        
        key = self._get_cache_key(cache_data)
        
        if not self._is_cache_valid(key):
            self.metadata["stats"]["misses"] += 1
            return None
        
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            self.metadata["stats"]["misses"] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                context = pickle.load(f)
            
            self.metadata["stats"]["hits"] += 1
            self._save_metadata()
            return context
        except Exception:
            self.metadata["stats"]["misses"] += 1
            return None
    
    def cache_context(self, question: str, path: str, file_timestamps: Dict[str, float], context: Dict[str, Any]):
        """Cache context extraction."""
        cache_data = {
            "type": "context",
            "question": question,
            "path": str(Path(path).resolve()),
            "file_timestamps": file_timestamps
        }
        
        key = self._get_cache_key(cache_data)
        cache_file = self._get_cache_path(key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(context, f)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "timestamp": time.time(),
                "type": "context",
                "size": len(str(context))
            }
            self._save_metadata()
        except Exception as e:
            print(f"Warning: Could not cache context: {e}")
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache entries."""
        if cache_type:
            # Clear specific type
            to_remove = []
            for key, entry in self.metadata["entries"].items():
                if entry.get("type") == cache_type:
                    cache_file = self._get_cache_path(key)
                    cache_file.unlink(missing_ok=True)
                    to_remove.append(key)
            
            for key in to_remove:
                del self.metadata["entries"][key]
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            self.metadata["entries"] = {}
        
        self._save_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data by key."""
        if not self._is_cache_valid(key):
            self.metadata["stats"]["misses"] += 1
            return None
        
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            self.metadata["stats"]["misses"] += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.metadata["stats"]["hits"] += 1
            self._save_metadata()
            return data
        except Exception:
            self.metadata["stats"]["misses"] += 1
            return None
    
    def set(self, key: str, data: Any):
        """Cache data by key."""
        cache_file = self._get_cache_path(key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "timestamp": time.time(),
                "type": "generic",
                "size": len(str(data))
            }
            self._save_metadata()
        except Exception as e:
            print(f"Warning: Could not cache data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.metadata["stats"]["hits"] + self.metadata["stats"]["misses"]
        hit_rate = (self.metadata["stats"]["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        # Count cache files
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "hits": self.metadata["stats"]["hits"],
            "misses": self.metadata["stats"]["misses"],
            "entries": len(self.metadata["entries"]),
            "total_size_mb": total_size / (1024 * 1024),
            "created": self.metadata["stats"]["created"]
        }
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        to_remove = []
        
        for key, entry in self.metadata["entries"].items():
            age = current_time - entry["timestamp"]
            if age > self.max_age:
                cache_file = self._get_cache_path(key)
                cache_file.unlink(missing_ok=True)
                to_remove.append(key)
        
        for key in to_remove:
            del self.metadata["entries"][key]
        
        if to_remove:
            self._save_metadata()
        
        return len(to_remove)


# Global cache instance
cache_manager = CacheManager()