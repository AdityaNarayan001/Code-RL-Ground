"""PR task loader."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class PRTask:
    """A PR task for training."""
    pr_id: str
    title: str
    description: str
    difficulty: int
    files_changed: List[str]
    depends_on: List[str]
    data: Dict[str, Any]
    expected_changes: Dict[str, Any] = field(default_factory=dict)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PRTask':
        """Create from JSON data."""
        return cls(
            pr_id=data['pr_id'],
            title=data['title'],
            description=data['description'],
            difficulty=data.get('difficulty', 1),
            files_changed=data.get('files_changed', []),
            depends_on=data.get('depends_on', []),
            data=data,
            expected_changes=data.get('expected_changes', {}),
            test_cases=data.get('test_cases', [])
        )


class PRLoader:
    """Load PR tasks from dataset."""
    
    def __init__(self, dataset_path: Path):
        """Initialize loader.
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.index_path = self.dataset_path / "index.json"
        self.prs_path = self.dataset_path / "prs"
        
        self._index: Optional[Dict[str, Any]] = None
        self._tasks: Dict[str, PRTask] = {}
    
    def load_index(self) -> Dict[str, Any]:
        """Load dataset index."""
        if self._index is None:
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)
        return self._index
    
    def get_all_pr_ids(self) -> List[str]:
        """Get all PR IDs in the dataset."""
        index = self.load_index()
        return [pr['id'] for pr in index['prs']]
    
    def get_topological_order(self) -> List[str]:
        """Get PRs in topological order (respecting dependencies)."""
        index = self.load_index()
        return index.get('topological_order', self.get_all_pr_ids())
    
    def get_curriculum_order(self) -> List[str]:
        """Get PRs in curriculum order."""
        index = self.load_index()
        return index.get('curriculum_order', self.get_topological_order())
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph."""
        index = self.load_index()
        return index.get('dependency_graph', {})
    
    def load_pr(self, pr_id: str) -> PRTask:
        """Load a specific PR task.
        
        Args:
            pr_id: PR ID (e.g., 'PR-001')
            
        Returns:
            PRTask object
        """
        if pr_id in self._tasks:
            return self._tasks[pr_id]
        
        # Find PR file
        pr_file = self.prs_path / f"pr_{pr_id.split('-')[1]}.json"
        
        if not pr_file.exists():
            # Try alternate naming
            for f in self.prs_path.glob("*.json"):
                with open(f, 'r') as fp:
                    data = json.load(fp)
                if data.get('pr_id') == pr_id:
                    pr_file = f
                    break
        
        if not pr_file.exists():
            raise FileNotFoundError(f"PR not found: {pr_id}")
        
        with open(pr_file, 'r') as f:
            data = json.load(f)
        
        task = PRTask.from_json(data)
        self._tasks[pr_id] = task
        return task
    
    def load_all(self) -> List[PRTask]:
        """Load all PR tasks.
        
        Returns:
            List of all PRTask objects
        """
        tasks = []
        for pr_id in self.get_all_pr_ids():
            tasks.append(self.load_pr(pr_id))
        return tasks
    
    def load_in_curriculum_order(self) -> List[PRTask]:
        """Load tasks in curriculum order.
        
        Returns:
            List of PRTask objects in curriculum order
        """
        tasks = []
        for pr_id in self.get_curriculum_order():
            tasks.append(self.load_pr(pr_id))
        return tasks
    
    def get_dependencies(self, pr_id: str) -> List[str]:
        """Get dependencies for a PR.
        
        Args:
            pr_id: PR ID
            
        Returns:
            List of dependency PR IDs
        """
        graph = self.get_dependency_graph()
        return graph.get(pr_id, [])
    
    def get_all_dependencies(self, pr_id: str) -> List[str]:
        """Get all transitive dependencies for a PR.
        
        Args:
            pr_id: PR ID
            
        Returns:
            List of all dependency PR IDs in order
        """
        graph = self.get_dependency_graph()
        
        def get_deps_recursive(pid: str, visited: set) -> List[str]:
            if pid in visited:
                return []
            visited.add(pid)
            
            result = []
            for dep in graph.get(pid, []):
                result.extend(get_deps_recursive(dep, visited))
                result.append(dep)
            return result
        
        deps = get_deps_recursive(pr_id, set())
        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for d in deps:
            if d not in seen:
                seen.add(d)
                unique_deps.append(d)
        return unique_deps
