"""Repository state management for PR-based training."""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import tempfile


@dataclass
class FileState:
    """State of a single file."""
    path: str
    content: str
    exists: bool = True


@dataclass
class RepoSnapshot:
    """Snapshot of repository state."""
    files: Dict[str, FileState]
    applied_prs: List[str]
    
    def get_file(self, path: str) -> Optional[str]:
        """Get file content by path."""
        if path in self.files and self.files[path].exists:
            return self.files[path].content
        return None
    
    def list_files(self) -> List[str]:
        """List all file paths."""
        return [p for p, f in self.files.items() if f.exists]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'files': {p: {'content': f.content, 'exists': f.exists} 
                     for p, f in self.files.items()},
            'applied_prs': self.applied_prs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepoSnapshot':
        """Create from dictionary."""
        files = {
            p: FileState(path=p, content=d['content'], exists=d['exists'])
            for p, d in data['files'].items()
        }
        return cls(files=files, applied_prs=data['applied_prs'])


class RepoStateManager:
    """Manages repository state across PR applications.
    
    Handles:
    - Loading base repository state
    - Applying ground truth PRs for dependencies
    - Creating working copies for model to modify
    - Comparing states for reward computation
    """
    
    def __init__(self, base_repo_path: Path, pr_data_path: Path, cache_path: Path):
        """Initialize state manager.
        
        Args:
            base_repo_path: Path to base repository (before any PRs)
            pr_data_path: Path to PR definitions
            cache_path: Path for temporary files
        """
        self.base_repo_path = Path(base_repo_path)
        self.pr_data_path = Path(pr_data_path)
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load base state
        self.base_state = self._load_base_state()
        
        # Cache for PR data
        self._pr_cache: Dict[str, Dict] = {}
        
        # Cache for computed states (state after applying certain PRs)
        self._state_cache: Dict[str, RepoSnapshot] = {}
    
    def _load_base_state(self) -> RepoSnapshot:
        """Load the base repository state."""
        files = {}
        
        for file_path in self.base_repo_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                rel_path = str(file_path.relative_to(self.base_repo_path))
                try:
                    content = file_path.read_text()
                    files[rel_path] = FileState(path=rel_path, content=content)
                except UnicodeDecodeError:
                    # Skip binary files
                    pass
        
        return RepoSnapshot(files=files, applied_prs=[])
    
    def _load_pr(self, pr_id: str) -> Dict:
        """Load PR data from JSON file."""
        if pr_id not in self._pr_cache:
            pr_file = self.pr_data_path / f"{pr_id.lower().replace('-', '_')}.json"
            if not pr_file.exists():
                # Try alternate naming
                pr_file = self.pr_data_path / f"pr_{pr_id.split('-')[1]}.json"
            
            with open(pr_file, 'r') as f:
                self._pr_cache[pr_id] = json.load(f)
        
        return self._pr_cache[pr_id]
    
    def _apply_pr_to_state(self, state: RepoSnapshot, pr_id: str) -> RepoSnapshot:
        """Apply a PR's expected changes to a state."""
        pr_data = self._load_pr(pr_id)
        expected_changes = pr_data.get('expected_changes', {})
        
        # Deep copy files
        new_files = {p: FileState(path=p, content=f.content, exists=f.exists) 
                    for p, f in state.files.items()}
        
        for file_path, changes in expected_changes.items():
            action = changes.get('action', 'modify')
            
            if action == 'create':
                # New file
                content = '\n'.join(changes.get('content', []))
                new_files[file_path] = FileState(path=file_path, content=content)
            
            elif action == 'modify':
                # Append additions to existing file
                if file_path in new_files:
                    additions = '\n'.join(changes.get('additions', []))
                    current_content = new_files[file_path].content
                    new_content = current_content.rstrip() + '\n' + additions
                    new_files[file_path] = FileState(
                        path=file_path, 
                        content=new_content
                    )
            
            elif action == 'delete':
                if file_path in new_files:
                    new_files[file_path].exists = False
        
        return RepoSnapshot(
            files=new_files,
            applied_prs=state.applied_prs + [pr_id]
        )
    
    def get_state_for_pr(self, pr_id: str, dependency_prs: List[str]) -> RepoSnapshot:
        """Get repository state with all dependencies applied.
        
        Args:
            pr_id: The PR to prepare state for
            dependency_prs: List of PRs that must be applied first
            
        Returns:
            RepoSnapshot with dependencies applied (but not the target PR)
        """
        # Create cache key
        cache_key = '_'.join(sorted(dependency_prs)) if dependency_prs else 'base'
        
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]
        
        # Start from base state
        state = RepoSnapshot(
            files={p: FileState(path=p, content=f.content, exists=f.exists)
                  for p, f in self.base_state.files.items()},
            applied_prs=[]
        )
        
        # Apply each dependency PR in order
        for dep_pr in dependency_prs:
            state = self._apply_pr_to_state(state, dep_pr)
        
        self._state_cache[cache_key] = state
        return state
    
    def get_expected_state_after_pr(self, pr_id: str, dependency_prs: List[str]) -> RepoSnapshot:
        """Get expected state after PR is correctly applied.
        
        Args:
            pr_id: The PR to apply
            dependency_prs: Dependencies already applied
            
        Returns:
            RepoSnapshot with PR applied (ground truth)
        """
        base_state = self.get_state_for_pr(pr_id, dependency_prs)
        return self._apply_pr_to_state(base_state, pr_id)
    
    def create_working_directory(self, state: RepoSnapshot) -> Path:
        """Create a temporary working directory from state.
        
        Args:
            state: Repository state to materialize
            
        Returns:
            Path to temporary directory
        """
        work_dir = self.cache_path / f"work_{id(state)}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        
        for file_path, file_state in state.files.items():
            if file_state.exists:
                full_path = work_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(file_state.content)
        
        return work_dir
    
    def state_from_directory(self, directory: Path, applied_prs: List[str]) -> RepoSnapshot:
        """Create state snapshot from a directory.
        
        Args:
            directory: Directory to snapshot
            applied_prs: List of applied PRs for tracking
            
        Returns:
            RepoSnapshot of directory contents
        """
        files = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                rel_path = str(file_path.relative_to(directory))
                try:
                    content = file_path.read_text()
                    files[rel_path] = FileState(path=rel_path, content=content)
                except UnicodeDecodeError:
                    pass
        
        return RepoSnapshot(files=files, applied_prs=applied_prs)
    
    def cleanup_work_dir(self, work_dir: Path):
        """Clean up a working directory."""
        if work_dir.exists() and str(work_dir).startswith(str(self.cache_path)):
            shutil.rmtree(work_dir)
    
    def compare_states(self, actual: RepoSnapshot, expected: RepoSnapshot) -> Dict[str, Any]:
        """Compare two repository states.
        
        Returns:
            Dictionary with comparison metrics
        """
        all_paths = set(actual.files.keys()) | set(expected.files.keys())
        
        matching_files = 0
        total_files = 0
        file_diffs = {}
        
        for path in all_paths:
            actual_file = actual.files.get(path)
            expected_file = expected.files.get(path)
            
            actual_exists = actual_file and actual_file.exists
            expected_exists = expected_file and expected_file.exists
            
            if expected_exists:
                total_files += 1
                
                if actual_exists:
                    actual_content = actual_file.content.strip()
                    expected_content = expected_file.content.strip()
                    
                    if actual_content == expected_content:
                        matching_files += 1
                    else:
                        file_diffs[path] = {
                            'type': 'content_mismatch',
                            'expected_lines': len(expected_content.split('\n')),
                            'actual_lines': len(actual_content.split('\n'))
                        }
                else:
                    file_diffs[path] = {'type': 'missing'}
            elif actual_exists:
                file_diffs[path] = {'type': 'extra'}
        
        return {
            'matching_files': matching_files,
            'total_expected_files': total_files,
            'file_match_ratio': matching_files / total_files if total_files > 0 else 1.0,
            'file_diffs': file_diffs,
            'exact_match': matching_files == total_files and not file_diffs
        }
