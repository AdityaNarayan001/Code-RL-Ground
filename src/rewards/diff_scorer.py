"""Diff scoring between expected and actual code."""

import difflib
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class DiffResult:
    """Result of diff comparison."""
    similarity: float  # 0.0 to 1.0
    matching_lines: int
    total_lines: int
    additions_matched: float  # How many expected additions are present
    diff_text: str  # Human-readable diff


class DiffScorer:
    """Score similarity between expected and actual code."""
    
    def __init__(self):
        self.differ = difflib.Differ()
    
    def score_files(
        self,
        actual_files: Dict[str, str],
        expected_files: Dict[str, str]
    ) -> Dict[str, DiffResult]:
        """Score similarity for multiple files.
        
        Args:
            actual_files: Dictionary of {filepath: content} for actual
            expected_files: Dictionary of {filepath: content} for expected
            
        Returns:
            Dictionary of {filepath: DiffResult}
        """
        results = {}
        
        all_paths = set(actual_files.keys()) | set(expected_files.keys())
        
        for path in all_paths:
            actual = actual_files.get(path, "")
            expected = expected_files.get(path, "")
            results[path] = self.score_content(actual, expected)
        
        return results
    
    def score_content(self, actual: str, expected: str) -> DiffResult:
        """Score similarity between two content strings.
        
        Args:
            actual: Actual content
            expected: Expected content
            
        Returns:
            DiffResult with similarity metrics
        """
        # Normalize whitespace
        actual_lines = self._normalize(actual).split('\n')
        expected_lines = self._normalize(expected).split('\n')
        
        # Compute sequence similarity
        matcher = difflib.SequenceMatcher(None, actual_lines, expected_lines)
        similarity = matcher.ratio()
        
        # Count matching lines
        matching = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                matching += (i2 - i1)
        
        # Generate diff text
        diff = list(difflib.unified_diff(
            actual_lines,
            expected_lines,
            fromfile='actual',
            tofile='expected',
            lineterm=''
        ))
        diff_text = '\n'.join(diff)
        
        # Calculate additions matched
        additions_matched = self._score_additions(actual, expected)
        
        return DiffResult(
            similarity=similarity,
            matching_lines=matching,
            total_lines=len(expected_lines),
            additions_matched=additions_matched,
            diff_text=diff_text
        )
    
    def _normalize(self, content: str) -> str:
        """Normalize content for comparison."""
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in content.split('\n')]
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)
    
    def _score_additions(self, actual: str, expected: str) -> float:
        """Score how many expected additions are present in actual.
        
        Focuses on function definitions and key code blocks.
        """
        # Extract function definitions
        actual_funcs = set(re.findall(r'def\s+(\w+)\s*\(', actual))
        expected_funcs = set(re.findall(r'def\s+(\w+)\s*\(', expected))
        
        if not expected_funcs:
            return 1.0 if actual == expected else 0.0
        
        matching_funcs = actual_funcs & expected_funcs
        func_score = len(matching_funcs) / len(expected_funcs)
        
        # Also check for key code patterns
        expected_patterns = self._extract_key_patterns(expected)
        if not expected_patterns:
            return func_score
        
        pattern_matches = sum(1 for p in expected_patterns if p in actual)
        pattern_score = pattern_matches / len(expected_patterns)
        
        return (func_score + pattern_score) / 2
    
    def _extract_key_patterns(self, code: str) -> List[str]:
        """Extract key patterns from code for matching."""
        patterns = []
        
        # Return statements
        returns = re.findall(r'return\s+[^#\n]+', code)
        patterns.extend(returns[:5])  # Limit
        
        # Raise statements
        raises = re.findall(r'raise\s+\w+', code)
        patterns.extend(raises[:3])
        
        # Key operations
        ops = re.findall(r'[\w]+\s*=\s*[^#\n]{5,30}', code)
        patterns.extend(ops[:5])
        
        return patterns
    
    def overall_score(self, file_results: Dict[str, DiffResult]) -> float:
        """Compute overall similarity score across all files.
        
        Args:
            file_results: Dictionary of file results
            
        Returns:
            Overall similarity score (0.0 to 1.0)
        """
        if not file_results:
            return 0.0
        
        # Weight by lines in expected files
        total_weight = 0
        weighted_score = 0
        
        for path, result in file_results.items():
            weight = max(1, result.total_lines)
            total_weight += weight
            weighted_score += result.similarity * weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
