"""Composite reward function for code generation."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from .syntax_checker import SyntaxChecker
from .diff_scorer import DiffScorer
from .test_runner import TestRunner
from ..utils.config import Config, RewardsConfig
from ..utils.repo_state import RepoSnapshot


@dataclass
class RewardResult:
    """Result of reward computation."""
    total: float
    breakdown: Dict[str, float]
    errors: List[str] = field(default_factory=list)
    test_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Clamp total to [0, 1]
        self.total = max(0.0, min(1.0, self.total))


class RewardFunction:
    """Compute rewards for code generation attempts.
    
    Combines multiple signals:
    - Syntax validity
    - Code compilation
    - Test passing
    - File matching
    - Exact match bonus
    """
    
    def __init__(self, config: Config):
        """Initialize reward function.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.reward_config = config.rewards
        self.weights = config.rewards.weights
        
        # Initialize components
        self.syntax_checker = SyntaxChecker()
        self.diff_scorer = DiffScorer()
        self.test_runner = TestRunner(
            timeout=config.environment.sandbox.timeout_per_execution
        )
    
    def compute(
        self,
        actual_state: RepoSnapshot,
        expected_state: RepoSnapshot,
        pr_data: Dict[str, Any],
        working_dir: Optional[Path] = None
    ) -> RewardResult:
        """Compute reward for a code generation attempt.
        
        Args:
            actual_state: The state produced by the model
            expected_state: The expected state (ground truth)
            pr_data: PR task definition
            working_dir: Working directory for test execution
            
        Returns:
            RewardResult with total reward and breakdown
        """
        breakdown = {}
        errors = []
        test_results = None
        
        # Get files from states
        actual_files = {
            path: fs.content 
            for path, fs in actual_state.files.items() 
            if fs.exists
        }
        expected_files = {
            path: fs.content 
            for path, fs in expected_state.files.items() 
            if fs.exists
        }
        
        # 1. Syntax validity
        syntax_valid, syntax_errors = self.syntax_checker.all_valid(actual_files)
        breakdown['syntax_valid'] = self.weights.syntax_valid if syntax_valid else 0.0
        if not syntax_valid:
            errors.extend(syntax_errors)
        
        # 2. Compilation check (for Python, same as syntax)
        # In future, could add import resolution checking
        compiles = syntax_valid
        breakdown['compiles'] = self.weights.compiles if compiles else 0.0
        
        # 3. Test passing
        test_cases = pr_data.get('test_cases', [])
        if test_cases and working_dir:
            tr = self.test_runner.run_test_cases(
                code_files=actual_files,
                test_cases=test_cases,
                working_dir=working_dir
            )
            test_results = {
                'passed': tr.passed,
                'failed': tr.failed,
                'total': tr.total,
                'all_passed': tr.all_passed
            }
            
            if tr.total > 0:
                test_score = tr.passed / tr.total
                breakdown['tests_pass'] = self.weights.tests_pass * test_score
            else:
                breakdown['tests_pass'] = self.weights.tests_pass
            
            if not tr.all_passed:
                for detail in tr.details:
                    if detail.get('status') != 'passed':
                        errors.append(f"Test {detail.get('name', '?')}: {detail.get('error', 'failed')[:100]}")
        else:
            # No tests defined
            breakdown['tests_pass'] = self.weights.tests_pass * 0.5  # Partial credit
        
        # 4. File matching
        # Compare expected changes
        files_changed = pr_data.get('files_changed', [])
        if files_changed:
            match_score = self._compute_file_match(
                actual_files, 
                expected_files, 
                files_changed
            )
            breakdown['files_match'] = self.weights.files_match * match_score
        else:
            breakdown['files_match'] = 0.0
        
        # 5. Exact match bonus
        is_exact = self._check_exact_match(actual_files, expected_files, files_changed)
        breakdown['exact_match'] = self.weights.exact_match if is_exact else 0.0
        
        # Compute total
        total = sum(breakdown.values())
        
        return RewardResult(
            total=total,
            breakdown=breakdown,
            errors=errors,
            test_results=test_results
        )
    
    def _compute_file_match(
        self,
        actual_files: Dict[str, str],
        expected_files: Dict[str, str],
        target_files: List[str]
    ) -> float:
        """Compute file matching score.
        
        Args:
            actual_files: Actual file contents
            expected_files: Expected file contents
            target_files: Files that should be changed
            
        Returns:
            Match score (0.0 to 1.0)
        """
        if not target_files:
            return 0.0
        
        scores = []
        
        for filepath in target_files:
            actual = actual_files.get(filepath, "")
            expected = expected_files.get(filepath, "")
            
            if not expected:
                continue
            
            result = self.diff_scorer.score_content(actual, expected)
            scores.append(result.similarity)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _check_exact_match(
        self,
        actual_files: Dict[str, str],
        expected_files: Dict[str, str],
        target_files: List[str]
    ) -> bool:
        """Check if target files exactly match expected.
        
        Args:
            actual_files: Actual file contents
            expected_files: Expected file contents
            target_files: Files to check
            
        Returns:
            True if all target files match exactly
        """
        for filepath in target_files:
            actual = actual_files.get(filepath, "").strip()
            expected = expected_files.get(filepath, "").strip()
            
            if actual != expected:
                return False
        
        return True
    
    def compute_partial(
        self,
        actual_content: str,
        expected_content: str
    ) -> float:
        """Compute partial reward for a single file.
        
        Useful for intermediate feedback.
        
        Args:
            actual_content: Actual file content
            expected_content: Expected file content
            
        Returns:
            Partial reward (0.0 to 1.0)
        """
        # Check syntax
        syntax_result = self.syntax_checker.check(actual_content)
        if not syntax_result.valid:
            return 0.1  # Small reward for attempt
        
        # Compute similarity
        diff_result = self.diff_scorer.score_content(actual_content, expected_content)
        
        # Combine scores
        syntax_score = 0.2
        similarity_score = 0.8 * diff_result.similarity
        
        return syntax_score + similarity_score
