"""Composite reward function for code generation."""

import importlib
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from .syntax_checker import SyntaxChecker
from .diff_scorer import DiffScorer
from .test_runner import TestRunner
from ..utils.config import Config, RewardsConfig


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
    - Syntax validity (0.1)
    - Test passing (0.5)
    - File matching (0.1)
    - Exact match bonus (0.2)
    - Import check (0.1)
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
        actual_state,
        expected_state,
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

        # 2. Import check — try importing each Python file and catch ImportError
        import_ok = self._check_imports(actual_files)
        breakdown['import_check'] = self.weights.import_check if import_ok else 0.0

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
                breakdown['tests_pass'] = 0.0

            if not tr.all_passed:
                for detail in tr.details:
                    if detail.get('status') != 'passed':
                        errors.append(f"Test {detail.get('name', '?')}: {detail.get('error', 'failed')[:100]}")
        else:
            # No tests defined — no free reward
            breakdown['tests_pass'] = 0.0

        # 4. File matching
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

        # 5. Exact match bonus (uses same normalization as diff_scorer)
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

    def _check_imports(self, files: Dict[str, str]) -> bool:
        """Try importing each Python file to catch ImportErrors.

        Args:
            files: Dictionary of {filepath: content}

        Returns:
            True if all files import without ImportError
        """
        for filepath, content in files.items():
            if not filepath.endswith('.py'):
                continue
            # Skip files that aren't pure source modules
            basename = Path(filepath).name
            if basename in ('setup.py', '__main__.py', '__init__.py'):
                continue
            # Skip test files (they import the package which isn't available in exec context)
            if basename.startswith('test_') or '/tests/' in filepath or filepath.startswith('tests/'):
                continue
            try:
                code = compile(content, filepath, 'exec')
                # Use a clean module name (no slashes)
                mod_name = f'_import_check_{basename}'
                module = types.ModuleType(mod_name)
                module.__file__ = filepath
                exec(code, module.__dict__)
            except ImportError:
                return False
            except (SystemExit, KeyboardInterrupt):
                pass
            except Exception:
                pass
        return True

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

        Uses the same normalization as DiffScorer: rstrip each line,
        strip leading/trailing blank lines.

        Args:
            actual_files: Actual file contents
            expected_files: Expected file contents
            target_files: Files to check

        Returns:
            True if all target files match exactly
        """
        for filepath in target_files:
            actual = actual_files.get(filepath, "")
            expected = expected_files.get(filepath, "")

            if self._normalize(actual) != self._normalize(expected):
                return False

        return True

    @staticmethod
    def _normalize(content: str) -> str:
        """Normalize content the same way as DiffScorer._normalize.

        Rstrip each line, strip leading/trailing blank lines.
        """
        lines = [line.rstrip() for line in content.split('\n')]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)

    def compute_partial(
        self,
        actual_content: str,
        expected_content: str
    ) -> float:
        """Compute partial reward for a single file.

        Uses the same weighted formula as compute():
        - syntax_valid weight for syntax
        - import_check weight for import validation
        - files_match weight scaled by similarity
        - exact_match weight if content matches exactly
        - tests_pass is 0.0 (no tests in partial mode)

        If ``partial_credit.enabled`` is False, returns 0.0 immediately.

        Args:
            actual_content: Actual file content
            expected_content: Expected file content

        Returns:
            Partial reward (0.0 to 1.0)
        """
        if not self.reward_config.partial_credit.enabled:
            return 0.0

        score = 0.0

        # Syntax check
        syntax_result = self.syntax_checker.check(actual_content)
        if syntax_result.valid:
            score += self.weights.syntax_valid
        else:
            # Small reward for attempt even if syntax is invalid
            return 0.1 * self.weights.syntax_valid

        # Import check
        if self._check_imports({'partial.py': actual_content}):
            score += self.weights.import_check

        # File similarity (uses function_match_weight from config)
        diff_result = self.diff_scorer.score_content(actual_content, expected_content)
        pc = self.reward_config.partial_credit
        # Blend line similarity and function matching using config weights
        blended_similarity = (
            pc.line_match_weight * diff_result.similarity
            + pc.function_match_weight * diff_result.additions_matched
        )
        score += self.weights.files_match * blended_similarity

        # Exact match
        if self._normalize(actual_content) == self._normalize(expected_content):
            score += self.weights.exact_match

        # tests_pass = 0.0 in partial mode (no tests)

        return max(0.0, min(1.0, score))
