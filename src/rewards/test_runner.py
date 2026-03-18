"""Test runner for evaluating generated code."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import tempfile
import os


@dataclass
class TestResult:
    """Result of running tests."""
    passed: int
    failed: int
    errors: int
    total: int
    details: List[Dict[str, Any]]
    all_passed: bool
    output: str


class TestRunner:
    """Run tests to evaluate generated code."""

    def __init__(self, timeout: int = 30):
        """Initialize test runner.

        Args:
            timeout: Maximum time for test execution in seconds
        """
        self.timeout = timeout

    def run_test_cases(
        self,
        code_files: Dict[str, str],
        test_cases: List[Dict[str, Any]],
        working_dir: Optional[Path] = None
    ) -> TestResult:
        """Run test cases against code files.

        Args:
            code_files: Dictionary of {filepath: content}
            test_cases: List of test case definitions
            working_dir: Working directory for execution

        Returns:
            TestResult with pass/fail information
        """
        results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'total': len(test_cases),
            'details': [],
            'output': ''
        }

        # Create temporary directory if needed
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(working_dir) if working_dir else Path(temp_dir)

            # Write code files
            for filepath, content in code_files.items():
                file_path = work_dir / filepath
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

            # Run each test
            for i, test in enumerate(test_cases):
                test_name = test.get('description', f'test_{i}')
                test_result = self._run_single_test(test, work_dir, code_files)

                if test_result['status'] == 'passed':
                    results['passed'] += 1
                elif test_result['status'] == 'timeout':
                    results['errors'] += 1
                elif test_result['status'] == 'failed':
                    results['failed'] += 1
                else:
                    results['errors'] += 1

                results['details'].append({
                    'name': test_name,
                    **test_result
                })

        results['all_passed'] = results['passed'] == results['total']

        return TestResult(**results)

    def _run_single_test(
        self,
        test: Dict[str, Any],
        work_dir: Path,
        code_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Run a single test case.

        Args:
            test: Test case definition
            work_dir: Working directory
            code_files: Code files for context

        Returns:
            Dictionary with test result
        """
        # Build test code
        if 'test_code' in test:
            test_code = test['test_code']
        elif 'function' in test and 'input' in test:
            test_code = self._build_simple_test(test)
        elif 'input' in test and 'expected' in test:
            # Infer function name from code files
            expected_fn = test.get('expected_function', None)
            func_name = self._infer_function_name(code_files, expected_function=expected_fn)
            if func_name:
                # Promote to simple test with inferred function name
                enriched_test = {**test, 'function': func_name}
                test_code = self._build_simple_test(enriched_test)
            else:
                test_code = self._build_inline_test(test, code_files)
        else:
            return {'status': 'error', 'error': 'Invalid test definition'}

        # Determine which module to import
        imports = self._get_imports(code_files)

        # Build full test script
        script = self._build_test_script(imports, test_code)

        # Write and execute
        script_path = work_dir / "_test_runner.py"
        script_path.write_text(script)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(work_dir),
                env={**os.environ, 'PYTHONPATH': str(work_dir)}
            )

            if result.returncode == 0:
                return {'status': 'passed', 'output': result.stdout}
            else:
                return {
                    'status': 'failed',
                    'error': result.stderr or result.stdout,
                    'output': result.stdout
                }

        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Test timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
        finally:
            if script_path.exists():
                script_path.unlink()

    def _build_simple_test(self, test: Dict[str, Any]) -> str:
        """Build a simple function test."""
        func_name = test['function']
        inputs = test['input']
        expected = test['expected']

        if isinstance(inputs, list):
            args = ', '.join(repr(x) for x in inputs)
        else:
            args = repr(inputs)

        if expected == "ValueError":
            return f'''
try:
    {func_name}({args})
    raise AssertionError("Expected ValueError but none raised")
except ValueError:
    pass  # Expected
'''
        else:
            return f'''
result = {func_name}({args})
expected = {repr(expected)}
assert result == expected, f"Expected {{expected}}, got {{result}}"
'''

    def _build_inline_test(self, test: Dict[str, Any], code_files: Dict[str, str] = None) -> str:
        """Build an inline test without function name.

        Tests ALL candidate functions and reports which ones pass/fail,
        rather than exiting early on the first passing function.
        """
        inputs = test['input']
        expected = test['expected']

        # Try to find all public functions from code files
        func_names = self._extract_all_function_names(code_files or {})

        if func_names:
            # Test all functions and collect results
            func_checks = []
            func_checks.append("_any_passed = False")
            func_checks.append("_results = {}")

            for fn in func_names:
                if expected == "ValueError":
                    func_checks.append(f'''
try:
    {fn}({repr(inputs)})
    _results["{fn}"] = "no_error"
except ValueError:
    _results["{fn}"] = "passed"
    _any_passed = True
except Exception as _e:
    _results["{fn}"] = f"error: {{_e}}"
''')
                else:
                    func_checks.append(f'''
try:
    _result = {fn}({repr(inputs)})
    if _result == {repr(expected)}:
        _results["{fn}"] = "passed"
        _any_passed = True
    else:
        _results["{fn}"] = f"returned {{_result!r}}"
except Exception as _e:
    _results["{fn}"] = f"error: {{_e}}"
''')

            joined = "\n".join(func_checks)
            if expected == "ValueError":
                return f'''{joined}
if not _any_passed:
    raise AssertionError(f"No function raised ValueError for input {repr(inputs)}. Results: {{_results}}")
'''
            else:
                return f'''{joined}
if not _any_passed:
    raise AssertionError(f"No function returned {repr(expected)} for input {repr(inputs)}. Results: {{_results}}")
'''

        # Absolute fallback: evaluate as expression
        if expected == "ValueError":
            return f'''
try:
    eval({repr(inputs)})
    raise AssertionError("Expected ValueError")
except ValueError:
    pass
'''
        else:
            return f'''
result = eval({repr(inputs)})
assert result == {repr(expected)}, f"Expected {repr(expected)}, got {{result}}"
'''

    def _infer_function_name(
        self,
        code_files: Dict[str, str],
        expected_function: Optional[str] = None,
        base_files: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Infer the most likely function name from code files.

        Picks the function that was ADDED (not present in base_files).
        If base_files is not available, falls back to the expected_function
        parameter or the last defined function.

        Args:
            code_files: Current code files
            expected_function: Optional hint for expected function name
            base_files: Optional base repo files to diff against

        Returns:
            Function name or None
        """
        import re

        if expected_function:
            return expected_function

        # Collect all functions from current code
        current_funcs = []
        for filepath, content in code_files.items():
            if not filepath.endswith('.py'):
                continue
            for match in re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE):
                func_name = match.group(1)
                if not func_name.startswith('_'):
                    current_funcs.append((filepath, func_name))

        if not current_funcs:
            return None

        # If base_files provided, find functions that were added
        if base_files:
            base_funcs = set()
            for filepath, content in base_files.items():
                if not filepath.endswith('.py'):
                    continue
                for match in re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE):
                    base_funcs.add(match.group(1))

            added = [(fp, fn) for fp, fn in current_funcs if fn not in base_funcs]
            if added:
                return added[-1][1]

        # Fallback: return last candidate
        return current_funcs[-1][1]

    def _extract_all_function_names(self, code_files: Dict[str, str]) -> List[str]:
        """Extract all public function names from code files."""
        import re

        func_names = []
        for filepath, content in code_files.items():
            if not filepath.endswith('.py'):
                continue
            for match in re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE):
                func_name = match.group(1)
                if not func_name.startswith('_'):
                    func_names.append(func_name)

        return func_names

    def _get_imports(self, code_files: Dict[str, str]) -> List[str]:
        """Get import statements for code files."""
        imports = []

        for filepath in code_files:
            if filepath.endswith('.py') and not filepath.startswith('_'):
                # Convert path to module name
                module = filepath.replace('/', '.').replace('\\', '.')[:-3]
                if module.startswith('.'):
                    module = module[1:]

                # Handle __init__.py
                if module.endswith('.__init__'):
                    module = module[:-9]

                if module:
                    imports.append(f"from {module} import *")

        return imports

    def _build_test_script(self, imports: List[str], test_code: str) -> str:
        """Build complete test script."""
        import_block = '\n'.join(imports)

        return f'''
import sys
sys.path.insert(0, '.')

{import_block}

# Test execution
{test_code}

print("PASS")
'''

    def run_pytest(
        self,
        test_dir: Path,
        working_dir: Path
    ) -> TestResult:
        """Run pytest on a test directory.

        Args:
            test_dir: Directory containing test files
            working_dir: Working directory

        Returns:
            TestResult
        """
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_dir), '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=self.timeout * 2,
                cwd=str(working_dir),
                env={**os.environ, 'PYTHONPATH': str(working_dir)}
            )

            # Parse pytest output
            passed = result.stdout.count(' PASSED')
            failed = result.stdout.count(' FAILED')
            errors = result.stdout.count(' ERROR')

            return TestResult(
                passed=passed,
                failed=failed,
                errors=errors,
                total=passed + failed + errors,
                details=[],
                all_passed=(failed == 0 and errors == 0),
                output=result.stdout + result.stderr
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                passed=0, failed=0, errors=1, total=1,
                details=[{'status': 'timeout', 'error': 'Timeout'}],
                all_passed=False,
                output='Test execution timed out'
            )
        except Exception as e:
            return TestResult(
                passed=0, failed=0, errors=1, total=1,
                details=[{'status': 'error', 'error': str(e)}],
                all_passed=False,
                output=str(e)
            )
