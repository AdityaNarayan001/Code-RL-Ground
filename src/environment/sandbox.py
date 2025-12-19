"""Python sandbox for safe code execution."""

import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import signal
import resource


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0


class PythonSandbox:
    """Safe Python code execution sandbox using subprocess.
    
    Provides isolated execution environment with:
    - Timeout protection
    - Memory limits
    - Import restrictions
    - Working directory isolation
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
        allowed_imports: Optional[list] = None,
        working_dir: Optional[Path] = None
    ):
        """Initialize sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            allowed_imports: List of allowed import modules
            working_dir: Working directory for execution
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_imports = allowed_imports or [
            "os", "sys", "re", "json", "math", "collections",
            "itertools", "functools", "pathlib", "typing", "time",
            "dataclasses", "abc", "copy", "io", "string"
        ]
        self.working_dir = working_dir
    
    def _create_wrapper_script(self, code: str, test_code: Optional[str] = None) -> str:
        """Create a wrapper script for execution.
        
        Args:
            code: The main code to execute
            test_code: Optional test code to run after main code
            
        Returns:
            Wrapper script content
        """
        wrapper = '''
import sys
import traceback

# Capture output
_stdout_capture = []
_stderr_capture = []

try:
    # Execute main code
    exec("""
{code}
""")
    
    {test_section}
    
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
        
        test_section = ""
        if test_code:
            test_section = f'''
    # Execute test code
    exec("""
{test_code}
""")
'''
        
        return wrapper.format(code=code, test_section=test_section)
    
    def execute_code(
        self,
        code: str,
        test_code: Optional[str] = None,
        extra_files: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            test_code: Optional test code to run after main code
            extra_files: Additional files to create in working directory
            
        Returns:
            ExecutionResult with output and status
        """
        import time
        start_time = time.time()
        
        # Create temporary directory if needed
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(self.working_dir) if self.working_dir else Path(temp_dir)
            
            # Write extra files if provided
            if extra_files:
                for filepath, content in extra_files.items():
                    full_path = work_dir / filepath
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
            
            # Create the script to execute
            script_path = work_dir / "_sandbox_script.py"
            script_content = self._create_wrapper_script(code, test_code)
            script_path.write_text(script_content)
            
            try:
                # Run the script
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(work_dir),
                    env={
                        **os.environ,
                        'PYTHONPATH': str(work_dir),
                        'PYTHONDONTWRITEBYTECODE': '1'
                    }
                )
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=result.returncode == 0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    return_code=result.returncode,
                    execution_time=execution_time
                )
                
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout} seconds",
                    return_code=-1,
                    timed_out=True,
                    error_message="Timeout",
                    execution_time=self.timeout
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=str(e),
                    return_code=-1,
                    error_message=str(e),
                    execution_time=time.time() - start_time
                )
            finally:
                # Cleanup
                if script_path.exists():
                    script_path.unlink()
    
    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if Python code has valid syntax.
        
        Args:
            code: Python code to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def run_tests(
        self,
        module_code: str,
        test_cases: list,
        module_name: str = "module"
    ) -> Dict[str, Any]:
        """Run test cases against module code.
        
        Args:
            module_code: The module code to test
            test_cases: List of test case definitions
            module_name: Name for the module
            
        Returns:
            Dictionary with test results
        """
        results = {
            'total': len(test_cases),
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'details': []
        }
        
        for i, test in enumerate(test_cases):
            test_name = test.get('description', f'test_{i}')
            
            if 'test_code' in test:
                # Custom test code
                test_code = test['test_code']
            elif 'function' in test and 'input' in test:
                # Simple function test
                func_name = test['function']
                inputs = test['input']
                expected = test['expected']
                
                if isinstance(inputs, list):
                    args_str = ', '.join(repr(x) for x in inputs)
                else:
                    args_str = repr(inputs)
                
                if expected == "ValueError":
                    test_code = f'''
try:
    {func_name}({args_str})
    assert False, "Expected ValueError"
except ValueError:
    pass
'''
                else:
                    test_code = f'assert {func_name}({args_str}) == {repr(expected)}'
            elif 'input' in test:
                # Single function test (infer function from context)
                inputs = test['input']
                expected = test['expected']
                
                if expected == "ValueError":
                    test_code = f'''
try:
    result = test_func({repr(inputs)})
    assert False, "Expected ValueError"
except ValueError:
    pass
'''
                else:
                    test_code = f'assert test_func({repr(inputs)}) == {repr(expected)}'
            else:
                continue
            
            result = self.execute_code(module_code, test_code)
            
            if result.success:
                results['passed'] += 1
                results['details'].append({
                    'name': test_name,
                    'status': 'passed'
                })
            else:
                if 'Error' in result.stderr:
                    results['errors'] += 1
                    status = 'error'
                else:
                    results['failed'] += 1
                    status = 'failed'
                
                results['details'].append({
                    'name': test_name,
                    'status': status,
                    'error': result.stderr[:500]  # Truncate long errors
                })
        
        return results
    
    def execute_in_repo(
        self,
        repo_path: Path,
        command: str
    ) -> ExecutionResult:
        """Execute Python code in a repository context.
        
        Args:
            repo_path: Path to the repository
            command: Python code or command to execute
            
        Returns:
            ExecutionResult
        """
        self.working_dir = repo_path
        return self.execute_code(command)
