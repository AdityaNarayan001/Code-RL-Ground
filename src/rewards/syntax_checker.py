"""Syntax checking for Python code."""

import ast
import builtins
import sys
from typing import Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class SyntaxErrorInfo:
    """A syntax error in code."""
    line: int
    column: int
    message: str
    text: Optional[str] = None


@dataclass
class SyntaxResult:
    """Result of syntax check."""
    valid: bool
    errors: List[SyntaxErrorInfo]
    warnings: List[str] = field(default_factory=list)


class SyntaxChecker:
    """Check Python code for syntax errors."""

    # Standard library modules (subset for quick checking)
    STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
        'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'builtins',
        'calendar', 'cmath', 'collections', 'colorsys', 'concurrent', 'configparser',
        'contextlib', 'copy', 'csv', 'ctypes', 'dataclasses', 'datetime', 'decimal',
        'difflib', 'email', 'enum', 'errno', 'fnmatch', 'fractions', 'functools',
        'getpass', 'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
        'importlib', 'inspect', 'io', 'itertools', 'json', 'keyword', 'linecache',
        'locale', 'logging', 'math', 'mimetypes', 'multiprocessing', 'numbers',
        'operator', 'os', 'pathlib', 'pickle', 'platform', 'pprint', 'queue',
        'random', 're', 'shlex', 'shutil', 'signal', 'socket', 'sqlite3',
        'statistics', 'string', 'struct', 'subprocess', 'sys', 'tempfile',
        'textwrap', 'threading', 'time', 'timeit', 'traceback', 'types',
        'typing', 'unicodedata', 'unittest', 'urllib', 'uuid', 'warnings',
        'weakref', 'xml', 'zipfile', 'zlib',
    }

    def __init__(self):
        pass

    def check(self, code: str) -> SyntaxResult:
        """Check code for syntax errors.

        Also validates that the code compiles and checks for obvious
        import issues with standard library modules.

        Args:
            code: Python code to check

        Returns:
            SyntaxResult with validity, errors, and warnings
        """
        # Step 1: Parse with ast
        try:
            tree = ast.parse(code)
        except builtins.SyntaxError as e:
            error = SyntaxErrorInfo(
                line=e.lineno or 0,
                column=e.offset or 0,
                message=e.msg if hasattr(e, 'msg') else str(e),
                text=e.text
            )
            return SyntaxResult(valid=False, errors=[error])

        # Step 2: Compile to bytecode (catches additional issues beyond ast.parse)
        try:
            compile(code, '<string>', 'exec')
        except builtins.SyntaxError as e:
            error = SyntaxErrorInfo(
                line=e.lineno or 0,
                column=e.offset or 0,
                message=e.msg if hasattr(e, 'msg') else str(e),
                text=e.text
            )
            return SyntaxResult(valid=False, errors=[error])

        # Step 3: Check for missing standard library imports (warnings only)
        warnings = self._check_imports(tree)

        return SyntaxResult(valid=True, errors=[], warnings=warnings)

    def _check_imports(self, tree: ast.AST) -> List[str]:
        """Check for obvious import issues with standard library modules.

        Returns a list of warning strings for imports that reference
        known stdlib modules with potentially misspelled names.
        """
        warnings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split('.')[0]
                    # Warn if it looks like a misspelled stdlib module
                    if top_module not in self.STDLIB_MODULES and self._is_close_to_stdlib(top_module):
                        warnings.append(
                            f"Line {node.lineno}: import '{alias.name}' - "
                            f"'{top_module}' is not a known standard library module"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_module = node.module.split('.')[0]
                    if top_module not in self.STDLIB_MODULES and self._is_close_to_stdlib(top_module):
                        warnings.append(
                            f"Line {node.lineno}: from '{node.module}' import - "
                            f"'{top_module}' is not a known standard library module"
                        )

        return warnings

    def _is_close_to_stdlib(self, name: str) -> bool:
        """Check if a module name is close to a stdlib module (likely typo)."""
        import difflib
        close = difflib.get_close_matches(name, self.STDLIB_MODULES, n=1, cutoff=0.8)
        return len(close) > 0

    def check_file(self, filepath: str) -> SyntaxResult:
        """Check a file for syntax errors.

        Args:
            filepath: Path to Python file

        Returns:
            SyntaxResult
        """
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            return self.check(code)
        except Exception as e:
            return SyntaxResult(
                valid=False,
                errors=[SyntaxErrorInfo(line=0, column=0, message=str(e))]
            )

    def check_multiple(self, files: dict) -> dict:
        """Check multiple files for syntax errors.

        Args:
            files: Dictionary of {filepath: content}

        Returns:
            Dictionary of {filepath: SyntaxResult}
        """
        results = {}
        for filepath, content in files.items():
            if filepath.endswith('.py'):
                results[filepath] = self.check(content)
        return results

    def all_valid(self, files: dict) -> Tuple[bool, List[str]]:
        """Check if all files have valid syntax.

        Args:
            files: Dictionary of {filepath: content}

        Returns:
            Tuple of (all_valid, list of error messages)
        """
        results = self.check_multiple(files)
        errors = []

        for filepath, result in results.items():
            if not result.valid:
                for err in result.errors:
                    errors.append(f"{filepath}:{err.line}: {err.message}")

        return len(errors) == 0, errors
