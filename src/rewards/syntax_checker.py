"""Syntax checking for Python code."""

import ast
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SyntaxError:
    """A syntax error in code."""
    line: int
    column: int
    message: str
    text: Optional[str] = None


@dataclass
class SyntaxResult:
    """Result of syntax check."""
    valid: bool
    errors: List[SyntaxError]


class SyntaxChecker:
    """Check Python code for syntax errors."""
    
    def __init__(self):
        pass
    
    def check(self, code: str) -> SyntaxResult:
        """Check code for syntax errors.
        
        Args:
            code: Python code to check
            
        Returns:
            SyntaxResult with validity and any errors
        """
        try:
            ast.parse(code)
            return SyntaxResult(valid=True, errors=[])
        except SyntaxError as e:
            error = SyntaxError(
                line=e.lineno or 0,
                column=e.offset or 0,
                message=e.msg or str(e),
                text=e.text
            )
            return SyntaxResult(valid=False, errors=[error])
    
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
                errors=[SyntaxError(line=0, column=0, message=str(e))]
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
