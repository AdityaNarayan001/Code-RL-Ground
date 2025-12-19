"""Tool definitions for the code agent."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    """Definition of a tool available to the agent."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    handler: Optional[Callable] = None
    
    def to_prompt_format(self) -> str:
        """Convert tool to prompt-friendly format."""
        params_str = ", ".join(
            f"{name}: {info.get('type', 'any')}"
            for name, info in self.parameters.items()
        )
        return f"{self.name}({params_str}) - {self.description}"
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params
                }
            }
        }


class ToolRegistry:
    """Registry of available tools for the agent."""
    
    def __init__(self, working_dir: Optional[Path] = None):
        """Initialize tool registry.
        
        Args:
            working_dir: Working directory for file operations
        """
        self.working_dir = working_dir
        self.tools: Dict[str, Tool] = {}
        self._file_cache: Dict[str, str] = {}
        self._modifications: Dict[str, str] = {}
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of tools."""
        
        # read_file
        self.register(Tool(
            name="read_file",
            description="Read the contents of a file from the repository",
            parameters={
                "path": {"type": "string", "description": "Relative path to the file"}
            },
            required_params=["path"],
            handler=self._handle_read_file
        ))
        
        # write_file
        self.register(Tool(
            name="write_file",
            description="Create a new file with the given content",
            parameters={
                "path": {"type": "string", "description": "Relative path for the new file"},
                "content": {"type": "string", "description": "Content to write to the file"}
            },
            required_params=["path", "content"],
            handler=self._handle_write_file
        ))
        
        # edit_file
        self.register(Tool(
            name="edit_file",
            description="Edit an existing file by replacing old content with new content",
            parameters={
                "path": {"type": "string", "description": "Relative path to the file"},
                "old_content": {"type": "string", "description": "The exact content to find and replace"},
                "new_content": {"type": "string", "description": "The new content to replace with"}
            },
            required_params=["path", "old_content", "new_content"],
            handler=self._handle_edit_file
        ))
        
        # list_directory
        self.register(Tool(
            name="list_directory",
            description="List contents of a directory",
            parameters={
                "path": {"type": "string", "description": "Relative path to the directory (use '.' for root)"}
            },
            required_params=["path"],
            handler=self._handle_list_directory
        ))
        
        # run_python
        self.register(Tool(
            name="run_python",
            description="Execute a Python code snippet and return the output",
            parameters={
                "code": {"type": "string", "description": "Python code to execute"}
            },
            required_params=["code"],
            handler=self._handle_run_python
        ))
        
        # search_code
        self.register(Tool(
            name="search_code",
            description="Search for a pattern in the codebase",
            parameters={
                "pattern": {"type": "string", "description": "Text pattern to search for"},
                "file_pattern": {"type": "string", "description": "Optional glob pattern to filter files (e.g., '*.py')"}
            },
            required_params=["pattern"],
            handler=self._handle_search_code
        ))
        
        # submit
        self.register(Tool(
            name="submit",
            description="Submit the current changes as the solution. Call this when you're done.",
            parameters={},
            required_params=[],
            handler=self._handle_submit
        ))
    
    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def set_working_dir(self, path: Path):
        """Set the working directory for file operations."""
        self.working_dir = path
        self._file_cache.clear()
        self._modifications.clear()
        self._load_file_cache()
    
    def _load_file_cache(self):
        """Load all files into cache."""
        if self.working_dir and self.working_dir.exists():
            for file_path in self.working_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        rel_path = str(file_path.relative_to(self.working_dir))
                        self._file_cache[rel_path] = file_path.read_text()
                    except:
                        pass
    
    def get_modifications(self) -> Dict[str, str]:
        """Get all file modifications made during the episode."""
        return self._modifications.copy()
    
    def reset(self):
        """Reset modifications for a new episode."""
        self._modifications.clear()
        self._load_file_cache()
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with output or error
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}"
            )
        
        # Check required parameters
        for param in tool.required_params:
            if param not in kwargs:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Missing required parameter: {param}"
                )
        
        # Execute handler
        if tool.handler:
            try:
                return tool.handler(**kwargs)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool execution error: {str(e)}"
                )
        
        return ToolResult(
            success=False,
            output="",
            error="No handler registered for tool"
        )
    
    def _handle_read_file(self, path: str) -> ToolResult:
        """Handle read_file tool."""
        # Check modifications first
        if path in self._modifications:
            return ToolResult(success=True, output=self._modifications[path])
        
        # Check cache
        if path in self._file_cache:
            return ToolResult(success=True, output=self._file_cache[path])
        
        # Try to read from disk
        if self.working_dir:
            full_path = self.working_dir / path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    self._file_cache[path] = content
                    return ToolResult(success=True, output=content)
                except Exception as e:
                    return ToolResult(success=False, output="", error=str(e))
        
        return ToolResult(
            success=False,
            output="",
            error=f"File not found: {path}"
        )
    
    def _handle_write_file(self, path: str, content: str) -> ToolResult:
        """Handle write_file tool."""
        self._modifications[path] = content
        
        # Also write to disk if working_dir is set
        if self.working_dir:
            full_path = self.working_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        return ToolResult(
            success=True,
            output=f"Successfully created file: {path}"
        )
    
    def _handle_edit_file(self, path: str, old_content: str, new_content: str) -> ToolResult:
        """Handle edit_file tool."""
        # Get current content
        current = self._modifications.get(path) or self._file_cache.get(path)
        
        if current is None:
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}"
            )
        
        if old_content not in current:
            return ToolResult(
                success=False,
                output="",
                error=f"Could not find the specified content to replace in {path}"
            )
        
        # Perform replacement
        new_file_content = current.replace(old_content, new_content, 1)
        self._modifications[path] = new_file_content
        
        # Write to disk
        if self.working_dir:
            full_path = self.working_dir / path
            full_path.write_text(new_file_content)
        
        return ToolResult(
            success=True,
            output=f"Successfully edited file: {path}"
        )
    
    def _handle_list_directory(self, path: str) -> ToolResult:
        """Handle list_directory tool."""
        entries = []
        
        if path == "." or path == "":
            # List from all known files
            dirs = set()
            files = set()
            for file_path in list(self._file_cache.keys()) + list(self._modifications.keys()):
                parts = file_path.split("/")
                if len(parts) > 1:
                    dirs.add(parts[0] + "/")
                else:
                    files.add(parts[0])
            entries = sorted(dirs) + sorted(files)
        else:
            # List specific directory
            prefix = path.rstrip("/") + "/"
            items = set()
            for file_path in list(self._file_cache.keys()) + list(self._modifications.keys()):
                if file_path.startswith(prefix):
                    remaining = file_path[len(prefix):]
                    parts = remaining.split("/")
                    if len(parts) > 1:
                        items.add(parts[0] + "/")
                    else:
                        items.add(parts[0])
            entries = sorted(items)
        
        if not entries:
            return ToolResult(
                success=False,
                output="",
                error=f"Directory not found or empty: {path}"
            )
        
        return ToolResult(
            success=True,
            output="\n".join(entries)
        )
    
    def _handle_run_python(self, code: str) -> ToolResult:
        """Handle run_python tool."""
        from .sandbox import PythonSandbox
        
        sandbox = PythonSandbox(
            timeout=10,
            working_dir=self.working_dir
        )
        
        result = sandbox.execute_code(code)
        
        if result.success:
            return ToolResult(
                success=True,
                output=result.stdout or "(no output)"
            )
        else:
            return ToolResult(
                success=False,
                output=result.stdout,
                error=result.stderr
            )
    
    def _handle_search_code(self, pattern: str, file_pattern: str = "*.py") -> ToolResult:
        """Handle search_code tool."""
        import fnmatch
        
        matches = []
        all_files = {**self._file_cache, **self._modifications}
        
        for file_path, content in all_files.items():
            if not fnmatch.fnmatch(file_path, file_pattern) and not file_path.endswith('.py'):
                continue
            
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path}:{i}: {line.strip()}")
        
        if not matches:
            return ToolResult(
                success=True,
                output=f"No matches found for: {pattern}"
            )
        
        return ToolResult(
            success=True,
            output="\n".join(matches[:20])  # Limit results
        )
    
    def _handle_submit(self) -> ToolResult:
        """Handle submit tool."""
        return ToolResult(
            success=True,
            output="SUBMIT",
            data={"action": "submit", "modifications": self._modifications}
        )
    
    def get_tools_prompt(self) -> str:
        """Get a formatted prompt describing available tools."""
        lines = ["Available tools:"]
        for tool in self.tools.values():
            lines.append(f"  - {tool.to_prompt_format()}")
        lines.append("")
        lines.append("Use tools by outputting: <tool>tool_name(param1=\"value1\", param2=\"value2\")</tool>")
        return "\n".join(lines)
    
    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a tool call from model output.
        
        Args:
            text: Model output text
            
        Returns:
            Dictionary with tool name and arguments, or None
        """
        # Match <tool>name(args)</tool> pattern
        pattern = r'<tool>(\w+)\((.*?)\)</tool>'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            # Try alternate patterns
            pattern2 = r'```tool\n(\w+)\((.*?)\)\n```'
            match = re.search(pattern2, text, re.DOTALL)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Parse arguments
        args = {}
        if args_str:
            # Handle key=value pairs
            # This is a simple parser - could be made more robust
            try:
                # Try to parse as Python dict-like syntax
                # Handle both key="value" and key=value
                arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|([^,\s\)]+))'
                for m in re.finditer(arg_pattern, args_str):
                    key = m.group(1)
                    value = m.group(2) or m.group(3) or m.group(4)
                    args[key] = value
            except:
                pass
        
        return {"tool": tool_name, "args": args}
