"""Tool definitions for the code agent."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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

    def __init__(self, working_dir: Optional[Path] = None, available_tools: Optional[List[str]] = None):
        """Initialize tool registry.

        Args:
            working_dir: Working directory for file operations
            available_tools: Whitelist of tool names to register.  If ``None``
                all default tools are registered.  Comes from
                ``config.environment.tools.available``.
        """
        self.working_dir = working_dir
        self.tools: Dict[str, Tool] = {}
        self._file_cache: Dict[str, str] = {}
        self._modifications: Dict[str, str] = {}
        self._available_tools: Optional[List[str]] = available_tools

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
        """Register a tool.

        If an ``available_tools`` whitelist was provided at init time, tools
        whose name is not in the list are silently skipped.
        """
        if self._available_tools is not None and tool.name not in self._available_tools:
            logger.debug("Skipping tool '%s' (not in available_tools whitelist)", tool.name)
            return
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
    
    def _validate_path(self, path: str) -> Optional[ToolResult]:
        """Validate that a path does not escape the working directory.

        Returns a ToolResult with an error if the path is invalid, or None if valid.
        """
        if not self.working_dir:
            return None
        full_path = (self.working_dir / path).resolve()
        if not str(full_path).startswith(str(self.working_dir.resolve())):
            return ToolResult(
                success=False,
                output="",
                error=f"Path traversal detected: {path} escapes the working directory"
            )
        return None

    def _load_file_cache(self):
        """Load all files into cache."""
        if self.working_dir and self.working_dir.exists():
            for file_path in self.working_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        rel_path = str(file_path.relative_to(self.working_dir))
                        self._file_cache[rel_path] = file_path.read_text()
                    except (UnicodeDecodeError, PermissionError):
                        pass
                    except Exception as e:
                        logger.warning(f"Unexpected error loading file {file_path}: {e}")
    
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
        # Validate path
        path_error = self._validate_path(path)
        if path_error:
            return path_error

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
        # Validate path
        path_error = self._validate_path(path)
        if path_error:
            return path_error

        self._modifications[path] = content
        self._file_cache[path] = content
        
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
        # Validate path
        path_error = self._validate_path(path)
        if path_error:
            return path_error

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
        self._file_cache[path] = new_file_content

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
        # Validate path
        path_error = self._validate_path(path)
        if path_error:
            return path_error

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
        # Match <tool>...</tool> — grab everything inside, then parse tool name + args
        outer_match = re.search(r'<tool>(.*?)</tool>', text, re.DOTALL)

        if not outer_match:
            # Try alternate patterns
            outer_match = re.search(r'```tool\n(.*?)\n```', text, re.DOTALL)

        if not outer_match:
            return None

        inner = outer_match.group(1).strip()

        # Extract tool name: first word followed by (
        name_match = re.match(r'(\w+)\s*\(', inner)
        if not name_match:
            return None

        tool_name = name_match.group(1)
        # Args is everything after "name(" up to the last ")"
        rest = inner[name_match.end():]
        # Strip trailing ) if present (the outermost closing paren)
        if rest.rstrip().endswith(')'):
            rest = rest.rstrip()[:-1]
        args_str = rest.strip()

        # Parse arguments
        args = {}
        if args_str:
            try:
                # For write_file/edit_file: content arg contains code with quotes/newlines
                # Use a specialized extraction for 'content' argument
                if tool_name in ('write_file', 'edit_file'):
                    args = self._parse_content_args(args_str, tool_name)
                else:
                    # Standard key=value parsing for simple tools
                    arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|([^,\s\)]+))'
                    for m in re.finditer(arg_pattern, args_str):
                        key = m.group(1)
                        value = m.group(2) or m.group(3) or m.group(4)
                        if m.group(4) is not None:
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                pass
                        args[key] = value
            except (re.error, AttributeError) as e:
                logger.warning(f"Failed to parse tool arguments '{args_str}': {e}")

        return {"tool": tool_name, "args": args}

    @staticmethod
    def _parse_content_args(args_str: str, tool_name: str) -> Dict[str, str]:
        """Parse arguments for write_file/edit_file where content has quotes/newlines.

        Strategy: extract path first (simple), then treat everything after
        content= as the content value (greedy, handles embedded quotes).
        """
        args = {}

        # Extract path="..." (simple, no embedded quotes)
        path_match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', args_str)
        if path_match:
            args['path'] = path_match.group(1)

        if tool_name == 'write_file':
            # For write_file: content is everything after content=" until the end
            content_match = re.search(r'content\s*=\s*["\']', args_str)
            if content_match:
                # Start position after the opening quote
                start = content_match.end()
                # Content is everything from here to the end of args_str
                # (the outer regex already stripped the closing )</tool>)
                content = args_str[start:]
                # Strip trailing quote if present
                if content.endswith('"') or content.endswith("'"):
                    content = content[:-1]
                # Unescape common sequences
                content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
                args['content'] = content

        elif tool_name == 'edit_file':
            # For edit_file: old_content and new_content
            # Try to find old_content="..." and new_content="..."
            # These are harder — use the simple regex for now and fall back
            for key in ('old_content', 'new_content', 'content'):
                pattern = key + r'\s*=\s*["\']'
                km = re.search(pattern, args_str)
                if km:
                    start = km.end()
                    # Find the next key= or end of string
                    next_key = re.search(r',\s*(?:old_content|new_content|path)\s*=', args_str[start:])
                    if next_key:
                        val = args_str[start:start + next_key.start()]
                    else:
                        val = args_str[start:]
                    if val.endswith('"') or val.endswith("'"):
                        val = val[:-1]
                    val = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                    args[key] = val

        return args
