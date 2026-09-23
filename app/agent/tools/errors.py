"""Tool-related exceptions."""


class ToolError(Exception):
    """Base class for all tool errors."""


class ToolNotFoundError(ToolError):
    """Raised when a tool name is not registered."""


class ToolValidationError(ToolError):
    """Raised when tool arguments don't match the schema."""


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""