from typing import Any
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Conversation state retained for the original chatbot graph modules."""
    messages: Annotated[list, add_messages]


class AgentState(TypedDict, total=False):
    request: str
    route: str
    local_context: str
    search_results: list[dict[str, Any]]
    search_error: str
    draft: str
    feedback: str
    validation_errors: list[str]
    validation_attempts: int
    approved: bool

