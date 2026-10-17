"""LangGraph workflow for routed, grounded code assistance."""
import json
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults

from src.LanggraphAgenticAI.state.state import AgentState
from src.LanggraphAgenticAI.tools.vector_memory import retrieve_local_context


def _content(response: Any) -> str:
    value = getattr(response, "content", response)
    if isinstance(value, list):
        return "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in value)
    return str(value)


class CodeAssistantGraph:
    def __init__(self, model, memory=None):
        self.model = model
        self.memory = memory

    def _route(self, state: AgentState) -> dict:
        has_memory = self.memory is not None
        prompt = (
            "Choose information sources for this software engineering request. "
            "Return only JSON {\"route\":\"local|web|both|none\"}. "
            "local means answer from uploaded project files; web means current or external facts; "
            "both means the user needs project context plus current external facts; none means general knowledge. "
            f"Uploaded project files available: {has_memory}.\nRequest: {state['request']}"
        )
        try:
            raw = _content(self.model.invoke(prompt))
            match = re.search(r"\{.*?\}", raw, re.S)
            route = json.loads(match.group(0)).get("route") if match else "none"
            if route not in {"local", "web", "both", "none"}:
                route = "none"
        except Exception:
            route = "local" if has_memory else "none"
            if re.search(r"\b(latest|today|current|recent|version|release|pricing|news)\b", state["request"], re.I):
                route = "both" if has_memory else "web"
        if route == "local" and not has_memory:
            route = "none"
        if route == "both" and not has_memory:
            route = "web"
        return {"route": route}

    def _retrieve(self, state: AgentState) -> dict:
        return {"local_context": retrieve_local_context(self.memory, state["request"])}

    def _search(self, state: AgentState) -> dict:
        try:
            results = TavilySearchResults(max_results=5).invoke(state["request"])
            if isinstance(results, dict):
                results = results.get("results", [results])
            return {"search_results": results or [], "search_error": ""}
        except Exception as exc:
            return {"search_results": [], "search_error": str(exc)}

    def _generate(self, state: AgentState) -> dict:
        sources = []
        if state.get("local_context"):
            sources.append("UPLOADED PROJECT CONTEXT:\n" + state["local_context"])
        if state.get("search_results"):
            web = "\n".join(
                f"- {item.get('title', 'Web result')}: {item.get('content', '')}\n  URL: {item.get('url', '')}"
                for item in state["search_results"] if isinstance(item, dict)
            )
            sources.append("LIVE WEB SEARCH RESULTS:\n" + web)
        feedback = state.get("feedback", "")
        errors = state.get("validation_errors", [])
        prompt = (
            "You are a careful software engineering assistant. Analyze the user's request, explain assumptions, "
            "and provide actionable code guidance. Ground claims in supplied context when present; distinguish "
            "inference from evidence and cite web URLs inline. Never claim a command or test was run unless it was. "
            "If context is insufficient, say so. Return concise Markdown.\n\n"
            f"REQUEST:\n{state['request']}\n\n"
            f"EVIDENCE:\n{chr(10).join(sources) if sources else 'No retrieved evidence.'}\n\n"
            f"SEARCH STATUS:\n{state.get('search_error', '') or 'Search not requested or completed.'}\n\n"
            f"REVIEW FEEDBACK TO ADDRESS:\n{feedback}\n"
            f"VALIDATION ISSUES TO FIX:\n{'; '.join(errors)}"
        )
        return {"draft": _content(self.model.invoke(prompt)), "validation_attempts": state.get("validation_attempts", 0) + 1}

    def _validate(self, state: AgentState) -> dict:
        draft = state.get("draft", "").strip()
        errors = []
        if len(draft) < 30:
            errors.append("Response is empty or too short to be useful.")
        if state.get("search_results") and not re.search(r"https?://", draft):
            errors.append("Web-grounded answer must cite at least one source URL.")
        if re.search(r"\b(write|implement|create|fix|debug)\b.*\b(code|function|class|script|program|bug)\b", state["request"], re.I) and "```" not in draft:
            errors.append("Code request should include a fenced code example or explain why one is not appropriate.")
        return {"validation_errors": errors}

    @staticmethod
    def _after_route(state: AgentState) -> str:
        return state.get("route", "none")

    @staticmethod
    def _after_validate(state: AgentState) -> str:
        if state.get("validation_errors") and state.get("validation_attempts", 0) < 2:
            return "repair"
        return "done"

    def compile(self):
        builder = StateGraph(AgentState)
        builder.add_node("route", self._route)
        builder.add_node("retrieve_local", self._retrieve)
        builder.add_node("search_web", self._search)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", self._after_route, {"local": "retrieve_local", "web": "search_web", "both": "retrieve_local", "none": "generate"})
        builder.add_conditional_edges("retrieve_local", lambda s: "search" if s.get("route") == "both" else "generate", {"search": "search_web", "generate": "generate"})
        builder.add_edge("search_web", "generate")
        builder.add_edge("generate", "validate")
        builder.add_conditional_edges("validate", self._after_validate, {"repair": "generate", "done": END})
        return builder.compile()
