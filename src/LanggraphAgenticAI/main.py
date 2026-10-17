import os
import streamlit as st

from src.LanggraphAgenticAI.Graph.code_assistant_graph import CodeAssistantGraph
from src.LanggraphAgenticAI.LLMs.groqllm import GroqLLM
from src.LanggraphAgenticAI.tools.vector_memory import build_local_memory
from src.LanggraphAgenticAI.UI.streamlitui.loadui import LoadStreamlitUI


def load_langgraph_agenticai_app():
    st.set_page_config(page_title="Agentic Code Assistant", layout="wide")
    st.title("Agentic Code Assistant & Search Bot")
    st.caption("Routes requests to uploaded project memory, live web search, or both; validate and review each answer.")

    controls = LoadStreamlitUI().load_streamlit_ui()
    if controls.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = controls["TAVILY_API_KEY"]
    uploads = st.sidebar.file_uploader(
        "Add local project context",
        type=["py", "md", "txt", "json", "toml", "yaml", "yml", "js", "ts", "tsx", "jsx", "java", "go", "rs"],
        accept_multiple_files=True,
        help="Files are indexed in memory for this session. Retrieved excerpts are sent to Groq when used to answer.",
    )
    uploaded = [(file.name, file.getvalue().decode("utf-8", errors="replace")) for file in uploads or []]
    try:
        memory = build_local_memory(uploaded)
    except Exception as exc:
        st.sidebar.error(f"Could not index local files: {exc}")
        memory = None
    if memory:
        st.sidebar.success(f"Indexed {len(uploaded)} file(s) for this session.")

    if "pending_review" in st.session_state:
        pending = st.session_state.pending_review
        st.subheader("Review the generated answer")
        st.caption(f"Source route: {pending.get('route', 'unknown')}")
        st.markdown(pending.get("draft", ""))
        errors = pending.get("validation_errors", [])
        if errors:
            st.warning("Validation flagged: " + " ".join(errors))
        if pending.get("search_error"):
            st.warning("Live search was unavailable; the draft was generated without web results. Check your Tavily key or use local/general context.")
        col_approve, col_revise = st.columns([1, 3])
        with col_approve:
            if st.button("Approve answer", type="primary"):
                st.session_state.approved_answer = pending["draft"]
                del st.session_state.pending_review
                st.rerun()
        with col_revise:
            with st.form("review_feedback"):
                feedback = st.text_input("Request a revision", placeholder="e.g. show the edge cases and add a test")
                revise = st.form_submit_button("Revise answer")
            if revise and feedback.strip():
                try:
                    llm = GroqLLM(controls).get_llm_model()
                    graph = CodeAssistantGraph(llm, memory).compile()
                    revised = graph.invoke({"request": pending["request"], "feedback": feedback.strip()})
                    st.session_state.pending_review = revised | {"request": pending["request"]}
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not revise the answer: {exc}")

    if "approved_answer" in st.session_state:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.approved_answer)
        if st.button("Clear approved answer"):
            del st.session_state.approved_answer
            st.rerun()

    request = st.chat_input("Describe a coding problem or question")
    if request:
        try:
            if not controls.get("GROQ_API_KEY"):
                st.error("Enter a Groq API key in the sidebar to continue.")
                return
            llm = GroqLLM(controls).get_llm_model()
            graph = CodeAssistantGraph(llm, memory).compile()
            with st.spinner("Routing, retrieving evidence, and validating a draft…"):
                result = graph.invoke({"request": request, "validation_attempts": 0})
            st.session_state.pending_review = result | {"request": request}
            st.rerun()
        except Exception as exc:
            st.error(f"The workflow could not complete: {exc}")
