import streamlit as st

from src.LanggraphAgenticAI.UI.uiconfigfile import Config


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()

    def load_streamlit_ui(self):
        controls = {}
        with st.sidebar:
            st.subheader("Model settings")
            models = self.config.groq_model_options()
            controls["selected_groq_model"] = st.selectbox("Groq model", models)
            controls["GROQ_API_KEY"] = st.text_input(
                "Groq API key", value=st.session_state.get("GROQ_API_KEY", ""), type="password"
            )
            st.session_state["GROQ_API_KEY"] = controls["GROQ_API_KEY"]
            if not controls["GROQ_API_KEY"]:
                st.caption("A Groq API key is required to generate answers.")
            controls["TAVILY_API_KEY"] = st.text_input(
                "Tavily API key (for web search)", value=st.session_state.get("TAVILY_API_KEY", ""), type="password"
            )
            st.session_state["TAVILY_API_KEY"] = controls["TAVILY_API_KEY"]
        return controls
