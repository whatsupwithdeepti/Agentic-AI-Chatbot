from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input=user_controls_input

    def get_llm_model(self):
        try:
            groq_api_key=self.user_controls_input.get("GROQ_API_KEY")
            selected_groq_model=self.user_controls_input["selected_groq_model"]
            if not groq_api_key:
                raise ValueError("Please enter a Groq API key")
            
            llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)

        except Exception as e:
            raise ValueError(f"Could not initialize the Groq model: {e}") from e
        return llm
