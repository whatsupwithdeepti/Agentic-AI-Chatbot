# Agentic Code Assistant & Search Bot

A Streamlit code assistant built with LangGraph. It routes software questions to uploaded project context, live Tavily search, both sources, or neither; drafts an answer with Groq; checks basic output requirements; retries once when validation fails; and holds the result for human approval or revision.

## Workflow

1. **Route**: the model chooses `local`, `web`, `both`, or `none` based on the request and whether project files are available. A conservative heuristic is used if routing fails.
2. **Retrieve**: uploaded text/code files are split into chunks and indexed in an in-memory FAISS store using deterministic local feature-hash vectors. The index is session scoped and is not persisted. Retrieved excerpts are sent to Groq when included in a model prompt. These lexical vectors are lightweight, but less semantically capable than a trained embedding model.
3. **Search**: Tavily is called only when the route requires current or external information.
4. **Generate and validate**: the assistant cites retrieved URLs and validation checks that an answer exists, includes sources for web-grounded answers, and includes a fenced example for common code-writing requests. One repair attempt is allowed.
5. **Human review**: the answer is shown as a draft. The user can approve it or add feedback and request a revision.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Enter a Groq API key in the sidebar. Enter a Tavily API key when you want live search. Upload relevant project files in the sidebar to enable local retrieval. Without a Tavily key, choose requests that can be answered from uploaded context or general model knowledge.

## Current scope

Validation is a set of practical structural checks, not a proof of correctness. The local index is rebuilt from uploads in each app session and does not provide durable memory. Review is an explicit Streamlit approve/revise step; it does not pause a remotely running graph execution. Do not treat generated code as tested unless you run it in your environment.
