import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import requests
from pypdf import PdfReader

# 🔹 Fake LLM (for cloud deployment)
class FakeLLM:
    def invoke(self, prompt):
        return "🤖 Demo AI Response: This is a simulated answer (Ollama not available in cloud)."

llm = FakeLLM()

# 🔹 State
class AgentState(TypedDict):
    question: str
    answer: str
    history: List[str]
    pdf_text: str

# 🔹 Nodes

def explain_node(state):
    response = llm.invoke(f"Explain: {state['question']}")
    return {
        "answer": response,
        "history": state["history"] + [f"User: {state['question']}", f"AI: {response}"]
    }

def math_node(state):
    try:
        result = eval(state["question"])
        answer = f"🧮 Answer: {result}"
    except:
        answer = "❌ Invalid math expression"
    return {
        "answer": answer,
        "history": state["history"] + [f"User: {state['question']}", f"AI: {answer}"]
    }

def search_node(state):
    query = state["question"]
    try:
        res = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        data = res.json()
        answer = data.get("AbstractText", "🌐 No result found")
    except:
        answer = "❌ Search failed"
    return {
        "answer": answer,
        "history": state["history"] + [f"User: {query}", f"AI: {answer}"]
    }

def pdf_node(state):
    if not state.get("pdf_text"):
        return {
            "answer": "📄 Please upload a PDF first",
            "history": state["history"]
        }

    response = llm.invoke(f"""
    Based on document:
    {state['pdf_text'][:2000]}

    Answer: {state['question']}
    """)

    return {
        "answer": response,
        "history": state["history"] + [f"User: {state['question']}", f"AI: {response}"]
    }

# 🔹 Router

def router(state):
    q = state["question"].lower()

    # simple rule-based routing (reliable for demo)
    if any(char.isdigit() for char in q):
        return "math"
    elif "search" in q or "who" in q or "what" in q:
        return "search"
    elif "pdf" in q:
        return "pdf"
    return "explain"

# 🔹 Graph

graph = StateGraph(AgentState)

graph.add_node("explain", explain_node)
graph.add_node("math", math_node)
graph.add_node("search", search_node)
graph.add_node("pdf", pdf_node)

graph.add_conditional_edges(
    "__start__",
    router,
    {
        "math": "math",
        "search": "search",
        "pdf": "pdf",
        "explain": "explain"
    }
)

graph.add_edge("math", END)
graph.add_edge("search", END)
graph.add_edge("pdf", END)
graph.add_edge("explain", END)

app_graph = graph.compile()

# 🔹 UI

st.set_page_config(page_title="🔥 AI Super Agent", page_icon="🤖")
st.title("🤖 AI Super Agent")

# Memory
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# 📄 Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    st.session_state.pdf_text = text
    st.success("✅ PDF loaded")

# 💬 Chat input
user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = app_graph.invoke({
        "question": user_input,
        "history": st.session_state.history,
        "pdf_text": st.session_state.pdf_text
    })

    st.session_state.history = result["history"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
