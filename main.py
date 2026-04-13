import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_ollama import OllamaLLM
import requests
from pypdf import PdfReader

# 🔹 LLM
llm = OllamaLLM(model="llama3")

# 🔹 State
class AgentState(TypedDict):
    question: str
    answer: str
    history: List[str]
    step: int
    pdf_text: str

# 🔹 Nodes

def explain_node(state):
    response = llm.invoke(f"Explain clearly: {state['question']}")
    return {
        "answer": response,
        "history": state["history"] + [f"User: {state['question']}", f"AI: {response}"],
        "step": state["step"] + 1
    }

def math_node(state):
    try:
        result = eval(state["question"])
        answer = f"Answer: {result}"
    except:
        answer = "Invalid math"
    return {
        "answer": answer,
        "history": state["history"] + [f"User: {state['question']}", f"AI: {answer}"],
        "step": state["step"] + 1
    }

def search_node(state):
    query = state["question"]
    try:
        res = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        data = res.json()
        answer = data.get("AbstractText", "No result found")
    except:
        answer = "Search failed"
    return {
        "answer": answer,
        "history": state["history"] + [f"User: {query}", f"AI: {answer}"],
        "step": state["step"] + 1
    }

def pdf_node(state):
    if not state.get("pdf_text"):
        return {"answer": "Upload a PDF first", "history": state["history"], "step": state["step"] + 1}

    response = llm.invoke(f"""
    Based on this document:
    {state['pdf_text'][:3000]}

    Answer: {state['question']}
    """)

    return {
        "answer": response,
        "history": state["history"] + [f"User: {state['question']}", f"AI: {response}"],
        "step": state["step"] + 1
    }

# 🔹 Router
def router(state):
    q = state["question"]

    decision = llm.invoke(f"""
    Classify:
    {q}

    Options:
    math, explain, search, pdf

    Return only one word.
    """)

    decision = decision.lower()

    if "math" in decision:
        return "math"
    elif "search" in decision:
        return "search"
    elif "pdf" in decision:
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
        "explain": "explain",
        "search": "search",
        "pdf": "pdf"
    }
)

graph.add_edge("math", END)
graph.add_edge("explain", END)
graph.add_edge("search", END)
graph.add_edge("pdf", END)

app_graph = graph.compile()

# 🔹 Streamlit UI
st.set_page_config(page_title="🔥 AI Super Agent", page_icon="🤖")

st.title("🤖 AI Super Agent")

# Memory
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# 📄 PDF Upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    st.session_state.pdf_text = text
    st.success("PDF loaded!")

# 💬 Chat input
user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = app_graph.invoke({
        "question": user_input,
        "history": st.session_state.history,
        "step": 0,
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