import streamlit as st
from code_assistant.vector_db.chroma_store import ChromaStore
from code_assistant.llm.qwen_llm import QwenLLM   # your updated class

# -------------------------------------------------
# INITIALIZE MODEL & DATABASE ONLY ONCE
# -------------------------------------------------
@st.cache_resource
def load_llm():
    return QwenLLM()

@st.cache_resource
def load_db():
    return ChromaStore()

llm = load_llm()
db = load_db()

# -------------------------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(page_title="Code Assistant", page_icon="💻", layout="wide")
st.title("💻 CodeAssistant — Local LLM + Vector Search")

st.write("Ask anything related to **code, debugging, or software architecture**.")

# -------------------------------------------------
# CHAT MEMORY
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
user_input = st.chat_input("Ask your question…")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------------------------
    # 1. VECTOR SEARCH
    # -------------------------------------------------
    vector_result = db.search(user_input)

    # -------------------------------------------------
    # 2. LLM ANSWER
    # -------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = llm.generate_from_chunks(user_input, vector_result)
            assistant_text = answer

            st.markdown(assistant_text)

    # Save assistant answer
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
