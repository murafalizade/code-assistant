import streamlit as st
from code_assistant.llm.qrok_qwen_llm import GroqQwenLLM
from code_assistant.vector_db.chroma_store import ChromaStore
from code_assistant.llm.deepseek_llm import DeepSeekLLM

@st.cache_resource
def load_llm():
    return GroqQwenLLM()

@st.cache_resource
def load_db():
    return ChromaStore()

llm = load_llm()
db = load_db()


st.set_page_config(page_title="Bizden Code Assistant", page_icon="💻", layout="wide")
st.title("Bizden Code Assistant")

st.write("Ask anything related to **code, debugging, or software architecture**.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    vector_result = db.search(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = llm.generate_from_chunks(user_input, vector_result)
            assistant_text = answer

            st.markdown(assistant_text)

    # Save assistant answer
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
