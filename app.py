import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Page Setup
st.title(" Devxyn AI Chatbot (RAG)")
st.caption("Ask me anything about Muhammad Awais or Devxyn!")

# 2. AI Memory Loading
@st.cache_resource
def load_knowledge():
    loader = TextLoader("devxyn_knowledge.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

db = load_knowledge()

# 3. Chat History (Context Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. User Input and AI Response
user_query = st.chat_input("Ask a question...")

if user_query:
    # Show message of user
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Find from Vector DB
    results = db.similarity_search(user_query)
    best_answer = results[0].page_content if results else "Sorry, I don't know about that."
    
    # Show response of AI
    with st.chat_message("assistant"):
        st.markdown(f"**Found Info:** {best_answer}")
    st.session_state.messages.append({"role": "assistant", "content": f"**Found Info:** {best_answer}"})
