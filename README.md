# Task 4: RAG Chatbot System

### Objective
To develop a Retrieval-Augmented Generation (RAG) chatbot that answers user queries based on a custom knowledge base.

### Methodology
* **Document Processing:** Used LangChain's CharacterTextSplitter to break the custom text file into manageable chunks.
* **Vector Database:** Created embeddings using HuggingFace (`all-MiniLM-L6-v2`) and stored them in a FAISS vector store.
* **UI Integration:** Built an interactive chat interface using Gradio/Streamlit for real-time querying.

### Key Results
* The chatbot successfully retrieves context from the custom dataset before generating answers.
* Eliminated AI hallucinations by strictly grounding responses in the provided knowledge base.
