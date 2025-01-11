import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def clean_text(text):
    # Enhanced text cleaning
    text = text.replace("\n", " ")
    text = ' '.join(text.split())  # Remove multiple spaces
    text = text.replace("′", "'").replace("�", "")
    text = text.strip()
    return text

def format_source_reference(doc):
    """Format source document reference in a clean, readable way"""
    source = doc.metadata.get('source', 'Unknown')
    page = doc.metadata.get('page', 1)
    preview = clean_text(doc.page_content[:300])  # Shorter preview for clarity
    
    return f"""
---
**Source**: {os.path.basename(source)} (Page {page})
**Preview**: {preview}...
"""

def initialize_rag_components():
    # Load and process documents
    loader = DirectoryLoader(
        './data',
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    st.write(f"📚 Loaded {len(documents)} documents")
    
    # Improved text splitting strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    # Initialize embeddings with better model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster, lighter model
        model_kwargs={'device': 'cpu'}
    )
    
    # Enhanced vector store configuration
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Improved LLM configuration
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={
            "temperature": 0.5,  # Reduced for more focused responses
            "max_length": 2048,
            "top_p": 0.95,
            "do_sample": True,
        }
    )
    
    # Custom prompt template for better context utilization
    prompt_template = """
    Context: {context}
    
    Chat History: {chat_history}
    
    Current Question: {question}
    
    Please provide a clear, accurate response based on the context provided. If the answer isn't directly supported by the context, acknowledge this and provide the most relevant information available. Include specific references where appropriate.
    
    Response:"""
    
    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )
    
    # Enhanced memory configuration
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Simplified retrieval chain configuration
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Just using basic k parameter
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=True
    )
    
    return qa_chain

def main():
    st.write(f"🔑 HF Token status: {'HUGGINGFACEHUB_API_TOKEN' in os.environ}")
    
    try:
        qa_chain = initialize_rag_components()
        
        # Display chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle user input
        if prompt := st.chat_input("Ask me anything about the documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("📚 Searching documents..."):
                    response = qa_chain({"question": prompt})
                    
                    # Format the main response
                    response_text = clean_text(response["answer"])
                    
                    # Add source references in a collapsible section
                    if "source_documents" in response and response["source_documents"]:
                        response_text += "\n\n<details><summary>📚 Source References</summary>\n\n"
                        for doc in response["source_documents"]:
                            response_text += format_source_reference(doc)
                        response_text += "</details>"
                    
                    st.markdown(response_text, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Please check your configuration and try again.")

if __name__ == "__main__":
    main()