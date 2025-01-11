# RAG Chatbot with Streamlit and LangChain

This repository contains a **Retrieval-Augmented Generation (RAG) Chatbot** built using **Streamlit**, **LangChain**, and **ChromaDB**. The chatbot leverages advanced natural language models to answer user queries based on the content of uploaded PDF documents.

## Features

- **Document Loading**: Automatically processes PDF files using `PyPDFLoader`.
- **Text Splitting**: Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- **ChromaDB Integration**: Efficient vector store for storing and retrieving document embeddings.
- **Hugging Face Embeddings and LLMs**:
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - LLM: `google/flan-t5-large`
- **Conversational Memory**: Maintains a conversation context using `ConversationBufferMemory`.
- **Custom Prompt Templates**: Enhances chatbot responses with custom prompts.
- **Interactive Chat Interface**: Powered by Streamlit with real-time chat.

## Setup Instructions

### Prerequisites

1. Python 3.9+
2. Virtual environment tools (`venv` or `conda`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Set up a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Add your Hugging Face API token as an environment variable:
     ```bash
     export HUGGINGFACEHUB_API_TOKEN=<your_token>
     ```

5. Prepare the required directories:
   ```bash
   python -c "from utils import setup_environment; setup_environment()"
   ```

### Running the Application

1. Place your PDF documents in the `./data` directory.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Interact with the chatbot in your browser.

## Files

- **`app.py`**: Main application script that defines the chatbot interface and logic.
- **`utils.py`**: Utility script to set up directories and check environment variables.

## Usage

Upload your PDF files to the `./data` directory. The chatbot will process the documents, index their contents, and answer questions based on them. The responses include references to the relevant document sections.

## Citations and Acknowledgements

This project uses the following tools, models, and APIs:

- **Language Models and Embeddings**:
  - `sentence-transformers/all-MiniLM-L6-v2` and `google/flan-t5-large` from Hugging Face
- **APIs**:
  - Hugging Face API (requires token)
- **Vector Store**:
  - [ChromaDB](https://github.com/chroma-core/chroma)

Additionally, assistance was sought from:

- **ChatGPT** by OpenAI for generating code and content explanations.
- **Claude AI** by Anthropic for iterative refinements.
- Online resources for integrating LangChain and Streamlit components.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

If you’d like any additional details or adjustments, let me know!
