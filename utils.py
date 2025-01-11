import os

def setup_environment():
    """
    Set up the required environment variables and directories
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('./data'):  # Changed to Windows path format
        os.makedirs('./data')
    
    # Create directory for vector store
    if not os.path.exists('./chroma_db'):  # Changed to Windows path format
        os.makedirs('./chroma_db')
    
    # Check for HUGGINGFACEHUB_API_TOKEN
    if 'HUGGINGFACEHUB_API_TOKEN' not in os.environ:
        raise EnvironmentError(
            "Please set HUGGINGFACEHUB_API_TOKEN environment variable"
        )