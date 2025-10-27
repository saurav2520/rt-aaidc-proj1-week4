import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from glob import glob
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


# In src/app.py
def load_documents(data_dir: str = "data/") -> List[Dict[str, Any]]:
    """
    Load documents from the data directory.
    Handles .txt and .pdf files.

    Returns:
        List of document dictionaries with 'content' and 'metadata'
    """
    documents = []
    print(f"Loading documents from {data_dir}...")
    
    # Use glob to find all files in the data_dir
    for file_path in glob(os.path.join(data_dir, "*.*")):
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            file_name = os.path.basename(file_path)

            if ext.lower() == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
            elif ext.lower() == ".pdf":
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
            else:
                print(f"Skipping unsupported file type: {file_name}")
                continue
            
            # Convert LangChain Document objects to dictionaries
            for doc in loaded_docs:
                documents.append({
                    "content": doc.page_content,
                    "metadata": {"source": file_name, **doc.metadata} # Add source filename
                })
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    print(f"Loaded {len(documents)} document pages/files.")
    return documents
    


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        rag_template = """
        You are an expert assistant. Your task is to answer the user's question based *only* on the provided context.
        
        Follow these rules:
        1. Read the context carefully.
        2. If the context contains the answer, formulate a clear and concise response based solely on that information.
        3. Do not use any external knowledge or make up information.
        4. If the context does not contain the information needed to answer the question, state that you do not have enough information from the documents.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        self.prompt_template = ChatPromptTemplate.from_template(rag_template)

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM
        """
        # 1. Retrieve relevant context chunks
        search_results = self.vector_db.search(query=input, n_results=n_results)
        
        # 2. Combine the retrieved document chunks into a single context string
        context = "\n\n---\n\n".join(search_results["documents"])
        
        # Create the input for the chain (matching the prompt template)
        chain_input = {
            "context": context,
            "question": input
        }
        
        # 3. Use self.chain.invoke() to generate the response
        llm_answer = self.chain.invoke(chain_input)
        
        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
