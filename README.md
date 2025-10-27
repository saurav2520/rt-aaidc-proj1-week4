RAG-Based AI Assistant - AAIDC Project 1 

🤖 What is this?
This is a learning template for building a RAG (Retrieval-Augmented Generation) AI assistant. RAG systems combine document search with AI chat - they can answer questions about your specific documents by finding relevant information and using it to generate responses.

Think of it as: ChatGPT that knows about YOUR documents and can answer questions about them.

🤔 Why RAG?
Use Your Own Data: Answer questions about private documents, new articles, or any information not in the LLM's original training.

Reduce Hallucinations: The AI is instructed to answer only based on the documents you provide, making it more factual and trustworthy.

Always Up-to-Date: You can add new documents at any time to keep your assistant's knowledge current without expensive retraining.

🎯 What you'll build
By completing this project, you'll have an AI assistant that can:

📄 Load your documents (PDFs, text files, etc.)

🔍 Search through them to find relevant information

💬 Answer questions using the information it found

🧠 Combine multiple sources to give comprehensive answers

💡 How It Works (The Data Flow)
This diagram shows the "Query Pipeline" you will build in Step 7.

[ Your Question ]
       │
       ▼
[ 1. Search (VectorDB) ] ──> Finds relevant "Context" chunks from your docs
       │
       ▼
[ 2. Augment (Prompt) ] ──> Combines "Context" + "Your Question"
       │
       ▼
[ 3. Generate (LLM) ] ──> Creates an answer based *only* on the context
       │
       ▼
[ Final Answer ]
📝 Implementation Steps
You will implement a complete RAG system by filling in the TODO sections in the code. The 7 steps are grouped by the file you'll be working in.

Part A: The Vector Database (src/vectordb.py)
First, you'll build the "memory" of your assistant.

Step 3: Implement Text Chunking

Step 4: Implement Document Ingestion

Step 5: Implement Similarity Search

Part B: The RAG Application (src/app.py)
Next, you'll build the "brain" that uses the memory.

Step 1: Prepare Your Documents (in the data/ folder)

Step 2: Implement Document Loading

Step 6: Implement RAG Prompt Template

Step 7: Implement RAG Query Pipeline

Step 1: Prepare Your Documents
Location: data/ directory

The data/ directory contains sample files. Replace these with your own documents relevant to your domain:

data/
├── your_topic_1.txt
├── your_topic_2.pdf
└── ...etc
Each file should contain text content you want your RAG system to search through.

Step 2: Implement Document Loading
Location: src/app.py

Python

def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    # TODO: Implement document loading
    # HINT: Read the documents from the data directory
    # HINT: Return a list of documents
    # HINT: Your implementation depends on the type of documents you are using (.txt, .pdf, etc.)

    # Your implementation here
    return results
What you need to do:

Read files from the data/ directory.

Load the content of each file into memory (use PyPDFLoader for PDFs, TextLoader for .txt, etc.).

Return a list of document dictionaries with content and metadata keys.

Step 3: Implement Text Chunking
Location: src/vectordb.py

Python

def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into smaller chunks for better retrieval.
  
    Args:
        text: Input text to chunk
        chunk_size: Approximate number of characters per chunk
  
    Returns:
        List of text chunks
    """
    # TODO: Your implementation here
What you need to do:

Choose a chunking strategy.

Split the input text into manageable chunks.

Return a list of text strings.

Hint: LangChain's RecursiveCharacterTextSplitter is a great choice.

Step 4: Implement Document Ingestion
Location: src/vectordb.py

Python

def add_documents(self, documents: List[Dict[str, Any]]) -> None:
    """
    Process documents and add them to the vector database.
  
    Args:
        documents: List of documents with 'content' and optional 'metadata'
    """
    # TODO: Your implementation here
What you need to do:

Loop through the list of documents.

Use your chunk_text() method to split each document's content.

Create a unique ID for each chunk (e.g., using the uuid library).

Create embeddings for all chunks using self.embedding_model.encode().

Store the ids, documents (chunks), and metadatas in ChromaDB using self.collection.add().

Step 5: Implement Similarity Search
Location: src/vectordb.py

Python

def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Find documents similar to the query.
  
    Args:
        query: Search query
        n_results: Number of results to return
  
    Returns:
        Dictionary with search results
    """
    # TODO: Your implementation here
What you need to do:

Create an embedding for the query using self.embedding_model.encode([query]).

Search the ChromaDB collection using self.collection.query().

Return the results in the required dictionary format.

Step 6: Implement RAG Prompt Template
Location: src/app.py

Python

# Create RAG prompt template
# TODO: Implement your RAG prompt template
# HINT: Use ChatPromptTemplate.from_template() with a template string
# HINT: Your template should include placeholders for {context} and {question}
# HINT: Design your prompt to effectively use retrieved context to answer questions
self.prompt_template = None  # Your implementation here
What you need to do:

Design a prompt template string that instructs the LLM.

Include placeholders for {context} and {question}.

Crucially: Instruct the LLM to answer only based on the context, and to say "I don't know" if the answer isn't present.

Create the template object using ChatPromptTemplate.from_template().

Step 7: Implement RAG Query Pipeline
Location: src/app.py

Python

def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Answer questions using retrieved context.
  
    Args:
        question: User's question
        n_results: Number of context chunks to retrieve
  
    Returns:
        Dictionary with answer and context information
    """
    # TODO: Your implementation here
What you need to do:

Search: Use self.vector_db.search() to find relevant context chunks.

Combine: Join the retrieved chunks (e.g., search_results["documents"]) into a single context string.

Generate: Call self.chain.invoke() with the context string and the user's question.

Return: Return a dictionary containing the final answer and any other info you want to show.

🧪 Testing Your Implementation
Test Individual Components
(As recommended in the template, you can create a separate test.py file to run these)

Test chunking:

Python

from src.vectordb import VectorDB
vdb = VectorDB(init_empty=True) # Use a flag to stop __init__ from loading
chunks = vdb.chunk_text("Your test text here...")
print(f"Created {len(chunks)} chunks")
Test ingestion and search:

Python

vdb = VectorDB(collection_name="test_collection") # Use a temporary DB
documents = [{"content": "The quick brown fox jumps over the lazy dog.", "metadata": {"title": "Test"}}]
vdb.add_documents(documents)
results = vdb.search("What did the fox jump over?")
print(f"Found {len(results['documents'])} results: {results['documents']}")
Test Full System
Once all steps are implemented, run the main application:

Bash

python src/app.py
Try these example questions:

Factual Question: "What is [a specific topic from your documents]?"

"I Don't Know" Question: "What is the capital of Mars?" (This should fail gracefully)

Synthesis Question: "Compare [concept A] and [concept B] from your documents."

🚀 Setup Instructions
Prerequisites
Before starting, make sure you have:

Python 3.8 or higher installed

An API key from one of these providers:

OpenAI

Groq

Google AI

Quick Setup
Clone and install dependencies:

Bash

git clone [your-repo-url]
cd rt-aaidc-project1-template
pip install -r requirements.txt
Configure your API key:

Bash

# Create environment file (choose the method that works on your system)
cp .env.example .env    # Linux/Mac
copy .env.example .env  # Windows
Edit .env and add your API key. Only one key is needed.

Code snippet

# --- CHOOSE ONE PROVIDER ---

OPENAI_API_KEY=your_key_here
# OR
GROQ_API_KEY=your_key_here  
# OR
GOOGLE_API_KEY=your_key_here
⚠️ Common Issues / Troubleshooting
ModuleNotFoundError: No module named 'langchain': You forgot to run pip install -r requirements.txt in your virtual environment.

AuthenticationError or API key not found:

Make sure you created the .env file (it's hidden by default).

Ensure the variable name in .env (e.g., OPENAI_API_KEY) exactly matches the name in app.py.

No documents were loaded: Make sure you have placed your .txt or .pdf files inside the data/ directory.

Slow first run: The first time you run the app, it needs to download the embedding model (e.g., all-MiniLM-L6-v2), which can be several hundred MB. This is a one-time process.

🔧 Implementation Freedom
Important: This template uses specific packages (ChromaDB, LangChain, HuggingFace Transformers), but you are completely free to use whatever you prefer!

Vector Databases: FAISS, Pinecone, Weaviate, Qdrant

LLM Frameworks: LlamaIndex, Direct API calls, Ollama

Embedding Models: OpenAI ada-002, Cohere, other Hugging Face models

Text Processing: Custom logic, spaCy, NLTK

📁 Project Structure
rt-aaidc-project1-template/
├── src/
│   ├── app.py           # Main RAG application (Steps 2, 6-7)
│   └── vectordb.py      # Vector database wrapper (Steps 3-5)
├── data/               # Your documents go here (Step 1)
│   ├── *.txt          
├── requirements.txt    # All dependencies
├── .env.example       # Environment template
└── README.md          # This guide
🎓 Learning Objectives
By completing this project, you will:

✅ Understand RAG architecture and data flow

✅ Implement text chunking strategies

✅ Work with vector databases and embeddings

✅ Build LLM-powered applications with LangChain

✅ Handle multiple API providers

✅ Create production-ready AI applications

🏁 Success Criteria
Your implementation is complete when:

✅ You can load your own documents from the data/ folder.

✅ The system chunks and embeds those documents into ChromaDB.

✅ vectordb.search() returns relevant chunks for a query.

✅ The RAG system generates answers based only on the retrieved context.

✅ You can ask questions and get meaningful, factual responses from your documents.

Good luck building your RAG system! 🚀