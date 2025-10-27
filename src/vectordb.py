from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import uuid
import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks for better retrieval.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # Initialize a text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,  # Add a small overlap for context continuity
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split the text
        chunks = text_splitter.split_text(text)
        return chunks
       
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents and add them to the vector database.

        Args:
            documents: List of documents with 'content' and optional 'metadata'
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        print(f"Ingesting {len(documents)} documents...")
        for doc in documents:
            content = doc.get("content")
            metadata = doc.get("metadata", {})
            
            if not content:
                print(f"Skipping document with no content: {metadata.get('source', 'N/A')}")
                continue
                
            # 1. Chunk the document content
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                # Create metadata for each chunk
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                all_metadatas.append(chunk_metadata)
                
                # Create a unique ID for each chunk
                all_ids.append(str(uuid.uuid4()))
        
        if not all_chunks:
            print("No content to ingest.")
            return

        # 2. Create embeddings for all chunks
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks)
        
        # 3. Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Successfully ingested {len(all_chunks)} chunks into the vector database.")
    # Inside the VectorDB class in src/vectordb.py
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Find documents similar to the query.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary with search results
        """
        # 1. Create an embedding for the query
        # Note: encode() expects a list, so we pass [query]
        query_embedding = self.embedding_model.encode([query])
        
        # 2. Search the ChromaDB collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 3. Return results in the expected format
        # Chroma returns results nested in a list (one per query). We extract the first [0].
        return {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
            "ids": results.get("ids", [[]])[0]
        }
        
