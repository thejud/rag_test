#!/usr/bin/env python3
"""
RAG-enhanced document query system.

This module provides a Retrieval-Augmented Generation (RAG) system that enables 
semantic search and question-answering over local documents using vector embeddings 
and various LLM providers.

Key Features:
- Document processing (PDF, HTML, Markdown, text files)
- Vector embeddings using SentenceTransformer
- Persistent vector storage with ChromaDB
- Multiple LLM providers (OpenAI, Anthropic Claude, Ollama)
- Chunking with overlap for better context preservation

Example Usage:
    # Index documents and ask a question
    python rag_query.py -d ./docs -p "What is machine learning?" -m openai
    
    # Use Claude with specific chunk parameters
    python rag_query.py -d ./knowledge_base -p "Explain neural networks" -m claude -c 800 -o 150
    
    # Reindex documents and use verbose logging
    python rag_query.py -d ./papers -p "What are transformers?" -m gpt-4 -r -v

Classes:
    DocumentProcessor: Handles loading and chunking of various document formats
    VectorStore: Manages ChromaDB operations for semantic search
    LLMClient: Provides unified interface for different LLM providers
    RAGSystem: Orchestrates the complete RAG pipeline

Environment Variables:
    OPENAI_API_KEY: OpenAI API key for GPT models
    ANTHROPIC_API_KEY: Anthropic API key for Claude models
    RAG_DEFAULT_DOCS: Default document directory (optional)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import anthropic
import chromadb
# Document processing
import markdown
import openai
import PyPDF2
import requests
from bs4 import BeautifulSoup
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles loading and processing different document types."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(f"{__name__}.DocumentProcessor")
    
    def load_documents(self, doc_paths: List[str]) -> List[Dict[str, Any]]:
        """Load documents from specified paths."""
        documents = []
        
        for doc_path in doc_paths:
            path = Path(doc_path)
            if path.is_file():
                doc = self._load_single_document(path)
                if doc:
                    documents.append(doc)
            elif path.is_dir():
                for file_path in path.rglob('*'):
                    if file_path.is_file() and self._is_supported_file(file_path):
                        doc = self._load_single_document(file_path)
                        if doc:
                            documents.append(doc)
        
        return documents
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported."""
        supported_extensions = {'.md', '.txt', '.pdf', '.html', '.htm'}
        return file_path.suffix.lower() in supported_extensions
    
    def _load_single_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single document based on its type."""
        try:
            content = ""
            file_type = file_path.suffix.lower()
            
            if file_type == '.md':
                content = self._load_markdown(file_path)
            elif file_type == '.txt':
                content = self._load_text(file_path)
            elif file_type == '.pdf':
                content = self._load_pdf(file_path)
            elif file_type in ['.html', '.htm']:
                content = self._load_html(file_path)
            
            if content.strip():
                return {
                    'content': content,
                    'source': str(file_path),
                    'type': file_type,
                    'modified_time': file_path.stat().st_mtime
                }
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
        
        return None
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load markdown file and convert to text."""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _load_text(self, file_path: Path) -> str:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file and extract text."""
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def _load_html(self, file_path: Path) -> str:
        """Load HTML file and extract text."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks."""
        chunks = []
        
        for doc in documents:
            content = doc['content']
            doc_chunks = self._split_text(content)
            
            for i, chunk in enumerate(doc_chunks):
                chunks.append({
                    'content': chunk,
                    'source': doc['source'],
                    'chunk_id': f"{doc['source']}_{i}",
                    'type': doc['type'],
                    'modified_time': doc['modified_time']
                })
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]

class VectorStore:
    """Handles vector database operations using Chroma."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = logging.getLogger(f"{__name__}.VectorStore")
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return
        
        # ChromaDB has a maximum batch size limit, so we need to process in batches
        batch_size = 5000  # Conservative batch size
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch_chunks)} chunks)")
            
            contents = [chunk['content'] for chunk in batch_chunks]
            embeddings = self.embedding_model.encode(contents).tolist()
            
            ids = [chunk['chunk_id'] for chunk in batch_chunks]
            metadatas = [{
                'source': chunk['source'],
                'type': chunk['type'],
                'modified_time': chunk['modified_time']
            } for chunk in batch_chunks]
            
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'content': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i] if 'distances' in results else 0.0
            })
        
        return search_results
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

class LLMClient:
    """Handles interactions with different LLM providers."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.logger = logging.getLogger(f"{__name__}.LLMClient")
        
        # Initialize clients if API keys are available
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        self.logger.debug(f"OpenAI API key present: {bool(openai_key)}")
        self.logger.debug(f"Anthropic API key present: {bool(anthropic_key)}")
        
        if openai_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                self.logger.debug("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        
        if anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                self.logger.debug("Anthropic client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic client: {e}")
                self.anthropic_client = None
    
    def query_llm(self, model: str, prompt: str, context: str) -> str:
        """Query the specified LLM with context and prompt."""
        full_prompt = self._build_prompt(context, prompt)
        
        if model.startswith('gpt-') or model.startswith('openai'):
            return self._query_openai(model, full_prompt)
        elif model.startswith('claude'):
            return self._query_claude(model, full_prompt)
        elif model.startswith('ollama') or 'qwen' in model.lower() or 'llama' in model.lower() or 'deepseek' in model.lower():
            return self._query_ollama(model, full_prompt)
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def _build_prompt(self, context: str, user_prompt: str) -> str:
        """Build the full prompt with context."""
        return f"""Based on the following context, please answer the user's question:

Context:
{context}

User Question: {user_prompt}

Please provide a helpful and accurate answer based on the provided context. If the context doesn't contain relevant information, please say so.
"""
    
    def _query_openai(self, model: str, prompt: str) -> str:
        """Query OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        # Use cost-effective model if not specified
        if model == 'openai':
            model = 'gpt-3.5-turbo'
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _query_claude(self, model: str, prompt: str) -> str:
        """Query Claude API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")
        
        # Use default Claude model if not specified
        if model == 'claude':
            model = 'claude-3-haiku-20240307'
        
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _query_ollama(self, model: str, prompt: str) -> str:
        """Query local Ollama instance."""
        if model == 'ollama':
            model = 'qwen2.5vl:7b'
        elif model.startswith('ollama:'):
            model = model.replace('ollama:', '')
        if not model:
            model = 'qwen2.5vl:7b'
        
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error connecting to Ollama: {e}")

class RAGSystem:
    """Main RAG system that orchestrates document processing, vector search, and LLM queries."""
    
    def __init__(self, doc_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
        self.doc_paths = doc_paths
        self.logger = logging.getLogger(f"{__name__}.RAGSystem")
    
    def index_documents(self, reindex: bool = False) -> None:
        """Index documents into the vector store."""
        if reindex:
            self.logger.info("Clearing existing index...")
            self.vector_store.clear_collection()
        
        self.logger.info(f"Loading documents from: {', '.join(self.doc_paths)}")
        documents = self.doc_processor.load_documents(self.doc_paths)
        self.logger.info(f"Loaded {len(documents)} documents")
        
        chunks = self.doc_processor.chunk_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")
        
        self.logger.info("Adding to vector store...")
        self.vector_store.add_documents(chunks)
        self.logger.info("Indexing complete!")
    
    def query(self, prompt: str, model: str, n_results: int = 5) -> str:
        """Process a query using RAG."""
        self.logger.info(f"Searching for relevant context...")
        search_results = self.vector_store.search(prompt, n_results)
        
        # Debug: Show found relevant chunks
        self.logger.debug("Found relevant chunks:")
        for i, result in enumerate(search_results, 1):
            self.logger.debug(f"Chunk {i} (distance: {result['distance']:.4f}):")
            self.logger.debug(f"  Source: {result['source']}")
            self.logger.debug(f"  Content: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
        
        context = "\n\n".join([
            f"Source: {result['source']}\n{result['content']}"
            for result in search_results
        ])
        
        self.logger.info(f"Found {len(search_results)} relevant chunks")
        
        # Build the full prompt and show it as debug output
        full_prompt = self.llm_client._build_prompt(context, prompt)
        self.logger.debug("Full prompt being sent to LLM:")
        self.logger.debug("-" * 50)
        self.logger.debug(full_prompt)
        self.logger.debug("-" * 50)
        
        self.logger.info(f"Querying {model}...")
        
        response = self.llm_client.query_llm(model, prompt, context)
        return response

def main():
    parser = argparse.ArgumentParser(description="RAG-enhanced LLM queries with local documents")
    
    parser.add_argument('-d', '--docs', action='append', default=[],
                        help="Document paths to include (can be files or directories)")
    parser.add_argument('-m', '--model', default='openai',
                       choices=['openai', 'gpt-3.5-turbo', 'gpt-4', 'claude', 'claude-3-haiku-20240307', 
                               'claude-3-sonnet-20240229', 'ollama', 'qwen2.5vl:7b', 'llama3.2:latest', 'deepseek-r1:1.5b'],
                       help="LLM model to use")
    parser.add_argument('-p', '--prompt', required=True,
                       help="The query prompt")
    parser.add_argument('-n', '--num-results', type=int, default=5,
                       help="Number of relevant chunks to include in context")
    parser.add_argument('-r', '--reindex', action='store_true',
                       help="Reindex all documents before querying")
    parser.add_argument('-c', '--chunk-size', type=int, default=1000,
                       help="Size of text chunks for processing")
    parser.add_argument('-o', '--chunk-overlap', type=int, default=200,
                       help="Overlap between chunks")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="Enable verbose logging (DEBUG level)")
    parser.add_argument('-q', '--quiet', action='store_true',
                       help="Quiet mode (WARNING level only)")
    
    args = parser.parse_args()
    
    # Configure logging level based on arguments
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Set default document paths if none provided
    if not args.docs:
        default_docs = os.getenv('RAG_DEFAULT_DOCS', ["docs"])
        args.docs = [default_docs]
    
    # Expand tilde in paths
    args.docs = [os.path.expanduser(path) for path in args.docs]
    
    logger.debug(f"Document paths: {args.docs}")
    
    # Initialize RAG system
    rag_system = RAGSystem(
        doc_paths=args.docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Index documents if needed
    try:
        if args.reindex:
            rag_system.index_documents(reindex=True)
        else:
            # Check if we have any documents indexed
            try:
                test_results = rag_system.vector_store.search("test", n_results=1)
                if not test_results:
                    logger.info("No documents found in index. Indexing documents...")
                    rag_system.index_documents()
            except:
                logger.info("Indexing documents...")
                rag_system.index_documents()
        
        # Process query
        response = rag_system.query(args.prompt, args.model, args.num_results)
        print("\n" + "="*50 + "\n")
        print(response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
