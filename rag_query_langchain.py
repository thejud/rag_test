#!/usr/bin/env python3
"""
RAG-enhanced document query system with LangChain integration.

This module provides a Retrieval-Augmented Generation (RAG) system that enables 
semantic search and question-answering over local documents using LangChain 
components with ChromaDB and multiple LLM providers (OpenAI, Anthropic, Ollama).

Key Features:
- Document processing (PDF, HTML, Markdown, text files) using LangChain
- Vector embeddings using SentenceTransformer
- Persistent vector storage with ChromaDB via LangChain
- Multiple LLM providers: OpenAI GPT models, Anthropic Claude, Ollama local models
- Chunking with overlap for better context preservation

Example Usage:
    # Using OpenAI GPT models
    python rag_query_langchain.py -d ./docs -p "What is machine learning?" -m gpt-4
    
    # Using Anthropic Claude models  
    python rag_query_langchain.py -d ./knowledge_base -p "Explain neural networks" -m claude-3-haiku-20240307
    
    # Using Ollama local models
    python rag_query_langchain.py -d ./papers -p "What are transformers?" -m qwen2.5vl:7b -r -v

Classes:
    LangChainDocumentProcessor: Handles loading and chunking using LangChain
    LangChainVectorStore: Manages ChromaDB operations via LangChain
    LangChainRAGSystem: Orchestrates the complete RAG pipeline with LangChain

Environment Variables:
    OPENAI_API_KEY: OpenAI API key for GPT models
    ANTHROPIC_API_KEY: Anthropic API key for Claude models
    RAG_DEFAULT_DOCS: Default document directory (optional)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# LangChain imports
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import (DirectoryLoader, PyPDFLoader,
                                                  TextLoader,
                                                  UnstructuredHTMLLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


class LangChainDocumentProcessor:
    """Handles loading and processing different document types using LangChain."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(f"{__name__}.LangChainDocumentProcessor")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, doc_paths: List[str]) -> List[Document]:
        """Load documents from specified paths using LangChain loaders."""
        documents = []
        
        for doc_path in doc_paths:
            path = Path(doc_path)
            if path.is_file():
                docs = self._load_single_file(path)
                documents.extend(docs)
            elif path.is_dir():
                docs = self._load_directory(path)
                documents.extend(docs)
        
        self.logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single document based on its type."""
        try:
            file_type = file_path.suffix.lower()
            
            if file_type == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif file_type == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_type == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_type in ['.html', '.htm']:
                loader = UnstructuredHTMLLoader(str(file_path))
            else:
                self.logger.warning(f"Unsupported file type: {file_type}")
                return []
            
            documents = loader.load()
            self.logger.debug(f"Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def _load_directory(self, dir_path: Path) -> List[Document]:
        """Load all supported documents from a directory."""
        documents = []
        
        # Load different file types
        file_types = {
            '*.md': UnstructuredMarkdownLoader,
            '*.txt': TextLoader,
            '*.pdf': PyPDFLoader,
            '*.html': UnstructuredHTMLLoader,
            '*.htm': UnstructuredHTMLLoader,
        }
        
        for pattern, loader_class in file_types.items():
            try:
                if pattern == '*.pdf':
                    # PDF files need special handling
                    for pdf_file in dir_path.rglob(pattern):
                        loader = PyPDFLoader(str(pdf_file))
                        docs = loader.load()
                        documents.extend(docs)
                else:
                    # Use DirectoryLoader for other file types
                    loader = DirectoryLoader(
                        str(dir_path),
                        glob=pattern,
                        loader_cls=loader_class,
                        recursive=True
                    )
                    docs = loader.load()
                    documents.extend(docs)
            except Exception as e:
                self.logger.error(f"Error loading {pattern} files from {dir_path}: {e}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using LangChain text splitter."""
        chunks = self.text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class LangChainVectorStore:
    """Handles vector database operations using LangChain Chroma integration."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(f"{__name__}.LangChainVectorStore")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
    
    def add_documents(self, chunks: List[Document]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Add documents in batches to avoid memory issues
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch_chunks)} chunks)")
            
            self.vectorstore.add_documents(batch_chunks)
        
        self.logger.info("Successfully added all chunks to vector store")
    
    def search(self, query: str, n_results: int = 5) -> List[Document]:
        """Search for relevant document chunks."""
        results = self.vectorstore.similarity_search(query, k=n_results)
        return results
    
    def as_retriever(self, n_results: int = 5):
        """Return a retriever for use with LangChain chains."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": n_results}
        )
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            self.logger.info("Collection cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")


class LangChainRAGSystem:
    """Main RAG system using LangChain components with Ollama."""
    
    def __init__(self, doc_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
        self.doc_processor = LangChainDocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = LangChainVectorStore()
        self.doc_paths = doc_paths
        self.logger = logging.getLogger(f"{__name__}.LangChainRAGSystem")
        
        # LLM will be initialized when needed
        self.llm = None
        self.qa_chain = None
    
    def _initialize_llm(self, model: str) -> None:
        """Initialize the appropriate LLM based on model type."""
        try:
            # Determine LLM provider and initialize accordingly
            if model.startswith('gpt-') or model.startswith('openai'):
                self.llm = self._initialize_openai_llm(model)
                provider = "OpenAI"
            elif model.startswith('claude'):
                self.llm = self._initialize_anthropic_llm(model)
                provider = "Anthropic"
            else:
                # Default to Ollama for other models
                self.llm = self._initialize_ollama_llm(model)
                provider = "Ollama"
            
            # Create a custom prompt template
            template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            # Create the QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            self.logger.info(f"Initialized {provider} LLM with model: {model}")
            
        except Exception as e:
            raise ValueError(f"Error initializing LLM: {e}")
    
    def _initialize_openai_llm(self, model: str) -> ChatOpenAI:
        """Initialize OpenAI LLM."""
        # Clean up model name
        if model == 'openai':
            model = 'gpt-3.5-turbo'
        elif model.startswith('openai:'):
            model = model.replace('openai:', '')
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=1000,
            api_key=api_key
        )
    
    def _initialize_anthropic_llm(self, model: str) -> ChatAnthropic:
        """Initialize Anthropic LLM."""
        # Clean up model name
        if model == 'claude':
            model = 'claude-3-haiku-20240307'
        elif model.startswith('claude:'):
            model = model.replace('claude:', '')
        
        # Check for API key
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        return ChatAnthropic(
            model=model,
            temperature=0.7,
            max_tokens=1000,
            api_key=api_key
        )
    
    def _initialize_ollama_llm(self, model: str) -> OllamaLLM:
        """Initialize Ollama LLM."""
        # Clean up model name
        if model.startswith('ollama:'):
            model = model.replace('ollama:', '')
        elif model == 'ollama':
            model = 'qwen2.5vl:7b'
        
        return OllamaLLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.7
        )
    
    def index_documents(self, reindex: bool = False) -> None:
        """Index documents into the vector store."""
        if reindex:
            self.logger.info("Clearing existing index...")
            self.vector_store.clear_collection()
        
        self.logger.info(f"Loading documents from: {', '.join(self.doc_paths)}")
        documents = self.doc_processor.load_documents(self.doc_paths)
        
        if not documents:
            self.logger.warning("No documents loaded!")
            return
        
        chunks = self.doc_processor.chunk_documents(documents)
        
        if not chunks:
            self.logger.warning("No chunks created!")
            return
        
        self.logger.info("Adding to vector store...")
        self.vector_store.add_documents(chunks)
        self.logger.info("Indexing complete!")
    
    def query(self, prompt: str, model: str, n_results: int = 5) -> str:
        """Process a query using RAG with LangChain."""
        # Initialize LLM if needed
        if self.llm is None or self.qa_chain is None:
            self._initialize_llm(model)
        
        # Update retriever with new n_results
        self.qa_chain.retriever = self.vector_store.as_retriever(n_results)
        
        self.logger.info(f"Processing query with {model}...")
        
        try:
            # Run the query
            result = self.qa_chain.invoke({"query": prompt})
            
            # Log source documents for debugging
            if "source_documents" in result:
                self.logger.debug(f"Found {len(result['source_documents'])} source documents")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    self.logger.debug(f"Source {i}: {source}")
                    self.logger.debug(f"Content: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")
            
            return result["result"]
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="RAG-enhanced LLM queries with LangChain and Ollama")
    
    parser.add_argument('-d', '--docs', action='append', default=[],
                        help="Document paths to include (can be files or directories)")
    parser.add_argument('-m', '--model', default='qwen2.5vl:7b',
                       help="LLM model to use. Supports OpenAI (gpt-3.5-turbo, gpt-4), Anthropic (claude, claude-3-haiku-20240307), and Ollama (qwen2.5vl:7b, llama3.2:latest, deepseek-r1:1.5b)")
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
        default_docs = os.getenv('RAG_DEFAULT_DOCS', "test_docs")
        args.docs = [default_docs]
    
    # Expand tilde in paths
    args.docs = [os.path.expanduser(path) for path in args.docs]
    
    logger.debug(f"Document paths: {args.docs}")
    logger.debug(f"Model: {args.model}")
    
    # Initialize RAG system
    rag_system = LangChainRAGSystem(
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
            except Exception:
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
