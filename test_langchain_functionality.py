#!/usr/bin/env python

#!/usr/bin/env python3
"""
Test script for LangChain RAG functionality.

This script tests the LangChain-based RAG system to ensure all components
work correctly before and after the integration.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Test imports
def test_imports():
    """Test that all required packages can be imported."""
    print("Testing LangChain imports...")
    
    try:
        from langchain.schema import Document
        print("✅ langchain.schema imported")
        
        from langchain_chroma import Chroma
        print("✅ langchain_chroma imported")
        
        from langchain_community.document_loaders import TextLoader
        print("✅ langchain_community.document_loaders imported")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ langchain_community.embeddings imported")
        
        from langchain_ollama import OllamaLLM
        print("✅ langchain_ollama imported")
        
        from langchain_openai import ChatOpenAI
        print("✅ langchain_openai imported")
        
        from langchain_anthropic import ChatAnthropic
        print("✅ langchain_anthropic imported")
        
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✅ langchain_text_splitters imported")
        
        from langchain.chains import RetrievalQA
        print("✅ langchain.chains imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_langchain_components():
    """Test LangChain components individually."""
    print("\nTesting LangChain components...")
    
    try:
        # Import our classes
        from rag_query_langchain import (
            LangChainDocumentProcessor,
            LangChainVectorStore,
            LangChainRAGSystem
        )
        
        # Test document processor
        processor = LangChainDocumentProcessor(chunk_size=500, chunk_overlap=100)
        print("✅ LangChainDocumentProcessor initialized")
        
        # Test vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = LangChainVectorStore(
                collection_name="test_collection",
                persist_directory=temp_dir
            )
            print("✅ LangChainVectorStore initialized")
        
        # Test RAG system initialization
        rag_system = LangChainRAGSystem(
            doc_paths=["test_docs"],
            chunk_size=500,
            chunk_overlap=100
        )
        print("✅ LangChainRAGSystem initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


def test_document_loading():
    """Test document loading with LangChain."""
    print("\nTesting document loading...")
    
    try:
        from rag_query_langchain import LangChainDocumentProcessor
        
        processor = LangChainDocumentProcessor()
        documents = processor.load_documents(["test_docs"])
        
        if documents:
            print(f"✅ Loaded {len(documents)} documents")
            
            # Test chunking
            chunks = processor.chunk_documents(documents)
            print(f"✅ Created {len(chunks)} chunks")
            
            return True
        else:
            print("❌ No documents loaded")
            return False
            
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        return False


def test_vector_store():
    """Test vector store operations."""
    print("\nTesting vector store...")
    
    try:
        from rag_query_langchain import LangChainDocumentProcessor, LangChainVectorStore
        from langchain.schema import Document
        
        # Create test documents
        test_docs = [
            Document(page_content="This is about machine learning.", metadata={"source": "test1.txt"}),
            Document(page_content="Python is a programming language.", metadata={"source": "test2.txt"}),
            Document(page_content="Deep learning uses neural networks.", metadata={"source": "test3.txt"})
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = LangChainVectorStore(
                collection_name="test_collection",
                persist_directory=temp_dir
            )
            
            # Add documents
            vector_store.add_documents(test_docs)
            print("✅ Documents added to vector store")
            
            # Test search
            results = vector_store.search("machine learning", n_results=2)
            if results:
                print(f"✅ Search returned {len(results)} results")
                return True
            else:
                print("❌ Search returned no results")
                return False
                
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_llm_connections():
    """Test LLM connections for all providers."""
    print("\nTesting LLM connections...")
    
    success_count = 0
    total_tests = 0
    
    # Test Ollama
    try:
        from langchain_ollama import OllamaLLM
        
        llm = OllamaLLM(
            model="qwen2.5vl:7b",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        
        response = llm.invoke("Hello")
        if response:
            print("✅ Ollama connection successful")
            success_count += 1
        else:
            print("❌ Ollama: No response")
        total_tests += 1
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        total_tests += 1
    
    # Test OpenAI (only if API key is available)
    import os
    if os.getenv('OPENAI_API_KEY'):
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=50
            )
            
            response = llm.invoke("Hello")
            if response:
                print("✅ OpenAI connection successful")
                success_count += 1
            else:
                print("❌ OpenAI: No response")
            total_tests += 1
                
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
            total_tests += 1
    else:
        print("⚠️  OpenAI: API key not found, skipping test")
    
    # Test Anthropic (only if API key is available)
    if os.getenv('ANTHROPIC_API_KEY'):
        try:
            from langchain_anthropic import ChatAnthropic
            
            llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=50
            )
            
            response = llm.invoke("Hello")
            if response:
                print("✅ Anthropic connection successful")
                success_count += 1
            else:
                print("❌ Anthropic: No response")
            total_tests += 1
                
        except Exception as e:
            print(f"❌ Anthropic connection failed: {e}")
            total_tests += 1
    else:
        print("⚠️  Anthropic: API key not found, skipping test")
    
    print(f"LLM Tests: {success_count}/{total_tests} providers working")
    return success_count > 0


def test_full_rag_workflow():
    """Test the complete RAG workflow."""
    print("\nTesting full RAG workflow...")
    
    try:
        from rag_query_langchain import LangChainRAGSystem
        
        # Initialize RAG system
        rag_system = LangChainRAGSystem(
            doc_paths=["test_docs"],
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Index documents
        rag_system.index_documents(reindex=True)
        print("✅ Documents indexed successfully")
        
        # Test query
        response = rag_system.query(
            prompt="What are Python best practices?",
            model="qwen2.5vl:7b",
            n_results=3
        )
        
        if response:
            print("✅ RAG query successful")
            print(f"Response preview: {response[:200]}...")
            return True
        else:
            print("❌ RAG query failed")
            return False
            
    except Exception as e:
        print(f"❌ Full RAG workflow failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("LangChain RAG System Test Suite")
    print("="*60)
    
    # Suppress some logging for cleaner output
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    tests = [
        ("Imports", test_imports),
        ("Components", test_langchain_components),
        ("Document Loading", test_document_loading),
        ("Vector Store", test_vector_store),
        ("LLM Connections", test_llm_connections),
        ("Full RAG Workflow", test_full_rag_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! LangChain integration is working.")
        print("\nNext steps:")
        print("1. Try a real query: python rag_query_langchain.py -d test_docs -p 'What are Python best practices?'")
        print("2. Compare with original: python rag_query.py -d test_docs -p 'What are Python best practices?' -m ollama")
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()