#!/usr/bin/env python3

import os
import sys
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

# Set up logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_document_loading():
    """Test that documents can be loaded and processed."""
    logger.info("Testing document loading...")
    
    # Import the RAG system
    sys.path.insert(0, '.')
    from rag_query import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # Test loading from test_docs directory
    documents = processor.load_documents(['test_docs'])
    
    assert len(documents) >= 3, f"Expected at least 3 documents, got {len(documents)}"
    
    # Check that different file types are loaded
    file_types = [doc['type'] for doc in documents]
    assert '.md' in file_types, "Markdown file not loaded"
    assert '.txt' in file_types, "Text file not loaded" 
    assert '.html' in file_types, "HTML file not loaded"
    
    logger.info(f"✓ Successfully loaded {len(documents)} documents")
    return True

def test_document_chunking():
    """Test that documents are properly chunked."""
    logger.info("Testing document chunking...")
    
    sys.path.insert(0, '.')
    from rag_query import DocumentProcessor
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    documents = processor.load_documents(['test_docs'])
    chunks = processor.chunk_documents(documents)
    
    assert len(chunks) > len(documents), "Should have more chunks than documents"
    
    # Verify chunks have required fields
    for chunk in chunks[:3]:  # Check first 3 chunks
        assert 'content' in chunk, "Chunk missing content"
        assert 'source' in chunk, "Chunk missing source"
        assert 'chunk_id' in chunk, "Chunk missing chunk_id"
        assert len(chunk['content']) <= 600, "Chunk too large"  # Some buffer for sentence breaks
    
    logger.info(f"✓ Successfully created {len(chunks)} chunks from {len(documents)} documents")
    return True

def test_vector_store():
    """Test vector store operations."""
    logger.info("Testing vector store...")
    
    sys.path.insert(0, '.')
    from rag_query import DocumentProcessor, VectorStore
    
    # Use temporary directory for test database
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = VectorStore(persist_directory=temp_dir)
        processor = DocumentProcessor(chunk_size=500)
        
        documents = processor.load_documents(['test_docs'])
        chunks = processor.chunk_documents(documents)
        
        # Add documents to vector store
        vector_store.add_documents(chunks)
        
        # Test search
        results = vector_store.search("Python programming", n_results=3)
        
        assert len(results) > 0, "No search results returned"
        assert len(results) <= 3, "Too many results returned"
        
        # Check that results contain expected fields
        for result in results:
            assert 'content' in result, "Result missing content"
            assert 'source' in result, "Result missing source"
            assert len(result['content']) > 0, "Empty content in result"
        
        logger.info(f"✓ Vector store search returned {len(results)} relevant results")
        return True

def test_cli_interface():
    """Test the command-line interface with a mock model."""
    logger.info("Testing CLI interface...")
    
    # Create a minimal mock for testing without actual API calls
    test_script = """
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import sys
sys.path.insert(0, '.')
from rag_query import DocumentProcessor, VectorStore
import tempfile

# Test basic indexing workflow
processor = DocumentProcessor()
documents = processor.load_documents(['test_docs'])
chunks = processor.chunk_documents(documents)

with tempfile.TemporaryDirectory() as temp_dir:
    vector_store = VectorStore(persist_directory=temp_dir)
    vector_store.add_documents(chunks)
    results = vector_store.search("machine learning types", n_results=2)
    
    print(f"Found {len(results)} results for 'machine learning types'")
    if results:
        print(f"Top result from: {results[0]['source']}")
        print(f"Content preview: {results[0]['content'][:100]}...")
    
print("CLI workflow test completed successfully!")
"""
    
    # Write and run the test script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        test_file = f.name
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✓ CLI workflow test passed")
            logger.debug(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"✗ CLI workflow test failed: {result.stderr}")
            return False
    finally:
        os.unlink(test_file)

def check_dependencies():
    """Check that required dependencies are available."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'chromadb', 'sentence_transformers', 'markdown', 
        'bs4', 'PyPDF2', 'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"✗ Missing packages: {', '.join(missing)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    else:
        logger.info("✓ All required packages are installed")
        return True

def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test basic RAG functionality")
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="Enable verbose logging")
    parser.add_argument('-q', '--quiet', action='store_true', 
                       help="Quiet mode (errors only)")
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    print("=" * 50)
    print("RAG System Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Document Loading", test_document_loading),
        ("Document Chunking", test_document_chunking),
        ("Vector Store", test_vector_store),
        ("CLI Interface", test_cli_interface),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Basic functionality is working.")
        print("\nNext steps:")
        print("1. Set up your .env file with API keys")
        print("2. Try a real query: python rag_query.py -d test_docs -p 'What are Python best practices?'")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())