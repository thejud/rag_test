# RAG Query System

A simple RAG (Retrieval-Augmented Generation) system that augments LLM queries with content from local documents.

## Features

- Support for multiple document types: Markdown (.md), PDF (.pdf), plain text (.txt), HTML (.html/.htm)
- Multiple LLM backends: OpenAI, Claude, local Ollama models
- Chroma vector database for document storage and retrieval
- Intelligent text chunking with overlap
- Command-line interface with short flags

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

3. **Install Ollama (optional, for local models):**
   ```bash
   # On macOS
   brew install ollama
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (e.g., Qwen2)
   ollama pull qwen2:7b
   ```

4. **Make the script executable:**
   ```bash
   chmod +x rag_query.py
   ```

## Usage

### Basic Usage

```bash
# Query with default settings
python rag_query.py -p "What are the main themes in my notes?"

# Specify document directory and model
python rag_query.py -d ./my_docs -m claude -p "Summarize the key points about machine learning"

# Use multiple document sources
python rag_query.py -d ./notes -d ./books -d ./articles -p "Compare different AI approaches"
```

### Command Line Options

- `-p, --prompt`: The query prompt (required)
- `-d, --docs`: Document paths (files or directories, can be used multiple times)
- `-m, --model`: LLM model to use (default: openai)
- `-n, --num-results`: Number of relevant chunks to include (default: 5)
- `-r, --reindex`: Force reindexing of all documents
- `-c, --chunk-size`: Text chunk size in characters (default: 1000)
- `-o, --chunk-overlap`: Overlap between chunks (default: 200)

### Supported Models

**OpenAI:**
- `openai` (uses gpt-3.5-turbo)
- `gpt-3.5-turbo`
- `gpt-4`

**Claude:**
- `claude` (uses claude-3-haiku-20240307)
- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`

**Ollama (local):**
- `ollama` (uses qwen2:7b)
- `qwen2:7b`
- Any other Ollama model name

### Examples

```bash
# Quick query with default settings
python rag_query.py -p "What did I write about project management?"

# Use Claude with specific document folder
python rag_query.py -d ~/Documents/notes -m claude -p "Find information about Python best practices"

# Force reindex and use more context
python rag_query.py -r -n 10 -p "What are the main ideas across all my documents?"

# Use local model with custom chunk size
python rag_query.py -m qwen2:7b -c 500 -o 100 -p "Explain the concept of RAG"
```

## Configuration

### Environment Variables

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
RAG_DEFAULT_DOCS=./docs
```

### Default Document Directory

If you don't specify `-d/--docs`, the system will use:
1. The `RAG_DEFAULT_DOCS` environment variable
2. `./docs` as fallback

## How It Works

1. **Document Loading**: Processes markdown, PDF, text, and HTML files
2. **Text Chunking**: Splits documents into overlapping chunks for better context
3. **Vector Indexing**: Uses Chroma DB with sentence-transformers for embeddings
4. **Similarity Search**: Finds relevant chunks based on your query
5. **LLM Augmentation**: Sends context + query to your chosen LLM

## Troubleshooting

### Common Issues

**"No documents found in index"**
- Check your document paths with `-d`
- Use `-r` to force reindexing

**"Error connecting to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Verify the model is installed: `ollama list`

**API Key errors**
- Check your `.env` file configuration
- Ensure API keys are valid and have sufficient credits

**Memory issues with large document sets**
- Reduce chunk size with `-c`
- Process documents in smaller batches

### Performance Tips

- Use smaller chunk sizes (-c 500) for more precise matching
- Increase overlap (-o 300) for better context continuity
- Adjust number of results (-n) based on your needs
- Consider using local models for privacy-sensitive documents

## Data Storage

- Vector database: `./chroma_db/` directory
- Embeddings are persisted between runs
- Use `-r` flag to rebuild the index when documents change