# LangChain RAG System Flow Diagram

```mermaid
flowchart TD
    %% Input Stage
    A[User Query] --> B[CLI Arguments Parser]
    C[Document Paths] --> B
    D[Model Selection] --> B
    
    %% Document Processing Stage
    B --> E[LangChainDocumentProcessor]
    E --> F{Document Type?}
    F -->|.md| G[UnstructuredMarkdownLoader]
    F -->|.txt| H[TextLoader]
    F -->|.pdf| I[PyPDFLoader]
    F -->|.html/.htm| J[UnstructuredHTMLLoader]
    F -->|Directory| K[DirectoryLoader]
    
    G --> L[Document Objects]
    H --> L
    I --> L
    J --> L
    K --> L
    
    %% Text Splitting Stage
    L --> M[RecursiveCharacterTextSplitter]
    M --> N[Document Chunks]
    N --> O[chunk_size: 1000<br/>chunk_overlap: 200]
    
    %% Vector Store Stage
    O --> P[LangChainVectorStore]
    P --> Q[HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2]
    Q --> R[Vector Embeddings]
    R --> S[ChromaDB<br/>Persistent Storage]
    
    %% Query Processing Stage
    A --> T[Semantic Search]
    T --> S
    S --> U[Retrieve Top-K<br/>Relevant Chunks]
    U --> V[Context Assembly]
    
    %% LLM Integration Stage
    V --> W[RetrievalQA Chain]
    W --> X[Custom Prompt Template]
    X --> Y[OllamaLLM<br/>Local Model]
    Y --> Z{Model Type?}
    Z -->|qwen2.5vl:7b| AA[Qwen Model]
    Z -->|llama3.2:latest| BB[Llama Model]
    Z -->|deepseek-r1:1.5b| CC[DeepSeek Model]
    
    %% Output Stage
    AA --> DD[Generated Response]
    BB --> DD
    CC --> DD
    DD --> EE[Final Answer]
    
    %% System Components
    subgraph "Document Processing"
        E
        F
        G
        H
        I
        J
        K
        L
    end
    
    subgraph "Text Chunking"
        M
        N
        O
    end
    
    subgraph "Vector Storage"
        P
        Q
        R
        S
    end
    
    subgraph "Retrieval & Generation"
        T
        U
        V
        W
        X
        Y
        Z
        AA
        BB
        CC
    end
    
    %% Styling
    classDef inputStyle fill:#e1f5fe
    classDef processStyle fill:#f3e5f5
    classDef vectorStyle fill:#e8f5e8
    classDef llmStyle fill:#fff3e0
    classDef outputStyle fill:#fce4ec
    
    class A,C,D inputStyle
    class E,F,G,H,I,J,K,L,M,N,O processStyle
    class P,Q,R,S,T,U,V vectorStyle
    class W,X,Y,Z,AA,BB,CC llmStyle
    class DD,EE outputStyle
```

## Key Components:

1. **Document Processing**: LangChain loaders handle multiple file formats
2. **Text Splitting**: Recursive character splitter with configurable chunk size/overlap
3. **Vector Storage**: HuggingFace embeddings with ChromaDB persistence
4. **Retrieval**: Semantic search for relevant context
5. **Generation**: Ollama LLM with RetrievalQA chain and custom prompts

## Data Flow:

- Documents → Chunks → Vectors → Storage
- Query → Search → Context → LLM → Response
- All components use LangChain abstractions for modularity