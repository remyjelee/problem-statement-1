# Problem 1: Minimal RAG Solution

## 1. High-Level Architecture
```mermaid
graph TD
    User[User Input] --> App[Python RAG App]
    
    subgraph "Local Inference (Ollama)"
        Embed[Embeddings: all-minilm]
        Gen[LLM: phi3]
    end
    
    subgraph "Storage (ChromaDB)"
        VectorDB[(Vector Database)]
    end

    App -- "1. Vectorize Query" --> Embed
    Embed -- "Vector" --> App
    App -- "2. Similarity Search" --> VectorDB
    VectorDB -- "Top 5 Chunks" --> App
    App -- "3. Prompt + Context" --> Gen
    Gen -- "Answer" --> App
    App --> User
