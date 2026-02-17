import chromadb
import ollama
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
GEN_MODEL = "phi3" 
EMBED_MODEL = "all-minilm"

def build_knowledge_base():
    """
    Phase 1: Ingestion
    """
    print("--- Building Knowledge Base ---")
    client = chromadb.Client()

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings"
    )
    
    collection = client.create_collection(name="rag_demo", embedding_function=ollama_ef)

    documents = [
        "Problem Statement 1 requests the design and implementation of a minimal Retrieval-Augmented Generation (RAG) solution.",
        "The system must combine a retrieval mechanism with a Generative AI model to answer user queries based on a dataset.",
        "The solution is not intended to be production-ready; it should be a prototype with clear diagrams and explanations.",
        "Constraint: The system should use small models that can run locally on a CPU to minimize expenses.",
        "Key Component - Knowledge Base: A system to store and organize data for efficient access.",
        "Key Component - Semantic Layer: Transforms data and queries into semantic representations for comparison.",
        "Key Component - Retrieval System: Identifies relevant elements from the knowledge base using semantic representations.",
        "Key Component - Augmentation: Combines retrieved info with the user query to enrich context.",
        "Key Component - Generation: Uses a generative AI model to produce a grounded response."
    ]
    
    ids = [f"id_{i}" for i in range(len(documents))]

    collection.add(documents=documents, ids=ids)
    print(f"Stored {len(documents)} documents in vector store.\n")
    return collection

def query_system(collection, user_query):
    """
    Phase 2: Retrieval, Augmentation, Generation
    """
    print(f"--- Processing Query: '{user_query}' ---")
    
    # 1. RETRIEVAL
    results = collection.query(
        query_texts=[user_query],
        n_results=5 
    )
    
    retrieved_context = "\n".join(results['documents'][0])
    
    # 2. AUGMENTATION
    # Improved Prompt: Explicitly tells the model to stop after answering.
    prompt = f"""
    You are a technical assistant. Answer the user's question concisely using ONLY the context provided below.
    Do not add conversational filler. Do not make up questions.
    
    Context:
    {retrieved_context}
    
    Question: {user_query}
    """

    # 3. GENERATION
    print("AI Answer: ", end="", flush=True)
    stream = ollama.chat(
        model=GEN_MODEL, 
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )
    
    for chunk in stream:
        print(chunk['message']['content'], end="", flush=True)
    print("\n") 

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    db = build_knowledge_base()
    
    # Run Test Queries
    query_system(db, "What are the hardware constraints?")
    query_system(db, "List the key components.")
    query_system(db, "What is the goal of Problem Statement 1?")