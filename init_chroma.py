from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma


def get_embedding_function():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return embeddings


CHROMA_PATH="chroma/v1"
CHROMA_DB_INSTANCE = None # Reference to singleton instance of ChromaDB
def get_chroma_db():
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:

        # Prepare the Db
        CHROMA_DB_INSTANCE = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        print(f"✅ Init ChromaDB Instance from {CHROMA_PATH}")
    
    return CHROMA_DB_INSTANCE

if __name__ == "__main__":
    get_chroma_db()
