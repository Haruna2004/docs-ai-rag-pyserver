from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def get_embedding_function():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return embeddings


DATA_PATH = "data"
CHROMA_PATH = "chroma"