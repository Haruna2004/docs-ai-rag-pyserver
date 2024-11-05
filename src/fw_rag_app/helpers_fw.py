import os
from langchain.schema.document import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma


def get_embedding_function():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return embeddings


FW_CHROMA_PATH="chroma_fw/v1"
FW_CHROMA_DB_INSTANCE = None # Reference to singleton instance of ChromaDB
def get_chroma_db():
    global FW_CHROMA_DB_INSTANCE
    if not FW_CHROMA_DB_INSTANCE:

        # Prepare the Db
        FW_CHROMA_DB_INSTANCE = Chroma(
            persist_directory=FW_CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        print(f"✅ Init ChromaDB Instance from {FW_CHROMA_PATH}")
    
    return FW_CHROMA_DB_INSTANCE


def save_to_folder(documents: list[Document], output_folder: str, file_ext:str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Write each document to a sperate file
    for i,doc in enumerate(documents):
        filename = os.path.join(output_folder, f"doc_{i+1}.{file_ext}")
        with open(filename,'w', encoding='utf-8') as f:
            f.write(doc.page_content)
    print(f"Saved {len(documents)} documents to {output_folder}.")

# System prompt for llm - Change this
PROMPT_TEMPLATE = """ 
You are Flutterwave Docs asssistant. Users will ask question on using flutterwave. You can add additional details, code that is useful.
Answer the question using  the following context:
{context}
---
Answer the question based on the above context: {question}
"""
GROQ_MODEL_ID="mixtral-8x7b-32768"
MODEL_TEMP=0.5

