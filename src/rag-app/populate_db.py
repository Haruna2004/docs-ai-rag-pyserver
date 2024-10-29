import argparse
import os
import shutil 
from helpers import get_embedding_function, CHROMA_PATH, DATA_PATH
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.schema.document import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
# load enviroment vairables
load_dotenv()
CHUNK_SIZE=800
CHUNK_OVERLAP=80


def main():    
    # Create (or update) the datastore
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)


# load the document from directory
def load_documents():
    document_loader = DirectoryLoader(DATA_PATH,show_progress=True, 
    use_multithreading=True, silent_errors=True)
    print("Loading Documents...")
    return document_loader.load()

# Split each file into smaller chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        is_separator_regex=False, 
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into len {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first. 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    print("Saving to Chroma")
    db = Chroma(
        embedding_function=get_embedding_function(),persist_directory=CHROMA_PATH
    )
    db.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()