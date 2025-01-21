import os
import shutil 
from helpers_sq import get_embedding_function
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.schema.document import Document
from langchain_chroma import Chroma
from bs4 import BeautifulSoup 

from dotenv import load_dotenv
load_dotenv()

# Constants
DATA_PATH = "data/squad"
CHROMA_PATH = "chroma_sq/v1"
CHUNK_SIZE=800
CHUNK_OVERLAP=80


def main():    
    # Create (or update) the datastore
    documents = load_documents()
    cleaned_documents = clean_documents(documents)
    chunks = split_documents(cleaned_documents)
    save_to_chroma(chunks)

# load the document from directory
def load_documents():
    document_loader = DirectoryLoader(DATA_PATH,show_progress=True, 
    glob="**/*.mdx",
    use_multithreading=True, silent_errors=True)
    print("Loading Documents...")
    return document_loader.load()

# Remove HTML tags
def clean_document(document: Document):
    soup = BeautifulSoup(document.page_content, 'html.parser')
    return soup.get_text(strip=True)

# Extract plain text only
def clean_documents(documents: list[Document]):
    cleaned_documents = []
    for doc in documents:
        cleaned_text = clean_document(doc)
        cleaned_documents.append(Document(page_content=cleaned_text, metadata=doc.metadata))
    return cleaned_documents

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

# Add chunks with metadata to vector store
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