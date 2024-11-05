import os
import shutil 
import re
from helpers_fw import get_embedding_function
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.schema.document import Document
from langchain_chroma import Chroma
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

from dotenv import load_dotenv
load_dotenv()

# Constants
BASE_URL="https://developer.flutterwave.com/"
MAX_DEPTH=4
CHROMA_PATH = "chroma_fw/v1"
CHUNK_SIZE=800
CHUNK_OVERLAP=80


def main(): 
    # Create the datastore
    documents = load_web_pages()
    chunks = split_documents(documents)
    save_to_chroma(chunks)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Find the specific article elements
    article_content = soup.find("article", class_="content__article")
    # Extract text if the specified element exists, otherwise return an empty string
    if article_content:
        text = article_content.get_text()
    else:
        text = ""
     # Clean up newlines and whitespace
    return re.sub(r"\n\n+", "\n\n", text).strip()

# load web documents 
def load_web_pages():
    loader = RecursiveUrlLoader(
        url=BASE_URL,
        max_depth=MAX_DEPTH,
        continue_on_failure=True,
        prevent_outside=True,
        base_url=BASE_URL,
        extractor=bs4_extractor
    )
    print("🕸️ Loading web pages...")
    documents = loader.load()
    print(f"✔️ successfully loaded {len(documents)} documents")
    if len(documents) == 0:
        print("Zero documents loaded")
        exit(0)
    print(documents[0].metadata,"\n-----\n" ,documents[1].metadata)
    return documents


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