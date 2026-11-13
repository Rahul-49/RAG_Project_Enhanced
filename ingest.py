import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.getcwd()
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")

# 1. Load documents
documents = []

for path in glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt")):
    documents.extend(TextLoader(path, encoding="utf-8").load())

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 4. Create vector database
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIR
)

print("Ingestion complete!")