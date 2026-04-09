import os
from dotenv import load_dotenv
from langchain_community .document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("C:/Users/KisoreSubburaman/Documents/Kisore/Gen AI Bootcamp/Learn/langchain-python/langchain-course/mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"), dimensions=1024, model="text-embedding-3-small")

    print("ingesting to Pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("Ingestion complete.")
