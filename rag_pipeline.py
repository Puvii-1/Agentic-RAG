import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
    
load_dotenv()

CHROMA_DIR    = "./chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 4


def load_and_split(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks: list, collection_name: str = "my_docs") -> Chroma:
    import gc
    gc.collect()

    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    # Use a unique collection name each time instead of deleting
    import time
    unique_collection = f"docs_{int(time.time())}"

    vectordb = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = CHROMA_DIR,
        collection_name   = unique_collection,
    )
    return vectordb, unique_collection


def load_vectorstore(collection_name: str = "my_docs") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embeddings,
        collection_name    = collection_name,
    )


def build_qa_chain(vectordb: Chroma):
    llm = ChatGoogleGenerativeAI(
    model          = "gemini-2.5-flash",
    temperature    = 0,
    google_api_key = os.getenv("GOOGLE_API_KEY"),
)
    
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the provided document."

Context:
{context}

Question: {question}

Answer (be concise, cite page numbers if possible):""",
        input_variables=["context", "question"],
    )

    retriever = vectordb.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": TOP_K},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(chain_and_retriever, question: str) -> dict:
    chain, retriever = chain_and_retriever
    answer  = chain.invoke(question)
    docs    = retriever.invoke(question)

    sources = []
    for doc in docs:
        page    = doc.metadata.get("page", 0)
        snippet = doc.page_content[:150].replace("\n", " ")
        sources.append({"page": page + 1, "snippet": snippet})

    return {"answer": answer, "sources": sources}