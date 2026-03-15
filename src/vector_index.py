import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()


def get_embeddings() -> Embeddings:
    return OllamaEmbeddings(model=os.environ["EMBEDDINGS_NAME"])


class VectorIndex:
    @staticmethod
    def from_file(path: str, index_dir: str, embeddings: Embeddings = get_embeddings()) -> FAISS:
        docs: list[Document] = load_pdf(path)
        chunks: list[Document] = split_docs(docs)
        result: FAISS = FAISS.from_documents(chunks, embeddings)
        result.save_local(str(index_dir), index_name=os.environ["INDEX_NAME"])
        return result

    @staticmethod
    def load(index_dir: str, embeddings: Embeddings = get_embeddings()) -> FAISS:
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            index_name=os.environ["INDEX_NAME"],
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def from_args(args: argparse.Namespace, embeddings: Embeddings = get_embeddings()) -> FAISS:
        if args.from_input is not None:
            return VectorIndex.from_file(str(args.from_input), str(args.index_dir), embeddings)
        else:
            return VectorIndex.load(str(args.index_dir), embeddings)


def load_pdf(path: str) -> list[Document]:
    return PyPDFLoader(path).load()


def split_docs(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
