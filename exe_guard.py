import os
import time
import shutil
import re
from datetime import datetime
from typing import List, Dict
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

# Import the system prompt from system_prompt.py
from system_prompt import SYSTEM_PROMPT

load_dotenv()


class ExcelLoader:
    """A simple loader for Excel files (XLSX or XLS)."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        xls = pd.ExcelFile(self.file_path)
        docs = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            text = df.to_string(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": self.file_path, "sheet_name": sheet_name},
                )
            )
        return docs


class BRDRAG:
    def load_documents(self, document_paths: List[str]) -> List[Document]:
        documents = []
        # Process each file and tag its content with a doc_id (1st, 2nd, …)
        for idx, path in enumerate(document_paths, start=1):
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.lower().endswith((".docx", ".doc")):
                loader = Docx2txtLoader(path)
            elif path.lower().endswith((".xlsx", ".xls")):
                loader = ExcelLoader(path)
            else:
                print(f"Unsupported file type: {path}")
                continue

            docs = loader.load()
            # Add the document ID to each Document’s metadata
            for doc in docs:
                doc.metadata["doc_id"] = idx

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            # Split the document into chunks while preserving metadata
            documents.extend(text_splitter.split_documents(docs))
        return documents

    def splitDoc(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128
        )
        splits = text_splitter.split_documents(documents)
        return splits

    def getEmbedding(self):
        modelPath = "mixedbread-ai/mxbai-embed-large-v1"
        device = "cpu"  # or "cuda" if available
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": False}

        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        return embeddings

    def getResponse(self, document_paths: List[str], queries: List[str]) -> Dict[str, str]:
        """
        Accepts an array of document paths and an array of queries.
        If a query ends with a tag like "#1#" or "#2#", it will be executed only on the
        corresponding document (1st document for "#1#", etc.). Returns a dictionary mapping
        each original query (including its tag) to its answer.
        """
        responses = {}
        # Load and process documents (each chunk now has a "doc_id" in its metadata)
        documents = self.load_documents(document_paths)
        splits = self.splitDoc(documents)
        embeddings = self.getEmbedding()

        # Clean up old persist directories older than 30 minutes
        chroma_root = "docs/chroma"
        if os.path.exists(chroma_root):
            now = time.time()
            for folder in os.listdir(chroma_root):
                folder_path = os.path.join(chroma_root, folder)
                if os.path.isdir(folder_path):
                    if now - os.path.getmtime(folder_path) > 30 * 60:
                        shutil.rmtree(folder_path)

        # Setup vector store
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        persist_directory = f"docs/chroma/{current_timestamp}"

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        # Setup a default retriever (without filtering)
        default_retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Initialize Mistral through LangChain
        llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-large-latest")

        # Create a PromptTemplate that includes the system prompt from system_prompt.py
        prompt = PromptTemplate(
            template=f"""{SYSTEM_PROMPT}

Use the following context to answer the question.

Context: {{context}}
Question: {{input}}

Answer:""",
            input_variables=["context", "input"],
        )

        # Create default document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        default_retrieval_chain = create_retrieval_chain(default_retriever, document_chain)

        # Iterate over each query
        for query in queries:
            original_query = query  # preserve the query with its tag for output
            doc_filter = None
            # Look for a tag at the end of the query (e.g., "#1#")
            match = re.search(r"#(\d+)#$", query.strip())
            if match:
                doc_id = int(match.group(1))
                doc_filter = {"doc_id": doc_id}
                # Remove the document tag from the query text
                query = re.sub(r"#(\d+)#$", "", query).strip()

            # Use a filtered retriever if a document tag was provided
            if doc_filter:
                current_retriever = vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4, "filter": doc_filter},
                )
                current_retrieval_chain = create_retrieval_chain(current_retriever, document_chain)
            else:
                current_retriever = default_retriever
                current_retrieval_chain = default_retrieval_chain

            # Optional: Print token counts for debugging
            retrieved_docs = current_retriever.invoke(query)
            retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)
            context_tokens_rag = self.count_tokens(retrieved_text)
            print(f"Retrieved context token count (RAG) for query '{original_query}': {context_tokens_rag}")

            full_text = "\n\n".join([doc.page_content for doc in documents])
            context_tokens_llm = self.count_tokens(full_text)
            print(f"Full document context token count (LLM) for query '{original_query}': {context_tokens_llm}")

            # Execute chain for the current query
            response = current_retrieval_chain.invoke({"input": query})
            responses[original_query] = response["answer"]

        return responses

    def count_tokens(self, text: str) -> int:
        try:
            hf_token = os.getenv("HUGGINGFACE_API_KEY")
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3", token=hf_token
            )
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            return 0
