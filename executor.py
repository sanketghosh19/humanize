import os
import time
import shutil
from datetime import datetime
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer

# Import PromptTemplate from langchain and the system prompt from system_prompt.py
from langchain.prompts import PromptTemplate
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
    def load_documents(self, document_paths: List[str]) -> List[Dict]:
        documents = []
        for path in document_paths:
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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            documents.extend(text_splitter.split_documents(docs))
        return documents

    def splitDoc(self, documents):
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
        Returns a dictionary mapping each query to its answer with an annotation of which document (by order)
        contributed most to the answer.
        """
        responses = {}
        
        # Build a mapping from file path to document number (for labeling)
        file_index_map = {path: idx + 1 for idx, path in enumerate(document_paths)}
        
        # Load and process documents
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

        # Setup vector store with a unique persist directory
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        persist_directory = f"docs/chroma/{current_timestamp}"

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        # Setup retriever
        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Initialize Mistral through LangChain
        llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-large-latest")

        # Create prompt with a system prompt included from system_prompt.py
        prompt = PromptTemplate(
            template=f"""{SYSTEM_PROMPT}

Use the following context to answer the question.

Context: {{context}}
Question: {{input}}

Answer:""",
            input_variables=["context", "input"],
        )

        # Create document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Iterate over each query
        for query in queries:
            # Retrieve top-k documents for the query (used here for analysis)
            retrieved_docs = retriever.invoke(query)
            if retrieved_docs:
                # Use the first (most relevant) document to decide the source
                primary_doc = retrieved_docs[0]
                doc_source = primary_doc.metadata.get("source", "Unknown")
                doc_number = file_index_map.get(doc_source, "Unknown")
                doc_annotation = f"Document #{doc_number} ({doc_source})"
            else:
                doc_annotation = "No relevant document found"

            # (Optional) Print token counts for debugging
            retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)
            context_tokens_rag = self.count_tokens(retrieved_text)
            print(f"Retrieved context token count while using RAG for query '{query}': {context_tokens_rag}")

            full_text = "\n\n".join([doc.page_content for doc in documents])
            context_tokens_llm = self.count_tokens(full_text)
            print(f"Retrieved context token count while passing doc directly to LLM for query '{query}': {context_tokens_llm}")

            # Execute chain for the current query
            response = retrieval_chain.invoke({"input": query})
            
            # Append the document annotation to the answer
            final_answer = f"{doc_annotation}\nAnswer: {response['answer']}"
            responses[query] = final_answer

        return responses

    # Optional method to count tokens. To be removed if not needed.
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
