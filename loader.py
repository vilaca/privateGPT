import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS, LOADER_CHANNEL_NAME
import redis
import json 

red = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)

load_dotenv()


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


load_dotenv()


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def main():

    while True:
        sub = red.pubsub()    
        sub.subscribe(LOADER_CHANNEL_NAME)
        print("waiting!!:")
        for message in sub.listen():    
            if message is not None and isinstance(message, dict):    
                #print(message['data'])
                message_data = message['data']
                print(message_data)
                print(type(message_data))
                if isinstance(message_data, str):
                    data=json.loads(message_data)
                    file_path=data['file']
                    folder_path=data['folder']
                    documents = None
                    if file_path:
                        print(file_path)
                        documents = [load_single_document(file_path)]
                    elif folder_path:
                        print(folder_path)
                        documents = load_documents(folder_path)
                    if documents:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        texts = text_splitter.split_documents(documents)
                        doit(texts)
                    print("ok")

    # Load documents and split in chunks
    #print(f"Loading documents from {source_directory}")
    #documents = load_documents(source_directory)

    #print(f"Loaded {len(documents)} documents from {source_directory}")
    #print(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

def doit(texts: str):
    print("starting")
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    #source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Create embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, llama, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()
