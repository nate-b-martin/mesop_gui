import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader, PyPDFLoader

class LoadDocs():
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.docs = []
        self.split_text = []

    def load_documents(self):
        """
        Load documents from the specified data path.
        Supported file types:
        - PDF (.pdf)
        - Text (.txt)
        - JSON (.json)
        - CSV (.csv)
        """

        loaders = {
            "pdf": DirectoryLoader(
                self.data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
            ),
            "txt": DirectoryLoader(
                self.data_path, 
                loader_cls=TextLoader, 
                glob="**/*.txt"
            ),
            "json": DirectoryLoader(
                self.data_path, 
                loader_cls=TextLoader, 
                glob="**/*.json"
            ),
            "csv": DirectoryLoader(self.data_path, loader_cls=CSVLoader, glob="**/*.csv"),
        }

        for loader in loaders.values():
            loaded_docs = loader.load()
            self.docs.extend(loaded_docs)

    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.split_text = text_splitter.split_documents(self.docs)

    def print_chunks(self):
        for chunk in self.split_text:
            print(chunk.page_content)
            print("-"*100)

    def process_documents(self):
        print("Processing documents...")
        self.load_documents()
        self.split_documents()
        # self.print_chunks()
        return self.split_text