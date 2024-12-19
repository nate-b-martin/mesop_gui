import pandas as pd
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, UnstructuredPowerPointLoader, DataFrameLoader

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
        - PowerPoint (.pptx)
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
            # "csv": DirectoryLoader(self.data_path, loader_cls=csv_loader_factory, glob="*.csv"),
            "pptx": DirectoryLoader(self.data_path, loader_cls=UnstructuredPowerPointLoader, glob="**/*.pptx"),
            "csv": DirectoryLoader(
                self.data_path,
                glob="**/*.csv",
                loader_cls=lambda file_path: DataFrameLoader(
                    pd.DataFrame(pd.read_csv(file_path).apply(lambda row: ' '.join(row.astype(str)), axis=1)),
                    page_content_column=0
                )
            ),
        }

        for loader in loaders.values():
            loaded_docs = loader.load()
            self.docs.extend(loaded_docs)

    def load_modified_docs(self, file_paths=None):
        """
        Load and split documents from the provided file paths.
        If file_paths is None, use the default data path.
        """
        if file_paths is None:
            self.load_documents()
        else:
            self.docs = []
            for file_path in file_paths:
                file_type = file_path.split('.')[-1].lower()
                if file_type == 'pdf':
                    loader = PyPDFLoader(file_path)
                elif file_type in ['txt', 'json']:
                    loader = TextLoader(file_path)
                elif file_type == 'csv':
                    loader = DataFrameLoader(pd.read_csv(file_path), page_content_column="text")
                elif file_type == 'pptx':
                    loader = UnstructuredPowerPointLoader(file_path)
                elif file_type == 'xlsx':
                    loader= DataFrameLoader(pd.read_excel(file_path), page_content_column='text'),
                else:
                    print(f"Unsupported file type: {file_type}")
                    continue
                
                loaded_docs = loader.load()
                self.docs.extend(loaded_docs)
        
        self.split_documents()
        return self.split_text

    def split_documents(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len
        )
        self.split_text = [chunk for doc in self.docs for chunk in splitter.split_documents([doc])]

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