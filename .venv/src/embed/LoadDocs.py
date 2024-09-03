import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader, PyPDFLoader, UnstructuredPowerPointLoader, DataFrameLoader

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
            "csv": DirectoryLoader(self.data_path, loader_cls=CSVLoader, glob="**/*.csv"),
            "pptx": DirectoryLoader(self.data_path, loader_cls=UnstructuredPowerPointLoader, glob="**/*.pptx"),
        }

        #  handle csv files
        # csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        # for csv_file in csv_files:
        #     file_path = os.path.join(self.data_path, csv_file)
        #     df = pd.read_csv(file_path)
            
        #     # If 'text' column doesn't exist, create it by concatenating all columns
        #     if 'text' not in df.columns:
        #         df['text'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
            
        #     csv_loader = DataFrameLoader(df, page_content_column='text')
        #     loaded_docs = csv_loader.load()
        #     self.docs.extend(loaded_docs)

        for loader in loaders.values():
            loaded_docs = loader.load()
            self.docs.extend(loaded_docs)


    def split_documents(self):
        splitters = {
                "default": RecursiveCharacterTextSplitter(
                    chunk_size=1500, chunk_overlap=300, add_start_index=True
                ),
                "pdf": RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=200, add_start_index=True
                ),
                "csv": RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=100, add_start_index=True
                )
            }
        
        split_docs = []
        for doc in self.docs:
            file_type = doc.metadata.get('source', '').split('.')[-1].lower()
            splitter = splitters.get(file_type, splitters['default'])
            split_docs.extend(splitter.split_documents([doc]))
        
        self.split_text = split_docs

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