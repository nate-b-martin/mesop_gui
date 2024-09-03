from embed.base_embeddings import BaseEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv 
load_dotenv()

class LocalEmbeddings(BaseEmbeddings):

    print("Initializing LocalEmbeddings")

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        super().__init__(base_url, model_name)
        self.model_name = model_name
        self.base_url = base_url
        self.embeddings = OllamaEmbeddings(model=self.model_name, base_url=self.base_url, num_thread=6, show_progress=True) 
