from langchain_ollama.chat_models import ChatOllama
from model.base_model import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()

class LocalModel(BaseModel):
    print("Creating Local Model")
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):        
        super().__init__(model_name, base_url)
        self.temperature = 0
        self.numctx = 8096
        self.llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=self.temperature, num_ctx=self.numctx, keep_alive="10m")

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temperature: float):
        self.temperature = temperature
        self.update_llm()

    def get_numctx(self):
        return self.numctx

    def set_numctx(self, numctx: int):
        self.numctx = numctx

    def update_llm(self):
        self.llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=self.temperature, num_ctx=self.numctx)