from langchain_ollama.chat_models import ChatOllama

class LocalModel():
    print("Creating Local Model")
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):        
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = 0.3
        self.numctx = 8096
        self.llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=self.temperature, num_ctx=self.numctx)

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, model_name: str):
        self.model_name = model_name
        self._update_llm()

    def get_base_url(self):
        return self.base_url

    def set_base_url(self, base_url: str):
        self.base_url = base_url
        self._update_llm()

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temperature: float):
        self.temperature = temperature
        self._update_llm()

    def get_numctx(self):
        return self.numctx

    def set_numctx(self, numctx: int):
        self.numctx = numctx
        self._update_llm()

    def get_llm(self):
        return self.llm

    def _update_llm(self):
        self._llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=self.temperature, num_ctx=self.numctx)