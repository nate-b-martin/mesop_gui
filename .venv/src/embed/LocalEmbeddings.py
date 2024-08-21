from langchain_community.embeddings import OllamaEmbeddings

class LocalEmbeddings():
    print("Initializing LocalEmbeddings")
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.embeddings = OllamaEmbeddings(model=self.model_name, base_url = self.base_url)

    def get_model_name(self) -> str:
        return self.model_name

    def set_model_name(self, model_name: str):
        self.model_name = model_name
        self.update_embeddings()

    def get_base_url(self) -> str:
        return self.base_url

    def set_base_url(self, base_url: str):
        self.base_url = base_url
        self.update_embeddings()


    def get_embeddings(self):
        return self.embeddings

    def update_embeddings(self):
        self.embeddings = OllamaEmbeddings(model=self.model_name, base_url=self.base_url)