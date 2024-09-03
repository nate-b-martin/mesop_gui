from langchain_openai.embeddings import OpenAIEmbeddings
class AIEmbeddings():
    """Class for initializing OpenAI Embeddings"""
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None, base_url: str = None):
        """
        Initialize OpenAI Embeddings

        Args:
            model (str, optional): Model to use. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): Controls randomness. Defaults to 0.7.
            api_key (str, optional): OpenAI API key. Defaults to None.
            base_url (str, optional): Base URL for OpenAI API. Defaults to None.
            max_tokens (int, optional): Max number of tokens to generate. Defaults to 4096.
        """
        self.embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model
        self.base_url = base_url
    def get_model_name(self) -> str:
        return self.model

    def set_model_name(self, model: str):
        self.model = model
        self.update_embeddings()

    def get_base_url(self) -> str:
        return self.base_url

    def set_base_url(self, base_url: str):
        self.base_url = base_url
        self.update_embeddings()


    def get_embeddings(self):
        return self.embeddings

    def update_embeddings(self):
        self.embeddings = OpenAIEmbeddings(model=self.model, base_url=self.base_url)
