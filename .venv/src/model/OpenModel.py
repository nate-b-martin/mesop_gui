from langchain_openai.chat_models.base import ChatOpenAI
class OpenAIModel():
    """
    Class for initializing OpenAI language model
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, api_key: str = None, base_url: str = None, max_tokens: int = 4096):
        """
        Initialize OpenAI model

        Args:
            model (str, optional): Model to use. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): Controls randomness. Defaults to 0.7.
            api_key (str, optional): OpenAI API key. Defaults to None.
            base_url (str, optional): Base URL for OpenAI API. Defaults to None.
            max_tokens (int, optional): Max number of tokens to generate. Defaults to 4096.
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
    
    def get_model(self):
        return self.model

    def set_model(self, model: str):
        self.model = model
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
        self._llm = ChatOpenAI(model=self.model, base_url=self.base_url, temperature=self.temperature, num_ctx=self.numctx)
