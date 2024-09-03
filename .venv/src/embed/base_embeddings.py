class BaseEmbeddings:
    def __init__(self, base_url, model_name):
        self.model_name = model_name
        self.base_url = base_url
        self.embeddings = None

    def initialize(self):
        raise NotImplementedError

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.update_embeddings()

    def get_base_url(self):
        return self.base_url

    def set_base_url(self, base_url):
        self.base_url = base_url
        self.update_embeddings()

    def get_embeddings(self):
        return self.embeddings

    def update_embeddings(self):
        raise NotImplementedError
