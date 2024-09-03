from langchain_community.chat_message_histories import ChatMessageHistory
class BaseChat():
    def __init__(self, llm, doc_path):
        self.doc_path = doc_path
        self.chat_history = ChatMessageHistory()
        self.llm = llm 
        self.embeddings = None
        self.docs = None
        self.vector = None
        self.retriever = None
        self.history_aware_retriever = None

    def initialize(self):
        raise NotImplementedError

    def set_local_env(self):
        raise NotImplementedError

    def run_chain(self, input, history):
        raise NotImplementedError
