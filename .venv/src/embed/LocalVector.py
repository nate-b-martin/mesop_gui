from langchain_community.vectorstores import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from prompts.prompts import get_contextualize_q_prompt

class LocalVector():
    print("Initializing Local Vector")
    def __init__(self, documents, embeddings):
        self.docs = documents
        self.vector_store = Chroma.from_documents(
            documents=self.docs,
            embedding=embeddings,
            persist_directory="db"
        )
        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        return self.retriever
    
    def history_aware_retriever(self, llm):
        return create_history_aware_retriever(
            llm=llm, retriever=self.retriever, prompt=get_contextualize_q_prompt()
        )