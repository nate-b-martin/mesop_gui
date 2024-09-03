from langchain_community.chat_message_histories import ChatMessageHistory
from pprint import pprint
from model.OpenModel import OpenAIModel
from embed.AIEmbeddings import AIEmbeddings
from embed.LoadDocs import LoadDocs 
from embed.LocalVector import LocalVector 
import custom_chains.Chains as custom_chains
from langchain_core.runnables.utils import AddableDict
from prompts.prompts import get_helpful_assistant_prompt

class OpenChat():
    def __init__(self, doc_path="/Users/nathan.martin/Documents/projects/mesop/mesop_gui/.venv/src/test_data", model="gpt-4o-mini"):
        self.chat_history = ChatMessageHistory()
        self.llm = OpenAIModel(model=model)
        self.embeddings = AIEmbeddings()
        self.docs = LoadDocs(doc_path).process_documents()
        self.vector = LocalVector(self.docs, self.embeddings.get_embeddings())
        self.retriever = self.vector.get_retriever()
        self.histor_aware_retriever = self.vector.history_aware_retriever(llm=self.llm.get_llm())
        self.chain = custom_chains.retrieval_chain(
                llm=self.llm.get_llm(),
                retriever=self.histor_aware_retriever,
                prompt=get_helpful_assistant_prompt()
            )



    def run_chain(self, input, history):
        self.chat_history = history
        retrieved_docs = self.retriever.invoke(input)
        for doc in retrieved_docs:
            pprint(doc.metadata['source'])

        for doc in retrieved_docs:
            pprint(doc.page_content)

        return self.chain.stream({"input": input, "context": retrieved_docs, "chat_history": [{"role": m.role, "content": m.content} for m in history]})