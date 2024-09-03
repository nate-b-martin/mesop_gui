import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.LocalModel import LocalModel
from chat.base_chat import BaseChat
from embed.LocalEmbeddings import LocalEmbeddings
from embed.LoadDocs import LoadDocs 
from embed.LocalVector import LocalVector 
from custom_chains.Chains import retrieval_chain
from langchain_core.runnables.utils import AddableDict
from langchain_chroma.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from prompts.prompts import get_helpful_assistant_prompt
from dotenv import load_dotenv
from pprint import pprint
import os
load_dotenv()

class LocalChat(BaseChat):
    def __init__(self, llm, doc_path="/home/nathan/Documents/Projects/mesop_gui/.venv/src/test_data"):
        super().__init__(llm, doc_path)
        self.embeddings = LocalEmbeddings().get_embeddings()
        self.docs = LoadDocs(doc_path).process_documents()
        self.vector = LocalVector(self.docs, self.embeddings)
        # self.base_retriever = self.vector.get_retriever()
        self.base_retriever = Chroma(persist_directory="./vector_store",
            embedding_function=self.embeddings).as_retriever(search_type="similarity",search_kwargs= {"k": 5})
        self.history_aware_retriever = self.vector.history_aware_retriever(self.llm)
        self.chat_history = []

        # compressor = LLMChainExtractor.from_llm(self.llm)
        # self.retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor,
        #     base_retriever=self.base_retriever
        # )

    def run_qa_chain(self, input, history=None):

        qa_chain = retrieval_chain(
                llm=self.llm,
                retriever=self.history_aware_retriever,
                prompt=get_helpful_assistant_prompt()
            )

        retrieved_docs = self.base_retriever.invoke(input)

        for doc in retrieved_docs:
            if 'source' in doc.metadata:
                pprint(doc.metadata['source'])

        for doc in retrieved_docs:
            pprint(doc.page_content)

        response = qa_chain.stream(
            {
                "input": input, 
                "context": retrieved_docs, 
                "chat_history": [{"role": m.role, "content": m.content} for m in history]
            }
        )
        return response

    def get_chat_history(self):
        return self.chat_history
    
    def clear_chat_history(self):
        self.chat_history = []

if __name__ == "__main__":
    llm = LocalModel()
    chat = LocalChat(llm.get_llm())
    print('running chain')
    questions = [
                    "Hello, my name is Nathan.What are the names of all the pokemon in the context?",
                    "What are their moves?",
                    "What are their id's",
                    "What is my name?"
                 ]

    for question in questions:
        response = chat.run_qa_chain(question)

        for r in response:
            dict = AddableDict(r)
            if dict.get('answer') is not None:
                print(str(dict.get('answer')), end="")

