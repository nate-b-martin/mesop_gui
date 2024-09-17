from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from embed.LoadDocs import LoadDocs
from vector_utils.utils import find_recently_modified_files
from typing import List, Generator
import os

class MesopStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def get_tokens(self):
        return self.tokens

class OllamaChat:
    """
    Class to run a chat with Ollama.
    """

    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.stream_handler = MesopStreamHandler()
        self.llm = ChatOllama(
            model=self.model_name,
            base_url="http://localhost:11434",
            keep_alive="10m",
            temperature=0.1,
            num_gpu=-1,  # Use -1 to utilize all available GPUs on Windows
            num_ctx=8096,
            streaming=True,
            callbacks=[self.stream_handler]
        )
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)

        # load docs
        self.test_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "test_data")
        self.docs = self.load_docs()

        self.vector_store = self.create_vector_store()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = self.create_qa_chain()

    def create_vector_store(self):
        return Chroma.from_documents(
            documents=self.docs,
            embedding=self.embeddings,
            persist_directory="./vector_store",
        )

    def load_docs(self) -> list:
        load = LoadDocs(data_path=self.test_directory)
        docs = load.process_documents()
        return docs

    def update_vector_store(self):
        print(f'updating modified files')
        updated_docs = find_recently_modified_files()
        print(f'updated: {updated_docs}')
        load_docs = LoadDocs(self.test_directory)
        new_split = load_docs.load_modified_docs(updated_docs)

        if new_split:
            print(f'Modified files in the last 5 minutes: {updated_docs}')
            
            # Update the vector store with new documents
            self.vector_store.add_documents(new_split)
            print(f'Vector store updated with {len(new_split)} new documents.')
        else:
            print('No files modified in the last 5 minutes.')

    def create_qa_chain(self):
        prompt_template = """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Helpful Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "alpha": 0.5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=False,
            chain_type="stuff",
            verbose=True
        )

    def run_chain(self, input: str, history):
        return self.qa_chain({"question": input, "chat_history": history})

    def run_chain_streaming(self, input: str, history: List) -> Generator[str, None, None]:
        # if modified docs are found update the vector store
        self.update_vector_store()

        self.stream_handler.tokens = []  # Reset tokens
        self.qa_chain({"question": input, "chat_history": history})
        for token in self.stream_handler.get_tokens():
            yield token