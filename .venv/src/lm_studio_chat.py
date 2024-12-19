from langchain.storage import InMemoryStore
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from embed.LoadDocs import LoadDocs
from vector_utils.utils import find_recently_modified_files
from typing import List, Generator
import mesop.labs as mel
from dotenv import load_dotenv
import os
load_dotenv()

class MesopStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def get_tokens(self):
        return self.tokens

class LmStudioChat:
    """
    Class to run a chat with Lm Studio.
    """

    def __init__(self, model_name: str = "meta-llama_-_llama-3.2-3b-instruct"):
        self.model_name = model_name
        self.stream_handler = MesopStreamHandler()
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url="http://localhost:1234/v1", 
            api_key="lm-studio",
            streaming=True,
            callbacks=[self.stream_handler]
        )
        self.embeddings = OpenAIEmbeddings(
                model="CompendiumLabs/bge-large-en-v1.5-gguf", 
                base_url="http://localhost:1234/v1", 
                api_key="lm-studio", 
                show_progress_bar=True,
                check_embedding_ctx_length=False,
                chunk_size=1000,
            )

        # load docs
        self.test_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "test_data")
        self.docs = self.load_docs()

        self.vector_store = self.create_vector_store()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            output_key="answer",
            return_messages=True
            )
        self.qa_chain = self.create_qa_chain()
    
    def get_embeddings(self, text, model:str = "nomic-ai/nomic-embed-text-v1.5-GGUF"):
        text = text.replace("\n", " ")
        return self.llm.embeddings.create(input = [text], model=model).data[0].embedding

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

    def create_compression_retriever(self):
        compressor = LLMChainExtractor.from_llm(self.llm)

        base_retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor = compressor,
            base_retriever = base_retriever
        )

        return compression_retriever



    def create_qa_chain(self):
            prompt_template = """
            Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Helpful Answer:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


            store = InMemoryStore()
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.create_compression_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                chain_type="stuff",
                verbose="True"
            )

    def run_chain(self, input: str, history):
        return self.qa_chain({"question": input, "chat_history": history})

    def run_chain_streaming(self, input: str, history: List[mel.ChatMessage]) -> Generator[str, None, None]:
        # if modified docs are found update the vector store
        self.update_vector_store()

        self.stream_handler.tokens = []  # Reset tokens
        self.qa_chain({"question": input, "chat_history": history})
        for token in self.stream_handler.get_tokens():
            yield token