from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI 
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from prompts.prompts import get_qa_prompt, get_battle_announcer_prompt, get_contextualize_q_prompt
from pprint import pprint


def get_loader():
    loader = DirectoryLoader('.venv/src/poke_data', glob="**/*.json", loader_cls=TextLoader)

    documents = loader.load()
    return documents

def split_text(documents:str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    texts = text_splitter.split_documents([documents])
    return texts

def get_embeddings():
    embedding = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434",temperature=0.7)
    return embedding

def set_vector_db(embeddings, texts:list):
    vector_db = FAISS.from_documents(documents=texts, embedding=embeddings)
    return vector_db

def set_retriever(vector_db, llm, retriever ):
    retriever = vector_db.as_retriever()
    retriever.search_kwargs["k"] = 2

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, 
    )

def chat_chain():
    # llm = Ollama(model="gemma2", temperature=0.7, base_url="http://localhost:11434/")
    llm = ChatOpenAI(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        temperature=0,
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        max_tokens=4096
    )

def default_llm() -> ChatOpenAI:
    llm = ChatOpenAI(model="llama3.1", temperature=0.7, base_url="http://localhost:1234/v1", api_key="lm-studio")
    # llm = Ollama(model="gemma2", temperature=0.7, base_url="http://localhost:11434/")
    return llm


def test_chain():
    human = "Use the conversation memory below to help in answer the most recent query from the user:{memory} USER_QUERY:{text} ASSISTANT:"

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", get_battle_announcer_prompt()),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    llm = default_llm()
    loader = DirectoryLoader('/home/nathan/Documents/Projects/mesop_gui/.venv/src/poke_data', glob="**/*.json", loader_cls=TextLoader)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)

    embedding = OllamaEmbeddings(model="nomic-ai/nomic-embed-text-v1.5-GGUF", base_url="http://localhost:1234/v1",temperature=0.7)
    

    vector_db = FAISS.from_documents(documents=texts, embedding=embedding)
    first_question = "What are all the available pokemon?"
    docs = vector_db.similarity_search(first_question)
    len(docs)
    for doc in docs:
        pprint(doc.page_content)

    retriever = vector_db.as_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever,prompt=get_contextualize_q_prompt()
    )
    question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
    rag_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=question_answer_chain)
    return rag_chain

if __name__ == "__main__":
    test_chain().invoke("What are all the available pokemon?")


