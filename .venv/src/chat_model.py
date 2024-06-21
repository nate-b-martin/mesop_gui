from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from prompts.prompts import get_qa_prompt, get_battle_announcer_prompt, get_contextualize_q_prompt


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

def set_vector_db(embeddings:OllamaEmbeddings, texts:list):
    vector_db = Chroma.from_documents(documents=texts, embedding=embeddings)
    return vector_db

def set_retriever(vector_db:Chroma, llm, retriever ):
    retriever = vector_db.as_retriever()
    retriever.search_kwargs["k"] = 2

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, 
    )

def chat_chain():
    llm = Ollama(model="llama3", temperature=0.7, base_url="http://localhost:11434/")

    system = "You are a helpful assistant. Please, be brief and concise and to the point. Answer in as few words as possible but still give the user the info they are after. Your name is batman and you are 50 years old"

    human = "Use the conversation memory below to help in answer the most recent query from the user:{memory} USER_QUERY:{text} ASSISTANT:"

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", human),
    ])

    conversation_chain = prompt | llm
    return conversation_chain

def default_llm():
    llm = Ollama(model="llama3", temperature=0.7, base_url="http://localhost:11434/")
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

    embedding = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434",temperature=0.7)

    vector_db = Chroma.from_documents(documents=texts, embedding=embedding)

    retriever = vector_db.as_retriever()
    retriever.search_kwargs = {"k": 2}

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever,prompt=get_contextualize_q_prompt()
    )
    question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
    rag_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=question_answer_chain)
    return rag_chain

if __name__ == "__main__":
    conversation_chain = test_chain()
    response = conversation_chain.invoke({"input": "lets start a battle between bulbasaur and charmander", "chat_history": []})
    print(response)

