# Example: reuse your existing OpenAI setup
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI 
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import SKLearnVectorStore, Chroma
from langchain_nomic.embeddings import NomicEmbeddings 
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from prompts.prompts import get_qa_prompt, get_battle_announcer_prompt, get_contextualize_q_prompt
from langchain_core.runnables.utils import AddableDict
from dotenv import load_dotenv
import os
import jq


load_dotenv()

# GLOBALS
# EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"
MODEL = "llama3.1"
KEY = "your-key"

# Client
# client = ChatOllama(base_url="http://localhost:11434/", model="llama3.1", temperature=0)
# client = OpenAI(base_url="http://localhost:1234/v1", model="llama3.1", temperature=0)



# prompt

def load_docs():
    current_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(current_dir, "poke_data")
    os.makedirs(test_data_dir, exist_ok=True)
    loader = DirectoryLoader(path=test_data_dir, glob="**/*.json", loader_cls=TextLoader)
    return loader.load()

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    split_text = text_splitter.split_documents(docs)
    print('printing split text')
    print(spilt for spilt in split_text)
    return split_text

def vector_store(split_text):
    # embeddings = NomicEmbeddings(model=EMBEDDING_MODEL)
    embeddings = OpenAIEmbeddings(model="nomic-ai/nomic-embed-text-v1.5-GGUF", base_url="http://localhost:1234/v1", api_key="lm-studio", check_embedding_ctx_length=False)
    vector_db = Chroma.from_documents(documents=split_text, embedding=embeddings)
    return vector_db

def retriever(store):
    retriever = store.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# an example question about the 2022 Olympics
query = ''

docs = load_docs()
splits = split_docs(docs)
vector_store = vector_store(splits)
retriever = retriever(vector_store)

# prompt = PromptTemplate(
#     template="""You are an assistant for question-answering tasks. 
    
#     Use the following documents to answer the question. 
    
#     If you don't know the answer, just say that you don't know. 
    
#     Use three sentences maximum and keep the answer concise:
#     Documents: {documents} 
#     Question: {question} 
#     Answer: 
#     """,
#     input_variables=["question", "documents"],
# )


prompt = PromptTemplate(
    template="""You are an pokemon battle refferee and your tasks is to commentate the battle between two pokemon. 
    
    Use the following documents to keep the battle going. 
    
    Use some creativity to come up with battle scenarios using the users decisions, You will ask the user after each turn what they want to do next listing their options :
    Documents: {documents} 
    input: {input} 
    Answer: 
    """,
    input_variables=["input", "documents"],
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", get_battle_announcer_prompt()),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# llm for ollama server
# llm = ChatOllama(
#     model="llama3.1",
#     temperature=0,
# )

# llm for lm-studio server
llm = ChatOpenAI(
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    temperature=0,
    api_key="lm-studio",
    base_url="http://localhost:1234/v1",
    max_tokens=4096
)

history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=get_contextualize_q_prompt())

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=final_prompt)

rag_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=question_answer_chain)

# for stream in rag_chain.stream({"input": "Please give me a list of pokemon to choose form", "chat_history": []}):
#     print(stream)

# response = rag_chain.invoke({"input": "Please give me a list of pokemon to choose form", "chat_history": []})

while True:
    response = rag_chain.stream({"input": str(input('User Input: ')), "chat_history": []})

    for r in response:
        dict = AddableDict(r)
        if dict.get('answer') is not None:
            print(str(dict.get('answer')), end='')