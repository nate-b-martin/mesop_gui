import os
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS, SKLearnVectorStore
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from prompts.prompts import get_contextualize_q_prompt, get_qa_prompt
from dotenv import load_dotenv
from pprint import pprint
from langchain_core.runnables.utils import AddableDict
import json
load_dotenv()

# create llm model
open_ai_llm = ChatOpenAI(
    model = "gpt-4o-mini",
)

# load docs
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "test_data")

loaders = {
    "pdf": DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    ),
    "txt": DirectoryLoader(
        data_path, 
        loader_cls=TextLoader, 
        glob="**/*.txt"
     ),
    "json": DirectoryLoader(
        data_path, 
        loader_cls=TextLoader, 
        glob="**/*.json"
     ),
    "csv": DirectoryLoader(data_path, loader_cls=CSVLoader, glob="**/*.csv"),
    }

docs = []
for loader in loaders.values():
    loader = loader.load()
    docs.extend(loader)

# split docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
split_text = text_splitter.split_documents(docs)
# for chunk in split_text:
#     print(chunk.page_content)
#     print("-"*100)

# print(f'number of docs: {len(split_text)}')
# print(f'split text: {split_text}')

# create embeddings, Have to include check_embedding_ctx_length=False to avoid error
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# create vector db
vector_db = Chroma.from_documents(documents=split_text, embedding=embeddings)

# create retriever
retriever = vector_db.as_retriever()

history_aware_retriever = create_history_aware_retriever(
    llm=open_ai_llm, retriever=retriever, prompt=get_contextualize_q_prompt()
)

question_answer_chain = create_stuff_documents_chain(llm=open_ai_llm, prompt=get_qa_prompt(
    """
    Context: {context} 
    You are a helpful assistant that will use the context to answer the question. If you can't formulate an answer from the context, just say "I don't know" and try to answer the question to your best ability.
    Question: {input}
    """))

rage_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain,
)

# create chat history
chat_history = ChatMessageHistory()

while True:
    question = input("\nQuestion: ")
    if question == "exit":
        break

    chat_history.add_user_message(question)
    retrieved_docs = retriever.invoke(question)

    full_answer = ''
    for r in rage_chain.stream({"input": question, "context": retrieved_docs, "chat_history": chat_history.messages}):
        dict = AddableDict(r)
        if dict.get('answer') is not None:
            print(str(dict.get('answer')), end='')
            full_answer += str(dict.get('answer'))

    chat_history.add_ai_message(str(full_answer))
