from model.LocalModel import LocalModel
from embed.LocalEmbeddings import LocalEmbeddings
from embed.LoadDocs import LoadDocs 
from embed.LocalVector import LocalVector 
import custom_chains.Chains as custom_chains
from prompts.prompts import get_helpful_assistant_prompt,get_qa_prompt
from chat.LocalChat import LocalChat
from dotenv import load_dotenv

load_dotenv()

llm = LocalModel()
embeddings = LocalEmbeddings()
docs = LoadDocs("/home/nathan/Documents/Projects/mesop_gui/.venv/src/poke_data").process_documents()
vector = LocalVector(docs, embeddings.get_embeddings())
histor_aware_retriever = vector.history_aware_retriever(llm=llm.get_llm())

chain = custom_chains.retrieval_chain(
    llm=llm.get_llm(),
    retriever=histor_aware_retriever,
    prompt=get_helpful_assistant_prompt()
    )


chat = LocalChat(chain=chain, retriever=vector.get_retriever())

chat.run_chain()