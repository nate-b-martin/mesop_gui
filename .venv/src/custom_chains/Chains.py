from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from prompts.prompts import get_qa_prompt

def create_document_chain(llm, prompt):
    return create_stuff_documents_chain(llm, get_qa_prompt(prompt))

def retrieval_chain(retriever, llm, prompt):
    """
    Creates a retrieval chain.
    
    Note: The retriever should be history-aware for optimal performance.
    
    Args:
        retriever: A history-aware retriever object.
        llm: The language model to use.
        prompt: The prompt to use for the chain.
    
    Returns:
        A retrieval chain.
    """
    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=create_document_chain(llm, prompt),
    )