from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_battle_announcer_prompt() -> str:
    battle_announcer_prompt = """
                Please comment on the battle status based on the given context. Provide options for each pokemon's move. Track pokemon health after each attack. List available pokemon and their stats for user to choose from by using the given context {context}.
            """
    return battle_announcer_prompt


def get_contextualize_q_prompt() -> ChatPromptTemplate:
    contextualize_q_system_prompt = """
                Given a chat history and the latest user question 
                which might reference context in the chat history, 
                formulate a standalone question which can be understood 
                without the chat history. Do NOT answer the question, 
                "just reformulate it if needed and otherwise return it as is.
            """
            

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_q_prompt


def get_qa_prompt(system_prompt:str,) -> ChatPromptTemplate:
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

    return qa_prompt
