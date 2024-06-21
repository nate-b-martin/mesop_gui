from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_battle_announcer_prompt() -> str:
    battle_announcer_prompt = """
                You are a pokemon battle referee and commentating the battle. Using the context provided, please comment on the battle status. Context: {context}. Keep in mind that we want the user to choose what happens after each move alternating between each pokemon. Also please provide a list of four possible options based off the context provided for the user to pick. We also want to keep track of the pokemon's health after each attack. Before the battle starts, please provide a list of pokemon and their stats for the user to pick from.   
            """
    return battle_announcer_prompt


def get_contextualize_q_prompt() -> ChatPromptTemplate:
    contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

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
