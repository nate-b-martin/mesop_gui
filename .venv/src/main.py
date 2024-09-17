
import mesop as me
import mesop.labs as mel
from ollama_chat import OllamaChat
from lm_studio_chat import LmStudioChat
from model.LocalModel import LocalModel
from chat.LocalChat import LocalChat
from chat.OpenChat import OpenChat 
import poke_api
from langchain_core.runnables.utils import AddableDict
from dotenv import load_dotenv
import os

load_dotenv()

# chat_bot = LocalChat(LocalModel().get_llm(), doc_path='E:\Projects\mesop\mesop_gui\.venv\src\\test_data')
# ollama_chat = OllamaChat()
lm_studio_chat = LmStudioChat()
# _________ Classes _________
@me.stateclass
class State:
    pokemon_one:str = ""
    pokemon_two:str = ""
    input:str = ""

# _________ Pages _________
@me.page(path="/")
def app():
    # pokemon one
    me.input(label="Pokemon One", on_input=pokemon_one_input)

    # pokemon two 
    me.input(label="Pokemon Two", on_input=pokemon_two_input)

    # Load pokemon
    me.button(label="Load Pokemon", on_click=load_pokemon_button_one)

@me.page(path="/chat")
def chat():
    mel.chat(transform)

# _________ Functions _________

def pokemon_one_input(e: me.InputEvent) -> str:
    state = me.state(State)
    state.pokemon_one= e.value

def pokemon_two_input(e: me.InputEvent) -> str:
    state = me.state(State)
    state.pokemon_two= e.value

def transform(input:str, history: list[mel.ChatMessage]):
    chat_history = [{"role": message.role, "content": message.content} for message in history]

    yield from lm_studio_chat.run_chain_streaming(input, chat_history)
    
def load_pokemon_button_one(e: me.ClickEvent) -> None:
    state = me.state(State)
    print(state.pokemon_one)
    print(state.pokemon_two)
    poke_api.load_pokemon(state.pokemon_one)
    poke_api.load_pokemon(state.pokemon_two)
    me.navigate("/chat")
    
