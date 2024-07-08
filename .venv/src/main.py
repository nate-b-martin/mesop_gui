import mesop as me
import mesop.labs as mel
import chat_model
import poke_api
import json
from langchain_core.runnables.utils import AddableDict


# _________ Classes _________
@me.stateclass
class State:
    pokemon_one:str = ""
    pokemon_two:str = ""
    input:str = ""

# _________ Pages _________
@me.page(path="/")
def app():
    s = me.state(State)
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

def transform(prompt:str, history: list[mel.ChatMessage]) -> str:
    conversation_chain = chat_model.test_chain()
    # print(prompt, history)
    history_string = ""
    response = conversation_chain.stream({
        "input": prompt,
        "chat_history": [{"role": m.role, "content": m.content} for m in history],
    })

    for r in response:
        dict = AddableDict(r)
        if dict.get('answer') is not None:
            yield str(dict.get('answer'))
        

    # for message in conversation_chain.stream({
    #     "input": prompt,
    #     "chat_history": [{"role": m.role, "content": m.content} for m in history],
    # }):
    #     print(str(message))
        
        
    
def load_pokemon_button_one(e: me.ClickEvent) -> None:
    state = me.state(State)
    print(state.pokemon_one)
    print(state.pokemon_two)
    poke_api.load_pokemon(state.pokemon_one)
    poke_api.load_pokemon(state.pokemon_two)
    me.navigate("/chat")
    
