import pokebase as pd
import random
import os
import json

def get_moves(pokemon):

    moves = pd.pokemon(pokemon.lower()).moves
    random_moves = random.sample(moves, 4)
    return [move.move.name for move in random_moves]
    # return [move.move.name for move in moves]

def get_move_data(move_name, attribute):
    move = pd.move(move_name.lower())
    return {attribute: move[attribute]}



def get_stats(pokemon):
    return {stat.stat.name: stat.base_stat for stat in pd.pokemon(pokemon.lower()).stats}


def get_abilities(pokemon):
    abilities = pd.pokemon(pokemon.lower()).abilities
    ability_names = [ability.ability.name for ability in abilities]
    return ability_names

def get_weaknesses(pokemon):
    weaknesses = pd.pokemon(pokemon.lower()).weaknesses
    weakness_types = [weakness.type.name for weakness in weaknesses]
    return weakness_types

def get_resistances(pokemon):
    resistances = pd.pokemon(pokemon.lower()).resistances
    resistance_types = [resistance.type.name for resistance in resistances]
    return resistance_types

def get_pokemon(pokemon):
    current_pokemon = pd.pokemon(pokemon.lower())
    return {
        "name": current_pokemon.name,
        "id": current_pokemon.id,
        "height": current_pokemon.height,
        "weight": current_pokemon.weight,
        "base_experience": current_pokemon.base_experience,
        "stats": get_stats(current_pokemon.name),
        "abilities": get_abilities(current_pokemon.name),
        "moves": get_moves(current_pokemon.name)
    }

def load_pokemon_data():
    pokemon_one = str(input("Pokemon 1: "))
    pokemon_two = str(input("Pokemon 2: "))
    pokemon_data_list = [get_pokemon(pokemon_one), get_pokemon(pokemon_two)]

    current_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(current_dir, "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    with open(os.path.join(test_data_dir, "pokemon_data_list.json"), "w") as f:
        json.dump(pokemon_data_list, f, indent=4)

def load_pokemon(pokemon):
    data = get_pokemon(pokemon)
    current_dir = os.path.dirname(__file__)
    test_data_dir = os.path.join(current_dir, "./test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    pokemon_data_file = os.path.join(test_data_dir, "pokemon_data_list.json")
    if os.path.isfile(pokemon_data_file):
        with open(pokemon_data_file, "r+") as f:
            pokemon_data_list = json.load(f)
            if data not in pokemon_data_list:
                pokemon_data_list.append(data)
                f.seek(0)
                json.dump(pokemon_data_list, f, indent=4)
    else:
        with open(pokemon_data_file, "w") as f:
            json.dump([data], f, indent=4)


if __name__ == "__main__":
    load_pokemon('dragonite') 
