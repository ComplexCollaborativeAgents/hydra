''' Module containing helper functions for the polycaft domain '''
import math

from worlds.polycraft_world import PolycraftState, ItemType, EntityType


def has_new_item(state_diff: dict):
    ''' Check if the state diff dictionary reports on a new item in the inventory '''
    if "inventory" not in state_diff:
        return False
    for inventory_item, diff_attr in state_diff["inventory"].items():
        if diff_attr['other']==None: # New item
            return True
        if diff_attr['self']['count'] > diff_attr['other']['count']: # Count incremented
            return True
    return False

def get_new_entity_items(state_diff: dict):
    ''' returns the list of new entity items in the state diff dictionary '''
    entities_diff = state_diff["entities"]
    new_entity_items = []
    for entity_id, entity_attr in entities_diff.items():
        if entity_attr['self']['type'] == 'EntityItem':
            new_entity_items.append((entity_id, entity_attr['self']))
    return new_entity_items

def print_diff(state_diff: dict):
    ''' For debug: pretty printing of a state diff '''
    for item, item_attr in state_diff.items():
        if "self" in item_attr:
            print(f'{item}:{item_attr}')
        else:
            for sub_item, sub_item_attr in state_diff[item].items():
                print(f'{item}.{sub_item}:{sub_item_attr}')


def get_recipe_tree(state: PolycraftState, item_type :ItemType):
    ''' A helper function that computes the recipe tree for creating a given ItemType '''
    item_type_to_recipes = dict()
    open = [item_type.value]
    closed = set()
    while len(open)>0:
        item_type = open.pop()
        closed.add(item_type)

        recipes = []
        recipe_indices = state.get_recipe_indices_for(item_type)
        for recipe_idx in recipe_indices:
            recipe_obj = state.recipes[recipe_idx]
            type_to_count = dict()
            for recipe_input in recipe_obj['inputs']:
                input_item_type = recipe_input['Item']
                if input_item_type not in type_to_count:
                    type_to_count[input_item_type]=1
                else:
                    type_to_count[input_item_type]=type_to_count[input_item_type]+1

                if input_item_type not in closed:
                    open.append(input_item_type)

            recipe = []
            for type, count in type_to_count.items():
                recipe.append((type, count))
            recipes.append(recipe)
        if len(recipes)>0:
            item_type_to_recipes[item_type] = recipes

    return item_type_to_recipes


def print_recipe_tree(item_type_to_recipes: dict(), root_item_type: ItemType):
    ''' A helper function that prints a given item type to recipe in the form of a tree rooted by the given item_type '''
    open = [('', root_item_type.value)]
    closed = []

    while len(open)>0:
        (indent, item_type) = open.pop()
        closed.append(item_type)

        if item_type in item_type_to_recipes:
            print(f'{indent} Recipes for {item_type}')
            recipes = item_type_to_recipes[item_type]
            new_indent = f"\t{indent}"
            for i, recipe in enumerate(recipes):
                print(f'{new_indent} Recipe {i+1}/{len(recipes)} for {item_type}')
                new_new_indent = f'\t{new_indent}'
                for (type, count) in recipe:
                    print(f'{new_new_indent} {type} {count}')
                    if type in item_type_to_recipes:
                        if type not in closed:
                            open.append((new_indent, type))

def get_item_to_trades(state: PolycraftState):
    ''' A helper function that computes the recipe tree for creating a given ItemType '''
    item_type_to_trades = dict()
    for trader_entity_id, trades in state.trades.items():
        for trade in trades:
            outputs = trade['outputs']
            assert(len(outputs)==1) # TODO: Consider this assumption that a trade only outputs a single item type
            output = outputs[0]
            item_type = output['Item']
            if item_type not in item_type_to_trades:
                item_type_to_trades[item_type] = []
            item_type_to_trades[item_type].append( (trader_entity_id, trade) )
    return item_type_to_trades

def get_trades_for(state:PolycraftState, desired_inputs, desired_outputs):
    ''' Return a list of (trader_id, trade) tuples for trades of the specified format '''
    results = []
    for trader_entity_id, trades in state.trades.items():
        for trade in trades:
            if trade['outputs']==desired_outputs and trade['inputs']==desired_inputs:
                results.append((trader_entity_id, trade))
    return results

def print_item_to_trades(items_to_trades: dict):
    ''' A helper function that prints the item to trade dictionary in a pretty way'''
    for item_type, trades in items_to_trades.items():
        print(f'Trades to obtain {item_type}:')
        for (trader, trade) in trades:
            print(f'\t Trader {trader} wants ')
            for input in trade['inputs']:
                print(f'\t \t {input}')

def cell_to_coordinates(cell:str)->list:
    ''' Converts a cell id of the form "x,y,z" to a list of int coordinates [int(x),int(y),int(z)]'''
    return [int(coord) for coord in cell.split(",")]

def coordinates_to_cell(coordinates: list)->str:
    ''' Converts a list of cell coordinates ([x,y,z]) to a cell id ("x,y,z") '''
    return ",".join([str(coord) for coord in coordinates])

def compute_cell_distance(pos1:list, pos2:list):
    ''' Computes Euclidean distance between two vectors. TODO: Understand why math.dist() is not working in our conda environment '''
    return math.sqrt(sum([(pos1[i] - pos2[i]) ** 2 for i in range(len(pos1))]))


def distance_to_nearest_pogoist(after_state, cell):
    ''' Compute the distance between the given cell and the nearest entity of type POGOIST'''
    min_distance_to_pogoist = 100000 # Infinity TODO: Make nicer
    for entity, entity_attr in after_state.entities.items():
        if entity_attr['type'] == EntityType.POGOIST.value:
            cell_coord = cell_to_coordinates(cell)
            entity_coord = entity_attr['pos']
            dist = compute_cell_distance(cell_coord, entity_coord)
            if dist<min_distance_to_pogoist:
                min_distance_to_pogoist = dist
    return min_distance_to_pogoist