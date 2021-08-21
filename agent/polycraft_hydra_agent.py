import random

from agent.hydra_agent import HydraAgent, HydraPlanner, MetaModelRepair
from worlds.polycraft_world import *

class PolycraftHydraAgent(HydraAgent):
    ''' A Hydra agent for Polycraft of all the Hydra agents '''
    def __init__(self, planner : HydraPlanner = None,
                 meta_model_repair : MetaModelRepair = None):
        super().__init__(planner, meta_model_repair)

    def choose_action(self, world_state:PolycraftState):
        ''' Choose which action to perform in the given state '''

        # Choose random action types
        chosen_action = self._choose_random(world_state)

        return chosen_action

    def _choose_random(self, world_state: PolycraftState):
        ''' Choose a random action from the list of po'''
        action_class = random.choice([PolyTP, PolyEntityTP, PolyTurn,PolyTilt, PolyTilt, PolyBreak, PolyInteract
                       PolySelectItem, PolyUseItem, PolyPlaceItem, PolyCollect, PolyDeleteItem, PolyTradeItems,
                       PolyCraftItem])

        if action_class == PolyTP:
            coordinate = random.choice(world_state.game_map.keys()).split(",")
            return PolyTP(coordinate[0], coordinate[1], coordinate[2])
        elif action_class == PolyEntityTP:
            npc = random.choice(world_state.npcs)
            return PolyEntityTP(npc)
        elif action_class == PolyTurn:
            dir= random.choice(range(0,360,15))
            return PolyTurn(dir)
        elif action_class == PolyTilt:
            angle = random.choice(range(0,180,45))
            return PolyTilt(angle)
        elif action_class == PolyBreak:
            return PolyBreak()
        elif action_class == PolySelectItem:
            item_name = random.choice(world_state.inventory)
            return PolySelectItem(item_name)
        elif action_class == PolyUseItem:
            return PolyUseItem()
        elif action_class == PolyPlaceItem:
            item_name = random.choice(world_state.inventory)
            return PolyPlaceItem(item_name)
        elif action_class == PolyCollect:
            return PolyCollect()
        elif action_class == PolyTradeItems:
            # Choose trade from list of possible trades
            raise NotImplementedError("Add trades to state")
            # return PolyTradeItems(random_trade)
        elif action_class == PolyCraftItem:
            # Choose recipe from list of recipes
            raise NotImplementedError("Add recipes  to state")
            # return PolyCraftItem(random_recipe)
        else:
            raise ValueError("Bad action class {}".format(action_class))

    def should_repair(self, observation):
        ''' Choose if the agent should repair its meta model based on the given observation '''
        raise NotImplementedError()

    def repair_meta_model(self, observation):
        ''' Call the repair object to repair the current meta model '''
        raise NotImplementedError()
