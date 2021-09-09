import random
from worlds.polycraft_interface.client.polycraft_interface import TiltDir

from agent.hydra_agent import HydraAgent, HydraPlanner, MetaModelRepair
from worlds.polycraft_world import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Polycraft")

class PolycraftHydraAgent(HydraAgent):
    ''' A Hydra agent for Polycraft of all the Hydra agents '''
    def __init__(self, planner: HydraPlanner = None,
                 meta_model_repair: MetaModelRepair = None):
        super().__init__(planner, meta_model_repair)

    def choose_action(self, world_state: PolycraftState):
        ''' Choose which action to perform in the given state '''

        logger.info("World state summary is: {}".format(str(world_state)))

        # Choose random action types
        chosen_action = self._choose_random(world_state)

        return chosen_action

    def _choose_random(self, world_state: PolycraftState):
        ''' Choose a random action from the list of possible actions '''
        action_class = random.choice([PolyTP, PolyEntityTP, PolyTurn, PolyTilt, PolyTilt, PolyBreak, PolyInteract,
                                      PolySelectItem, PolyUseItem, PolyPlaceItem, PolyCollect, PolyDeleteItem, PolyTradeItems,
                                      PolyCraftItem])

        if action_class == PolyTP:
            coordinate = random.choice(list(world_state.game_map.keys())).split(",")
            return PolyTP(coordinate[0], coordinate[1], coordinate[2])
        elif action_class == PolyEntityTP:
            npcs = [npc for npc in world_state.entities.keys()]
            if len(npcs) > 0:
                npc = random.choice(npcs)
                return PolyEntityTP(npc)
            else:
                return PolyNoAction()
        elif action_class == PolyTurn:
            dir = random.choice(range(0, 360, 15))
            return PolyTurn(dir)
        elif action_class == PolyTilt:
            angle = random.choice([pitch.value for pitch in list(TiltDir)])    # Choose from DOWN, FORWARD, and UP
            return PolyTilt(angle)
        elif action_class == PolyBreak:
            return PolyBreak()
        elif action_class == PolySelectItem:
            items = [item['item'] for item in world_state.inventory.values()]
            if len(items) > 0:
                item_name = random.choice(items)
                return PolySelectItem(item_name)
            else:
                return PolyNoAction()   # No items to select
        elif action_class == PolyUseItem:
            return PolyUseItem()
        elif action_class == PolyPlaceItem:
            items = [item['item'] for item in world_state.inventory.values()]
            if len(items) > 0:
                item_name = random.choice(items)
                return PolyPlaceItem(item_name)
            else:
                return PolyNoAction()   # No items to place
        elif action_class == PolyDeleteItem:
            items = [item['item'] for item in world_state.inventory.values()]
            if len(items) > 0:
                item_name = random.choice(items)
                return PolyDeleteItem(item_name)
            else:
                return PolyNoAction()   # No items to place
        elif action_class == PolyCollect:
            return PolyCollect()
        elif action_class == PolyInteract:
            npcs = [npc for npc in world_state.entities.keys()]
            if len(npcs) > 0:
                npc = random.choice(npcs)
                return PolyInteract(npc)
            else:
                return PolyNoAction()   # Do nothing.
        elif action_class == PolyTradeItems:
            # Choose trade from list of possible trades
            if len(world_state.trades) > 0:
                random_trade = random.choice(world_state.trades)
            else:
                return PolyNoAction()
            return PolyTradeItems(random_trade['entity_id'], random_trade['input'])
        elif action_class == PolyCraftItem:
            if len(world_state.trades) > 0:
                random_recipe = random.choice(world_state.recipes)
            else:
                return PolyNoAction()
            return PolyCraftItem(random_recipe)
        else:
            raise ValueError("Bad action class {}".format(action_class))

    def should_repair(self, observation):
        ''' Choose if the agent should repair its meta model based on the given observation '''
        raise NotImplementedError()

    def repair_meta_model(self, observation):
        ''' Call the repair object to repair the current meta model '''
        raise NotImplementedError()
