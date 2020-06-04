from shapely.geometry import box

import worlds.science_birds as sb
from worlds.science_birds_interface.computer_vision.GroundTruthReader import GroundTruthReader
import settings
import json
import numpy as np
import logging
import pickle
from utils.state import State, Action, World

from worlds.science_birds import SBState

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("perception")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

class Perception():
    def __init__(self):
        '''Taken from naive_agent_groundtruth'''
        f = open(settings.SB_INIT_COLOR_MAP, 'r')
        result = json.load(f)
        self.look_up_matrix = np.zeros((len(result), 256))
        self.look_up_obj_type = np.zeros(len(result)).astype(str)
        obj_number = 0
        for d in result:
            if 'effects_21' in d['type']:
                obj_name = 'Platform'
            elif 'effects_34' in d['type']:
                obj_name = 'TNT'
            elif 'ice' in d['type']:
                obj_name = 'Ice'
            elif 'wood' in d['type']:
                obj_name = 'Wood'
            elif 'stone' in d['type']:
                obj_name = 'Stone'
            else:
                obj_name = d['type'][:-2]
            obj_color_map = d['colormap']
            self.look_up_obj_type[obj_number] = obj_name
            for pair in obj_color_map:
                self.look_up_matrix[obj_number][int(pair['x'])] = pair['y']
            obj_number += 1
        # normalise the look_up_matrix
        self.look_up_matrix = self.look_up_matrix / np.sqrt((self.look_up_matrix ** 2).sum(1)).reshape(-1, 1)

    def process_state(self, state): # TODO: This may need to be removed
        if isinstance(state,sb.SBState):
            return self.process_sb_state(state)
        return state

# Output: {0: {'type': 'redBird', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145b10>}, 1: {'type': 'slingshot', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145590>}, 2: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4690>}, 3: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4510>}, 4: {'type': 'pig', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4450>}}
    def process_sb_state(self,state):
        try:
            vision = GroundTruthReader(state.objects,self.look_up_matrix,self.look_up_obj_type)
        except:
            logger.info("perception failed on state: {}".format(state))
            return None
        state.sling = vision.find_slingshot_mbr()[0]
        # state.sling.width, state.sling.height = state.sling.height, state.sling.width TODO: Verify w. Wiktor/Matt
        new_objs = {}
        id = 0
        for type, objs in vision.allObj.items():
            for obj in objs:
                new_objs[id] = {'type':type,
                                'bbox':box(obj.top_left[0],obj.top_left[1],
                                            obj.bottom_right[0],obj.bottom_right[1])}
                if type == 'unknown':
                    for state_obj in state.objects:
                        if 'vertices' in state_obj and obj.vertices == state_obj['vertices']:
                            new_objs[id]['colormap'] = state_obj['colormap']
                    assert 'colormap' in new_objs[id].keys()
                id+=1
        state.objects = new_objs

        return ProcessedSBState(state, new_objs, vision.find_slingshot_mbr()[0])

    def add_qsrs_to_input(self,dictionary):
        '''augments symbolic input with qualitative spatial relationships and returns
            a dictionary.
            How do we want to pass knowledge around? Are dictionaries or tuples the
            right structures'''
        return dictionary


''' A Science bird after it was processed by the perception module '''
class ProcessedSBState(State):
    def __init__(self, sb_state: SBState, objects_dict, slingshot_obj):
        super().__init__()
        self.objects = objects_dict
        self.sling = slingshot_obj

        self.image = sb_state.image
        self.game_state = sb_state.game_state

    def summary(self):
        '''returns a summary of state'''
        ret = {}
        for key, obj in self.objects.items():
            ret['{}_{}'.format(obj['type'], key)] = (obj['bbox'].centroid.x, obj['bbox'].centroid.y)
        return ret

    def serialize_current_state(self, level_filename):
        pickle.dump(self, open(level_filename, 'wb'))

    def load_from_serialized_state(level_filename):
        return pickle.load(open(level_filename, 'rb'))
