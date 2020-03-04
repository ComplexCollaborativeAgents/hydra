import sys
import os
import settings
print(sys.path)
sys.path.append(os.path.join(settings.ROOT_PATH, 'worlds', 'science_birds_interface'))
print(sys.path)
from computer_vision.VisionRealShape import VisionRealShape
from computer_vision.GroundTruthReader import GroundTruthReader,NotVaildStateError
import worlds.science_birds as sb
from shapely.geometry import box

from computer_vision.game_object import GameObject


class Perception():


    def __init__(self):
        pass

    def process_state(self, state):
        if isinstance(state,sb.SBState):
            return self.process_sb_state(state)
        return state

# Output: {0: {'type': 'redBird', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145b10>}, 1: {'type': 'slingshot', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145590>}, 2: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4690>}, 3: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4510>}, 4: {'type': 'pig', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4450>}}
    def process_sb_state(self,state):
        vision = GroundTruthReader(state.objects)
        state.sling = vision.find_slingshot_mbr()[0]
        state.sling.width, state.sling.height = state.sling.height, state.sling.width
        new_objs = {}
        id = 0
        for type, objs in vision.allObj.items():
            for obj in objs:
                new_objs[id] = {'type':type,
                                'bbox':box(obj.top_left[0],obj.top_left[1],
                                            obj.bottom_right[0],obj.bottom_right[1])}
                id+=1
        state.objects = new_objs
        return state




    def add_qsrs_to_input(self,dictionary):
        '''augments symbolic input with qualitative spatial relationships and returns
            a dictionary.
            How do we want to pass knowledge around? Are dictionaries or tuples the
            right structures'''
        return dictionary