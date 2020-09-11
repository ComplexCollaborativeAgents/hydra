from shapely.geometry import box, Polygon

import worlds.science_birds as sb
from worlds.science_birds_interface.computer_vision.game_object import GameObject
from worlds.science_birds_interface.computer_vision.GroundTruthReader import GroundTruthReader
import settings
import json
import numpy as np
import logging
import pickle
from utils.state import State, Action, World


from worlds.science_birds import SBState
import csv
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

class Perception():
    def __init__(self):
        '''Taken from naive_agent_groundtruth'''
        self.model = np.loadtxt("{}/data/science_birds/perception/model".format(settings.ROOT_PATH), delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('{}/data/science_birds/perception/target_class'.format(settings.ROOT_PATH)).readlines()))
        self.new_level = True
        self.writer = csv.DictWriter(open('object_class.csv','w'), fieldnames=classification_cols())
        self.writer.writeheader()




    def process_state(self, state): # TODO: This may need to be removed
        if isinstance(state,sb.SBState):
            return self.process_sb_state(state)
        return state



# Output: {0: {'type': 'redBird', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145b10>}, 1: {'type': 'slingshot', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145590>}, 2: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4690>}, 3: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4510>}, 4: {'type': 'pig', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4450>}}
    def process_sb_state(self,state):
        vision = GroundTruthReader(state.objects, self.model, self.target_class)

        if self.new_level:
            for obj in vision.alljson:
                if 'coordinates' in obj['geometry']:
                    self.writer.writerow(self.obj_dictionary(obj))
            self.new_level = False

        # try:
        #     vision = GroundTruthReader(state.objects,self.model,self.target_class)
        # except Exception as e:
        #     logger.info ("perception exception: {}".format(e.__str__()))
        #     logger.info("perception failed on state: {}".format(state))
        #     return None
        sling = vision.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width # ScienceBirds reverses width and height


        # Birds get the smallest IDs. This helps a bit to track the birds TODO: Discuss a better approach
        bird_types = []
        for type, objs in vision.allObj.items():
            if "bird" in type.lower():
                bird_types.append(type)
        bird_types = sorted(bird_types)

        id = 0
        new_objs = {}
        for bird_type in bird_types:
            for obj in vision.allObj[bird_type]:
                obj: GameObject
                poly = Polygon([(i[0], i[1]) for i in obj.vertices])
                new_objs[id] = {'type': bird_type,
#                                'bbox': box(obj.top_left[0], obj.top_left[1],
#                                            obj.bottom_right[0], obj.bottom_right[1])
                                'polygon':poly}
                id += 1

        for type, objs in vision.allObj.items():
            if type in bird_types:
                continue

            for obj in objs:
                obj: GameObject
                poly = Polygon([(i[0], i[1]) for i in obj.vertices])
                new_objs[id] = {'type':type,
                                #'bbox':box(obj.top_left[0],obj.top_left[1],
                                #            obj.bottom_right[0],obj.bottom_right[1]),
                                'polygon':poly
                                }
                if type == 'unknown':
                    for state_obj in state.objects:
                        if 'vertices' in state_obj and obj.vertices == state_obj['vertices']:
                            new_objs[id]['colormap'] = state_obj['colormap']
                    assert 'colormap' in new_objs[id].keys()
                id+=1

        return ProcessedSBState(state, new_objs, vision.find_slingshot_mbr()[0])

    def add_qsrs_to_input(self,dictionary):
        '''augments symbolic input with qualitative spatial relationships and returns
            a dictionary.
            How do we want to pass knowledge around? Are dictionaries or tuples the
            right structures'''
        return dictionary

    def obj_dictionary(self,obj):
        colors = np.zeros(256)
        row = {}
        row['class'] = obj['properties']['label']
        row['num_vertices'] = len(obj['geometry']['coordinates'][0]) #just exterior
        poly = Polygon(obj['geometry']['coordinates'][0])
        row['area'] = poly.area
        for color_pair in obj['properties']['colormap']:
            colors[color_pair['color']]=color_pair['percent']
        for color in range(0,256):
            row['color{}'.format(color)] = colors[color]
        return row

def classification_cols():
    colormap = ['color{}'.format(x) for x in range(0, 256)]
    cols = ['class', 'num_vertices',  'area']
    cols.extend(colormap)
    return cols

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


