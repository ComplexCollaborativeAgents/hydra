import sys

from shapely.errors import TopologicalError

import settings
import os.path
sys.path.append(os.path.join(settings.ROOT_PATH, "worlds", "science_birds_interface"))

from shapely.geometry import box, Polygon

import worlds.science_birds as sb
from worlds.science_birds_interface.computer_vision.game_object import GameObject
from worlds.science_birds_interface.computer_vision.GroundTruthReader import GroundTruthReader
#import agent.consistency.observation
import settings
import json
import numpy as np
import logging
import pickle
from utils.state import State, Action, World
import sklearn.linear_model as lm

from worlds.science_birds import SBState
import csv
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

if settings.SB_COLLECT_PERCEPTION_DATA:
    # from utils.assess_classification import OBJECT_CLASSES
    from utils.assess_classification import object_class_convert

class Perception():
    def __init__(self):
        '''Taken from naive_agent_groundtruth'''
        self.model = np.loadtxt("{}/data/science_birds/perception/model".format(settings.ROOT_PATH), delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('{}/data/science_birds/perception/target_class'.format(settings.ROOT_PATH)).readlines()))
        self.logreg = pickle.load(open('{}/data/science_birds/perception/logreg_pII.p'.format(settings.ROOT_PATH),'rb'))
        self.threshold = settings.SB_CLASSIFICATION_THRESHOLD
        self.new_level = True
        self.writer = csv.DictWriter(open('object_class.csv','w'), fieldnames=classification_cols())
        if settings.DEBUG:
            self.writer.writeheader()


    def process_observation(self, ob):
#        if isinstance(ob,agent.consistency.observation.ScienceBirdObservation):
        if True:
            processed_states = []
            for state in ob.intermediate_states:
                processed_state = self.process_sb_state(state)
                for id, obj in processed_state.objects.items():
                    obj['type'] = ob.state.type_in_state(id) if ob.state.type_in_state(id) else obj['type']
                    processed_state.objects[id] = obj
                processed_states.append(processed_state)
            ob.intermediate_states=processed_states
        return True

    def process_state(self, state): # TODO: This may need to be removed
        if isinstance(state,sb.SBState):
            return self.process_sb_state(state)
        return state

        
# Output: {0: {'type': 'redBird', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145b10>}, 1: {'type': 'slingshot', 'bbox': <shapely.geometry.polygon.Polygon object at 0x112145590>}, 2: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4690>}, 3: {'type': 'wood', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4510>}, 4: {'type': 'pig', 'bbox': <shapely.geometry.polygon.Polygon object at 0x1120f4450>}}
    def process_sb_state(self, state):
        vision = GroundTruthReader(state.objects, self.model, self.target_class)

        if self.new_level and settings.DEBUG:
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



        new_objs = {}
        platforms = []
        for obj in vision.alljson:
            if obj['properties']['label'] == 'Slingshot':
                new_obj ={'type':'slingshot',
                          'polygon':Polygon(obj['geometry']['coordinates'][0])}
                new_objs[obj['properties']['id']] = new_obj
            elif obj['geometry'] and obj['geometry']['type'] != 'MultiPoint': #if it is not the ground or a trajectory
                if settings.SB_COLLECT_PERCEPTION_DATA:
                    type = self.classify_object_for_data_collection(obj)
                else:
                    type = self.classify_obj(obj)
                new_obj = {'type':type,
                            'polygon':Polygon(obj['geometry']['coordinates'][0])}
                if type == 'unknown':
                    new_obj['colormap']=obj['properties']['colormap']
                if type == 'platform':
                    new_obj['id'] = obj['properties']['id']
                    platforms.append(new_obj)
                else:
                    new_objs[obj['properties']['id']] = new_obj

        for platform in self.merge_platforms(platforms):
            new_objs[platform['id']] = platform

        # bird_types = []
        # for type, objs in vision.allObj.items():
        #     if "bird" in type.lower():
        #         bird_types.append(type)
        # bird_types = sorted(bird_types)
#         id = 0
#         new_objs = {}
#         for bird_type in bird_types:
#             for obj in vision.allObj[bird_type]:
#                 obj: GameObject
#                 poly = Polygon([(i[0], i[1]) for i in obj.vertices])
#                 new_objs[id] = {'type': bird_type,
# #                                'bbox': box(obj.top_left[0], obj.top_left[1],
# #                                            obj.bottom_right[0], obj.bottom_right[1])
#                                 'polygon':poly}
#                 id += 1
#
#         for type, objs in vision.allObj.items():
#             if type in bird_types:
#                 continue
#
#             for obj in objs:
#                 obj: GameObject
#                 poly = Polygon([(i[0], i[1]) for i in obj.vertices])
#                 new_objs[id] = {'type':type,
#                                 #'bbox':box(obj.top_left[0],obj.top_left[1],
#                                 #            obj.bottom_right[0],obj.bottom_right[1]),
#                                 'polygon':poly
#                                 }
#                 if type == 'unknown':
#                     for state_obj in state.objects:
#                         if 'vertices' in state_obj and obj.vertices == state_obj['vertices']:
#                             new_objs[id]['colormap'] = state_obj['colormap']
#                     assert 'colormap' in new_objs[id].keys()
#                 id+=1

        return ProcessedSBState(state, new_objs, vision.find_slingshot_mbr()[0])

    def merge_platforms(self,platforms):
        '''
        merge all platforms that are touching one another into a single object
        return a list of platforms
        '''
        ids = [k['id'] for k in platforms]
        touches = dict((id,[]) for id in ids)

        for plat_1 in platforms:
            for plat_2 in platforms:
                if plat_1['id'] != plat_2['id'] and plat_1['polygon'].intersects(plat_2['polygon']):
                    touches[plat_1['id']].append(plat_2)

        ret = []
        open_list = []
        try:
            for plat in platforms:
                poly = plat['polygon']
                if plat['id'] not in ids:
                    continue
                ids.remove(plat['id'])
                open_list = touches[plat['id']]
                while open_list:
                    plat_2 = open_list.pop(0)
                    if plat_2['id'] in ids:
                        poly = poly.union(plat_2['polygon'])
                        ids.remove(plat_2['id'])
                        open_list.extend(touches[plat_2['id']])
                ret.append({'id':plat['id'],'polygon':poly,'type':'platform'})
        except TopologicalError:
            logging.info("Ill-formed platform. Returning an empty list for the platforms")
        return ret

    def classify_object_for_data_collection(self, obj_json, translate_to_features=True):
        object_label = obj_json['properties']['label']
        # object_class = OBJECT_CLASSES[object_label]
        object_class = object_class_convert(object_label)
        return object_class


    #Translate to features is only false in testing.
    def classify_obj(self,obj_json,translate_to_features=True):
        feature_vector = self.obj_features(obj_json) if translate_to_features else obj_json
        prediction = self.logreg.predict_proba([feature_vector])
        pred_type = self.logreg.classes_[prediction[0].argmax()]
        probability = max(prediction[0])
        if probability > self.threshold:
            return pred_type
        else:
            return 'unknown'


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

    def obj_features(self, obj, no_class=True):
        dict = self.obj_dictionary(obj)
        ret = [dict[col] for col in classification_cols()]
        if no_class:
            return ret[1:]
        else:
            return ret

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

    def novel_objects(self):
        ''' Returns a list of the novel objects '''
        ret = []
        for key, obj in self.objects.items():
            if obj['type'] is 'unknown':
                ret.append([key,obj])
        return ret

    def serialize_current_state(self, level_filename):
        pickle.dump(self, open(level_filename, 'wb'))

    def type_in_state(self,id):
        if id in self.objects.keys():
            return self.objects[id]['type']
        return None

    def load_from_serialized_state(level_filename):
        return pickle.load(open(level_filename, 'rb'))


