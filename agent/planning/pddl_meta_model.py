'''
This module contains the meta-model according to which we generate PDDL+ problems for a given state observed in ScienceBirds.
It is designed so that it is mutable, i.e., one can modify the way the meta model generates PDDL+ problems.
'''


from agent.planning.pddl_plus import *
import copy
import math
from worlds.science_birds import SBState


import logging

fh = logging.FileHandler("hydra.log",mode='w')
formatter = logging.Formatter('%(asctime)-15s %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger("pddl_meta_model")
logger.setLevel(logging.INFO)
logger.addHandler(fh)


''' Utility functions '''
def get_x_coordinate(obj):
    return round(abs(obj[1]['bbox'].bounds[2] + obj[1]['bbox'].bounds[0]) / 2)
def get_y_coordinate(obj, groundOffset):
    return abs(round(abs(obj[1]['bbox'].bounds[1] + obj[1]['bbox'].bounds[3]) / 2) - groundOffset)
def get_radius(obj):
    return round((abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0]) / 2) * 0.75)
def get_height(obj):
    return abs(obj[1]['bbox'].bounds[3] - obj[1]['bbox'].bounds[1])
def get_width(obj):
    return abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0])

''' A generator for Pddl Objects '''
class PddlObjectType():
    ''' Accepts an object from SBState.objects '''
    def __init__(self):
        self.hyper_parameters = dict()
        self.pddl_type="object" # This the PDDL+ type of this object.

    ''' Subclasses should override this '''
    def _compute_obj_attributes(self, obj, problem_params:dict):
        return dict()

    ''' Populate a PDDL+ problem with details about this object '''
    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params:dict):
        name = self._get_name(obj)
        prob.objects.append([name, self.pddl_type])
        attributes = self._compute_obj_attributes(obj, problem_params)
        for attribute in attributes:
            value = attributes[attribute]
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value,  bool):
                if value==True:
                    prob.init.append([attribute, name])
                else: # value == False
                    prob.init.append(['not', [attribute, name]])
            else: # Attribute is a number
                prob.init.append(['=', [attribute, name], value])

    def _get_name(self, obj):
        type = obj[1]['type']
        return '{}_{}'.format(type, obj[0])



class PigType(PddlObjectType):
    def __init__(self):
        super(PigType,self).__init__()
        self.pddl_type = "pig"
        self.hyper_parameters["m_pig"]=1
        self.hyper_parameters["pddl_type"]="pig"

    def _compute_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()
        obj_attributes["x_pig"] = get_x_coordinate(obj)
        obj_attributes["y_pig"] = get_y_coordinate(obj, problem_params["groundOffset"])
        obj_attributes["pig_radius"] = get_radius(obj)
        obj_attributes["m_pig"] = 1
        obj_attributes["pig_dead"] = False

        problem_params["pigs"].append(self._get_name(obj))

        return obj_attributes

class BirdType(PddlObjectType):
    def __init__(self):
        super(BirdType, self).__init__()
        self.pddl_type = "bird"
        self.hyper_parameters["v_bird"] = 270
        self.hyper_parameters["vx_bird"] = 0
        self.hyper_parameters["vy_bird"] = 0
        self.hyper_parameters["m_bird"] = 1
        self.hyper_parameters["bounce_count"] = 0
        self.hyper_parameters["bird_released"] = False

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = dict()

        slingshot = problem_params["slingshot"]
        groundOffset = problem_params["groundOffset"]
        obj_attributes["x_bird"] = round((slingshot[1]['bbox'].bounds[0] + slingshot[1]['bbox'].bounds[2]) / 2) - 0 # TODO: Why this minos zero?
        obj_attributes["y_bird"] = round(abs(((slingshot[1]['bbox'].bounds[1] + slingshot[1]['bbox'].bounds[3]) / 2) - groundOffset) - 0) # TODO: Why this minos zero?

        yyy = get_y_coordinate(slingshot,groundOffset)
        if yyy!=obj_attributes["y_bird"]:
            print("wtf")

        obj_attributes["bird_id"] = problem_params["bird_index"]
        problem_params["bird_index"] = problem_params["bird_index"] + 1

        obj_attributes["v_bird"] = self.hyper_parameters["v_bird"]
        obj_attributes["vx_bird"] = self.hyper_parameters["vx_bird"]
        obj_attributes["vy_bird"] = self.hyper_parameters["vy_bird"]
        obj_attributes["m_bird"] = self.hyper_parameters["m_bird"]
        obj_attributes["bounce_count"] = self.hyper_parameters["bounce_count"]
        obj_attributes["bird_released"] = self.hyper_parameters["bird_released"]

        problem_params["birds"].append(self._get_name(obj))

        return obj_attributes



class PlatformType(PddlObjectType):
    def __init__(self):
        super(PlatformType, self).__init__()
        self.pddl_type ="platform"

    def _compute_obj_attributes(self, obj, problem_params: dict):
        problem_params["has_platform"] = True

        obj_attributes = dict()

        obj_attributes["x_platform"] = get_x_coordinate(obj)
        obj_attributes["y_platform"] = get_y_coordinate(obj, problem_params["groundOffset"])

        obj_attributes["platform_height"] = get_height(obj)
        obj_attributes["platform_width"] = get_width(obj)
        return obj_attributes

class BlockType(PddlObjectType):
    def __init__(self,block_life_multiplier = 1.0, block_mass_coeff=1.0):
        super(BlockType, self).__init__()
        self.pddl_type = "block"

        self.hyper_parameters["block_life_multiplier"] = block_life_multiplier
        self.hyper_parameters["block_mass_coeff"] = block_mass_coeff

    def _compute_obj_attributes(self, obj, problem_params: dict):
        problem_params["has_block"] = True

        obj_attributes = dict()

        groundOffset = problem_params["groundOffset"]
        obj_attributes["x_block"] = get_x_coordinate(obj)
        obj_attributes["y_block"] = get_y_coordinate(obj, groundOffset)
        obj_attributes["block_height"] = get_height(obj)
        obj_attributes["block_width"] = get_width(obj)
        obj_attributes["block_life"] = self.__compute_block_life()
        obj_attributes["block_mass"] = self.__compute_block_mass()
        obj_attributes["block_stability"] = self.__compute_stability(obj_attributes["block_width"],
                                                                     obj_attributes["block_height"],
                                                                     obj_attributes["y_block"],
                                                                     groundOffset)
        return obj_attributes

    def __compute_block_life(self):
        return str(math.ceil(265 * self.hyper_parameters["block_life_multiplier"]))
    def __compute_block_mass(self):
        return str(self.hyper_parameters["block_mass_coeff"])
    def __compute_stability(self, bl_width, bl_height, bl_y, groundOffset):
        return 265 * (bl_width / bl_height) \
               * (1 - (bl_y / groundOffset)) \
               * self.hyper_parameters["block_mass_coeff"]

class WoodType(BlockType):
    def __init__(self):
        super(WoodType, self).__init__(1.0, 0.375 * 1.3)

class IceType(BlockType):
    def __init__(self):
        super(IceType, self).__init__(0.5, 0.125*2)

class StoneType(BlockType):
    def __init__(self):
        super(StoneType, self).__init__(2.0, 1.2)

class TNTType(BlockType):
    def __init__(self):
        super(TNTType, self).__init__(0.001, 1.2)

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = super(TNTType, self)._compute_obj_attributes(obj, problem_params)
        obj_attributes["block_explosive"] = True
        return obj_attributes



class MetaModel():
    ''' Sets the default meta-model'''
    def __init__(self):
        # TODO: Read this from file instead of hard coding
        self.constant_numeric_fluents = dict()
        self.constant_boolean_fluents = dict()

        self.constant_numeric_fluents['gravity']=134.2
        self.constant_numeric_fluents['active_bird']=0
        self.constant_numeric_fluents['angle']=0
        self.constant_numeric_fluents['angle_rate'] = 10
        self.constant_numeric_fluents['ground_damper'] = 0.4

        self.constant_boolean_fluents['angle_adjusted']=False
        self.constant_boolean_fluents['pig_killed']=False

        self.metric = 'minimize(total-time)'

        # Mapping of type to Pddl object. All objects of this type will be clones of this pddl object
        self.object_types = dict()
        self.object_types["pig"]=PigType()
        self.object_types["bird"]=BirdType()
        self.object_types["block"]=BlockType()
        self.object_types["wood"]=WoodType()
        self.object_types["ice"] = IceType()
        self.object_types["stone"] = StoneType()
        self.object_types["TNT"] = TNTType()
        self.object_types["hill"] = PlatformType()


    ''' Get the sling object '''
    def get_sling(self, sb_state :SBState):
        sling = None
        for o in sb_state.objects.items():
            if o[1]['type'] == 'slingshot':
                sling = o
        return sling

    ''' Translate the initial SBState, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. '''
    def translate_sb_state_to_pddl_problem(self, sb_state : SBState):
        # There is an annoying disconnect in representations.
        # 'x_pig[pig_4]:450' vs. (= (x_pig pig4) 450)
        # 'pig_dead[pig_4]:False vs. (not (pig_dead pig_4))
        # We will use the PddlPlusProblem class as a common representations
        # init rep [['=', ['gravity'], '134.2'], ['=', ['active_bird'], '0'], ['=', ['angle'], '0'], ['=', ['angle_rate'], '20'], ['not', ['angle_adjusted']], ['not', ['bird_dead', 'redBird_0']], ['not', ['bird_released', 'redBird_0']], ['=', ['x_bird', 'redBird_0'], '192'], ['=', ['y_bird', 'redBird_0'], '29'], ['=', ['v_bird', 'redBird_0'], '270'], ['=', ['vy_bird', 'redBird_0'], '0'], ['=', ['bird_id', 'redBird_0'], '0'], ['not', ['wood_destroyed', 'wood_2']], ['=', ['x_wood', 'wood_2'], '445.0'], ['=', ['y_wood', 'wood_2'], '25.0'], ['=', ['wood_height', 'wood_2'], '12.0'], ['=', ['wood_width', 'wood_2'], '24.0'], ['not', ['wood_destroyed', 'wood_3']], ['=', ['x_wood', 'wood_3'], '447.0'], ['=', ['y_wood', 'wood_3'], '13.0'], ['=', ['wood_height', 'wood_3'], '13.0'], ['=', ['wood_width', 'wood_3'], '24.0'], ['not', ['pig_dead', 'pig_4']], ['=', ['x_pig', 'pig_4'], '449.0'], ['=', ['y_pig', 'pig_4'], '53.0'], ['=', ['margin_pig', 'pig_4'], '21']]
        # objects rep [('redBird_0', 'bird'), ('pig_4', 'pig'), ('wood_2', 'wood_block'), ('wood_3', 'wood_block'), ('dummy_ice', 'ice_block'), ('dummy_stone', 'stone_block'), ('dummy_platform', 'platform')]

        prob = PddlPlusProblem()
        prob.domain = 'angry_birds_scaled'
        prob.name = 'angry_birds_prob'
        prob.metric = self.metric
        prob.objects = []
        prob.init = []
        prob.goal = []

        #we should probably use the self.sling on the object
        slingshot = self.get_sling(sb_state)

        # A dictionary with global problem parameters
        problem_params = dict()
        problem_params["has_platform"]=False
        problem_params["has_block"]=False
        problem_params["bird_index"]=0
        problem_params["slingshot"]=slingshot
        problem_params["groundOffset"] = slingshot[1]['bbox'].bounds[3]
        # Above line redundant since we're storing the slingshot also, but it seems easier to store it also to save computations of the offset everytime we use it.
        problem_params["pigs"] = []
        problem_params["birds"] = []

        # Add objects to problem
        for obj in sb_state.objects.items():
            # Get type
            type_str = obj[1]['type']
            if 'bird' in type_str.lower():
                type = self.object_types["bird"]
            else:
                if type_str in self.object_types:
                    type = self.object_types[type_str]
                else:
                    logger.info("Unknown object type: %s" % type_str)
                    # TODO Handle unknown objects in some way (Error? default object?)
                    continue

            # Add object of this type to the problem
            type.add_object_to_problem(prob, obj, problem_params)

        # Add dummy platform and block if none exists TODO: Why do we need this?
        if problem_params["has_platform"]==False:
            prob.objects.append(['dummy_platform','platform'])
        if problem_params["has_block"]==False:
            prob.objects.append(['dummy_block','block'])

        # Add constants fluents
        for numeric_fluent in self.constant_numeric_fluents:
            prob.init.append(['=', [numeric_fluent], self.constant_numeric_fluents[numeric_fluent]])
        for boolean_fluent in self.constant_boolean_fluents:
            if self.constant_boolean_fluents[boolean_fluent]:
                prob.init.append([boolean_fluent])
            else:
                prob.init.append(['not',[boolean_fluent]])

        # Add goal
        pigs = problem_params["pigs"]
        assert len(pigs)>0
        for pig in pigs:
            prob.goal.append(['pig_dead', pig])

        prob_simplified = self.create_simplified_problem(prob)
        return prob, prob_simplified

    ''' Create a simplified version of the given problem, to help the planner plan '''
    def create_simplified_problem(self, prob : PddlPlusProblem):
        prob_simplified = PddlPlusProblem()
        prob_simplified.name = copy.copy(prob.name)
        prob_simplified.domain = copy.copy(prob.domain)
        prob_simplified.objects = copy.copy(prob.objects)
        prob_simplified.init = copy.copy(prob.init)
        prob_simplified.metric = copy.copy(prob.metric)
        prob_simplified.goal = list()
        prob_simplified.goal.append(['pig_killed'])
        return prob_simplified

