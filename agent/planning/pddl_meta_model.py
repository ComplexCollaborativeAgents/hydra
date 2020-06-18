import copy
import math
from agent.perception.perception import ProcessedSBState

from agent.planning.pddlplus_parser import *
import settings
import logging

from agent.perception.perception import Perception

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

# TODO: See how to merge the two functions below with the two above
def get_slingshot_x(slingshot):
    return round((slingshot[1]['bbox'].bounds[0] + slingshot[1]['bbox'].bounds[2]) / 2) - 0 # TODO: Why this minus zero
def get_slingshot_y(groundOffset, slingshot):
    return round(abs(((slingshot[1]['bbox'].bounds[1] + slingshot[1]['bbox'].bounds[3]) / 2) - groundOffset) - 0)
def get_scale(slingshot):
    # sling.width + sling.height
    return abs(round(slingshot[1]['bbox'].bounds[0] - slingshot[1]['bbox'].bounds[2])) + round(abs(slingshot[1]['bbox'].bounds[1] - slingshot[1]['bbox'].bounds[3]))


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

    ''' Subclasses should override this setting all attributes of that object '''
    def _compute_obj_attributes(self, obj, problem_params:dict):
        return dict()

    ''' Subclasses should override this settign all attributes of that object that can be observed'''
    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
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

    ''' Populate a PDDL+ state with details about this object '''
    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params:dict):
        name = self._get_name(obj)
        attributes = self._compute_observable_obj_attributes(obj, state_params)
        for attribute in attributes:
            value = attributes[attribute]
            fluent_name = (attribute, name)
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value,  bool):
                if value==True:
                    pddl_state.boolean_fluents.add(fluent_name)
                # TODO: Think how to handle booean fluents with False value. Not as trivial as it sounds
            else: # Attribute is a number
                pddl_state.numeric_fluents[fluent_name]=value

    def _get_name(self, obj):
        type = obj[1]['type']
        return '{}_{}'.format(type, obj[0])


''' The slingshot is currently not directly modeled in our model, so its object is currently ignored. 
TODO: Reconsider this design choice. '''
class SlingshotType:
    ''' Populate a PDDL+ problem with details about this object '''
    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params:dict):
        return # Do nothing, slingshot is currently not directly modeled as an object

    ''' Populate a PDDL+ state with details about this object '''
    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params:dict):
        return # Do nothing, slingshot is currently not directly modeled as an object




class PigType(PddlObjectType):
    def __init__(self):
        super(PigType,self).__init__()
        self.pddl_type = "pig"
        self.hyper_parameters["m_pig"]=1
        self.hyper_parameters["pddl_type"]="pig"

    def _compute_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)
        obj_attributes["m_pig"] = self.hyper_parameters["m_pig"]

        return obj_attributes

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()
        obj_attributes["x_pig"] = get_x_coordinate(obj)
        obj_attributes["y_pig"] = get_y_coordinate(obj, problem_params["groundOffset"])
        obj_attributes["pig_radius"] = get_radius(obj)
        obj_attributes["pig_dead"] = False

        problem_params["pigs"].add(self._get_name(obj))

        return obj_attributes

class BirdType(PddlObjectType):
    def __init__(self):
        super(BirdType, self).__init__()
        self.pddl_type = "bird"
        # self.hyper_parameters["v_bird"] = 270
        self.hyper_parameters["vx_bird"] = 0
        self.hyper_parameters["vy_bird"] = 0
        self.hyper_parameters["m_bird"] = 1
        self.hyper_parameters["bounce_count"] = 0
        self.hyper_parameters["bird_released"] = False

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)

        # obj_attributes["v_bird"] = self.hyper_parameters["v_bird"]
        obj_attributes["vx_bird"] = self.hyper_parameters["vx_bird"]
        obj_attributes["vy_bird"] = self.hyper_parameters["vy_bird"]
        obj_attributes["m_bird"] = self.hyper_parameters["m_bird"]
        obj_attributes["bounce_count"] = self.hyper_parameters["bounce_count"]
        obj_attributes["bird_released"] = self.hyper_parameters["bird_released"]

        return obj_attributes

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        obj_attributes = dict()

        slingshot = problem_params["slingshot"]
        groundOffset = problem_params["groundOffset"]

        # In the initial state, the planner assumes the birds are on the sling
        slingshot_x = get_slingshot_x(slingshot)
        slingshot_y = get_slingshot_y(groundOffset, slingshot)

        if "initial_state" in problem_params and problem_params["initial_state"] == True:
            obj_attributes["x_bird"] = slingshot_x
            obj_attributes["y_bird"] = slingshot_y
            obj_attributes["v_bird"] = (9.5 / 2.7) * (get_scale(slingshot))
        else:
            obj_attributes["x_bird"] = get_x_coordinate(obj)  # TODO: Why this minos zero?
            obj_attributes["y_bird"] = get_y_coordinate(obj, groundOffset)  # TODO: Why this minos zero?

            # Need to separate the case where we're before shooting the bird and after.
            # Before: the bird location is considered as the location of the slingshot,
            # afterwards, it's the location of the birds bounding box TODO: Replace this hack
            if obj_attributes["x_bird"] <= slingshot_x:
                obj_attributes["x_bird"] = slingshot_x
                obj_attributes["y_bird"] = slingshot_y


        obj_attributes["bird_id"] = problem_params["bird_index"]
        problem_params["bird_index"] = problem_params["bird_index"] + 1

        problem_params["birds"].add(self._get_name(obj))

        return obj_attributes

class PlatformType(PddlObjectType):
    def __init__(self):
        super(PlatformType, self).__init__()
        self.pddl_type ="platform"

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        problem_params["has_platform"] = True

        obj_attributes = dict()

        obj_attributes["x_platform"] = get_x_coordinate(obj)
        obj_attributes["y_platform"] = get_y_coordinate(obj, problem_params["groundOffset"])

        obj_attributes["platform_height"] = get_height(obj)
        obj_attributes["platform_width"] = get_width(obj)
        return obj_attributes

    def _compute_obj_attributes(self, obj, problem_params: dict):
        return self._compute_observable_obj_attributes(obj, problem_params)

class BlockType(PddlObjectType):
    def __init__(self,block_life_multiplier = 1.0, block_mass_coeff=1.0):
        super(BlockType, self).__init__()
        self.pddl_type = "block"

        self.hyper_parameters["block_life_multiplier"] = block_life_multiplier
        self.hyper_parameters["block_mass_coeff"] = block_mass_coeff

    def _compute_observable_obj_attributes(self, obj, problem_params:dict):
        problem_params["has_block"] = True

        obj_attributes = dict()

        groundOffset = problem_params["groundOffset"]
        obj_attributes["x_block"] = get_x_coordinate(obj)
        obj_attributes["y_block"] = get_y_coordinate(obj, groundOffset)
        obj_attributes["block_height"] = get_height(obj)
        obj_attributes["block_width"] = get_width(obj)
        return obj_attributes

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)

        groundOffset = problem_params["groundOffset"]
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
        return 265 * (bl_width / (bl_height + 1)) \
               * (1 - (bl_y / (groundOffset + 1))) \
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

        for (fluent, value) in [('active_bird', 0),
                                ('angle', 0),
                                ('angle_rate', 20),
                                ('ground_damper', 0.3)]:
            self.constant_numeric_fluents[fluent]=value

        for not_fluent in ['angle_adjusted',
                           'pig_killed']:
            self.constant_boolean_fluents[not_fluent]=False

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
        self.object_types["slingshot"] = SlingshotType()

    ''' Get the ralation between angle and twang time TODO: Think if this is right or not'''
    def get_angle_rate(self):
        return float(self.constant_numeric_fluents["angle_rate"])

    ''' Get the slingshot object '''
    def get_slingshot(self, sb_state :ProcessedSBState):
        sling = None
        for o in sb_state.objects.items():
            if o[1]['type'] == 'slingshot':
                sling = o
        return sling

    ''' Translate the initial SBState, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. '''
    def create_pddl_problem(self, sb_state : ProcessedSBState):
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
        slingshot = self.get_slingshot(sb_state)

        # A dictionary with global problem parameters
        problem_params = dict()
        problem_params["has_platform"]=False
        problem_params["has_block"]=False
        problem_params["bird_index"]=0
        problem_params["slingshot"]=slingshot
        problem_params["groundOffset"] = self.get_ground_offset(slingshot)
        problem_params["gravity"] =  0.48*9.81 / 2.7 * get_scale(slingshot)
        # Above line redundant since we're storing the slingshot also, but it seems easier to store it also to save computations of the offset everytime we use it.
        problem_params["pigs"] = set()
        problem_params["birds"] = set()
        problem_params["initial_state"]=True # This marks that SBState describes the initial state. Used for setting the bird's location in the slingshot's location. TODO: Reconsider this design choice

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
                    logger.debug("Unknown object type: %s" % type_str)
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
        prob.init.append(['=', ['gravity'], problem_params["gravity"]])

        # Add goal
        pigs = problem_params["pigs"]
        assert len(pigs)>0
        for pig in pigs:
            prob.goal.append(['pig_dead', pig])

        return prob

    ''' Get the ground offset '''
    def get_ground_offset(self, slingshot):
        return slingshot[1]['bbox'].bounds[3]

    ''' Translate the given observed SBState to a PddlPlusState object. 
    This is designed to handle intermediate state observed during execution '''
    def create_pddl_state(self, sb_state:ProcessedSBState):
        pddl_state = PddlPlusState()

        # we should probably use the self.sling on the object
        slingshot = self.get_slingshot(sb_state)

        # A dictionary with global problem parameters
        state_params = dict()
        state_params["has_platform"] = False
        state_params["has_block"] = False
        state_params["bird_index"] = 0
        state_params["slingshot"] = slingshot
        state_params["groundOffset"] = self.get_ground_offset(slingshot)
        # Above line redundant since we're storing the slingshot also, but it seems easier to store it also to save computations of the offset everytime we use it.
        state_params["pigs"] = set()
        state_params["birds"] = set()
        state_params["initial_state"] = False  # This marks that SBState describes the initial state. Used for setting the bird's location in the slingshot's location. TODO: Reconsider this design choice

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
            type.add_object_to_state(pddl_state, obj, state_params)

        return pddl_state

    ''' Creates a PDDL+ domain object for the given SB state 
    Current implementation simply copies an existing domain file '''
    def create_pddl_domain(self, sb_State: ProcessedSBState):
        domain_file = "%s/sb_domain.pddl" % str(settings.PLANNING_DOCKER_PATH)
        domain_parser = PddlDomainParser()
        return domain_parser.parse_pddl_domain(domain_file)

    ''' Create a simplified version of the given problem, to help the planner plan 
    TODO: Move this to planner'''
    def create_simplified_problem(self, prob : PddlPlusProblem):
        prob_simplified = PddlPlusProblem()
        prob_simplified.name = copy.copy(prob.name)
        prob_simplified.domain = copy.copy(prob.domain)
        prob_simplified.objects = copy.copy(prob.objects)

        removed_list = []
        first_bird_spotted = False
        for obj in prob.objects:
            if ((obj[1] == 'bird') and not first_bird_spotted):
                first_bird_spotted = True
            elif obj[1] == 'bird':
                removed_list.append(obj[0])
                prob_simplified.objects.remove(obj)

        prob_simplified.init = copy.copy(prob.init)

        for init_stmt in prob.init:
            for b_name in removed_list:
                if (len(init_stmt[1]) > 1) and (init_stmt[1][1] == b_name):
                    prob_simplified.init.remove(init_stmt)

        prob_simplified.metric = copy.copy(prob.metric)
        prob_simplified.goal = list()
        prob_simplified.goal.append(['pig_killed'])
        return prob_simplified

    ''' Created an even more simplified version of the problem to speed up planner ?'''
    def create_super_simplified_problem(self, prob :PddlPlusProblem):
        prob_super_simplified = PddlPlusProblem()
        prob_super_simplified.name = copy.copy(prob.name)
        prob_super_simplified.domain = copy.copy(prob.domain)
        prob_super_simplified.objects = copy.copy(prob.objects)

        removed_list = []
        super_removed_list = []
        first_bird_spotted = False
        for obj in prob.objects:
            if ((obj[1] == 'bird') and not first_bird_spotted):
                first_bird_spotted = True
            elif obj[1] == 'bird':
                removed_list.append(obj[0])
                prob_super_simplified.objects.remove(obj)
            elif (obj[1] == 'block'):
                super_removed_list.append(obj[0])
                prob_super_simplified.objects.remove(obj)

        prob_super_simplified.init = copy.copy(prob.init)

        for init_stmt in prob.init:
            for b_name in removed_list:
                if (len(init_stmt[1]) > 1) and (init_stmt[1][1] == b_name):
                    prob_super_simplified.init.remove(init_stmt)
            for bl_name in super_removed_list:
                if ((len(init_stmt[1]) > 1) and (init_stmt[1][1] == bl_name)) or (init_stmt[1] == bl_name):
                    prob_super_simplified.init.remove(init_stmt)

        prob_super_simplified.objects.append(['dummy_block', 'block'])
        prob_super_simplified.metric = copy.copy(prob.metric)
        prob_super_simplified.goal = list()
        prob_super_simplified.goal.append(['pig_killed'])

        return prob_super_simplified