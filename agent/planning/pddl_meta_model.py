import copy
import math
from agent.perception.perception import ProcessedSBState

from agent.planning.pddlplus_parser import *
import settings
import logging

from utils.point2D import Point2D

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
    return round((abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0]) / 2) * 0.9)
def get_height(obj):
    return abs(obj[1]['bbox'].bounds[3] - obj[1]['bbox'].bounds[1])
def get_width(obj):
    return abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0])

''' Returns the location of the closest object to aim for '''
def get_closest_object_xy(pddl_problem : PddlPlusProblem):
    state = PddlPlusState(pddl_problem.init)
    target_pigs = state.get_pigs()
    closest_obj_x = None
    closest_obj_y = None
    assert len(target_pigs) > 0
    for pig in target_pigs:
        x_pig = state[('x_pig', pig)]
        y_pig = state[('y_pig', pig)]
        if (closest_obj_x == None) or (
                math.sqrt(x_pig ** 2 + y_pig ** 2) < math.sqrt(closest_obj_x ** 2 + closest_obj_y ** 2)):
            closest_obj_x = x_pig
            closest_obj_y = y_pig
            # print("\n\nNEW CLOSEST TARGET: " + pig + "(" + str(closest_obj_x) + ", " + str(closest_obj_y) + ")\n")
    for plat in state.get_platforms():
        x_plat = state[('x_platform', plat)]
        y_plat = state[('y_platform', plat)]
        if (y_plat - ((state[('y_height', plat)]) / 2)) >= 10.0:
            y_plat = 0
        if (closest_obj_x == None) or (
                math.sqrt(x_plat ** 2 + y_plat ** 2) < math.sqrt(closest_obj_x ** 2 + closest_obj_y ** 2)):
            closest_obj_x = x_plat
            closest_obj_y = y_plat
            # print("\n\nNEW CLOSEST TARGET: " + plat + "(" + str(closest_obj_x) + ", " + str(closest_obj_y) + ")\n")
    for blo in state.get_blocks():
        x_blo = state[('x_block', blo)]
        y_blo = state[('y_block', blo)]
        if (closest_obj_x == None) or (
                math.sqrt(x_blo ** 2 + y_blo ** 2) < math.sqrt(closest_obj_x ** 2 + closest_obj_y ** 2)):
            closest_obj_x = x_blo
            closest_obj_y = y_blo
            # print("\n\nNEW CLOSEST TARGET: " + blo + "(" + str(closest_obj_x) + ", " + str(closest_obj_y) + ")\n")
    return closest_obj_x, closest_obj_y

''' Returns a min/max launch angle to hit the given target point '''
def estimate_launch_angle(slingshot, targetPoint, meta_model):
    # calculate relative position of the target (normalised)
    scale_factor = meta_model.hyper_parameters['scale_factor']
    _velocity = 9.5 / scale_factor
    X_OFFSET = 0.45
    Y_OFFSET = 0.6

    ground_offset = slingshot[1]['bbox'].bounds[3]

    scale = get_scale(slingshot)

    # print ('scale ', scale)
    # System.out.println("scale " + scale)
    # ref = Point2D(int(slingshot.X + X_OFFSET * slingshot.width), int(slingshot.Y + Y_OFFSET * slingshot.height))
    ref = Point2D(int(get_slingshot_x(slingshot) + X_OFFSET * get_width(slingshot)), int(Y_OFFSET * get_height(slingshot)))
    # print ('ref point', str(ref))
    x = (targetPoint.X - ref.X)
    y = (targetPoint.Y - ref.Y)

    # print ('sling X', get_slingshot_x(slingshot))
    # print ('sling Y', get_slingshot_y(ground_offset, slingshot))

    # print ('X', x)
    # print ('Y', y)

    # gravity
    g = 0.48 * meta_model.constant_numeric_fluents['gravity_factor'] / scale_factor * scale
    # print('gravity', g)
    # launch speed
    v = _velocity * scale
    # print ('launch speed ', v)

    solution_existence_factor = v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2

    # the target point cannot be reached
    if solution_existence_factor < 0:
        print ('\n\nNO SOLUTION!\n\n')
        return 0.0, 90.0

    # solve cos theta from projectile equation

    cos_theta_1 = math.sqrt(
        (x ** 2 * v ** 2 - x ** 2 * y * g + x ** 2 * math.sqrt(v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2)) / (2 * v ** 2 * (x ** 2 + y ** 2)))
    cos_theta_2 = math.sqrt(
        (x ** 2 * v ** 2 - x ** 2 * y * g - x ** 2 * math.sqrt(v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2)) / (2 * v ** 2 * (x ** 2 + y ** 2)))
    #        print ('cos_theta_1 ', cos_theta_1, ' cos_theta_2 ', cos_theta_2)

    distance_between = math.sqrt(x ** 2 + y ** 2)  # ad-hoc patch

    theta_1 = math.acos(cos_theta_1) + distance_between * 0.0001  # compensate the rounding error
    # print('theta 1', math.degrees(theta_1))
    theta_2 = math.acos(cos_theta_2) + distance_between * 0.00005  # compensate the rounding error
    # print('theta 2', math.degrees(theta_2))

    return math.floor(math.degrees(theta_1)), math.ceil(math.degrees(theta_2))

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
            obj_attributes["v_bird"] = round((9.5 / 2.7) * (get_scale(slingshot)))
        else:
            obj_attributes["x_bird"] = get_x_coordinate(obj)
            obj_attributes["y_bird"] = get_y_coordinate(obj, groundOffset)

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

        obj_attributes["platform_height"] = get_height(obj) * 1.1
        obj_attributes["platform_width"] = get_width(obj) * 1.1
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
        return round(265 * (bl_width / (bl_height + 1)) \
               * (1 - (bl_y / (groundOffset + 1))) \
               * self.hyper_parameters["block_mass_coeff"])


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
        self.hyper_parameters = dict() # These are parameters that do not appear in the PDDL files
        self.hyper_parameters['scale_factor']= 2.7

        self.constant_numeric_fluents = dict()
        self.constant_boolean_fluents = dict()


        for (fluent, value) in [('active_bird', 0),
                                # ('angle', 0),
                                ('angle_rate', 20),
                                ('ground_damper', 0.3),
                                ('gravity_factor', 9.81)]:
            self.constant_numeric_fluents[fluent]=value

        for not_fluent in ['angle_adjusted',
                           # 'increasing',
                           # 'decreasing',
                           'pig_killed'
                           ]:
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

        pddl_problem = PddlPlusProblem()
        pddl_problem.domain = 'angry_birds_scaled'
        pddl_problem.name = 'angry_birds_prob'
        pddl_problem.metric = self.metric
        pddl_problem.objects = []
        pddl_problem.init = []
        pddl_problem.goal = []

        #we should probably use the self.sling on the object
        slingshot = self.get_slingshot(sb_state)

        # A dictionary with global problem parameters
        problem_params = dict()
        problem_params["has_platform"]=False
        problem_params["has_block"]=False
        problem_params["bird_index"]=0
        problem_params["slingshot"]=slingshot
        problem_params["groundOffset"] = self.get_ground_offset(slingshot)
        problem_params["gravity"] = round(0.48* self.constant_numeric_fluents['gravity_factor'] /
                                          self.hyper_parameters['scale_factor'] * get_scale(slingshot))
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
            type.add_object_to_problem(pddl_problem, obj, problem_params)

        # Add dummy platform and block if none exists TODO: Why do we need this?
        if problem_params["has_platform"]==False:
            pddl_problem.objects.append(['dummy_platform','platform'])
        if problem_params["has_block"]==False:
            pddl_problem.objects.append(['dummy_block','block'])

        # Add constants fluents
        for numeric_fluent in self.constant_numeric_fluents:
            pddl_problem.init.append(['=', [numeric_fluent], self.constant_numeric_fluents[numeric_fluent]])
        for boolean_fluent in self.constant_boolean_fluents:
            if self.constant_boolean_fluents[boolean_fluent]:
                pddl_problem.init.append([boolean_fluent])
            else:
                pddl_problem.init.append(['not',[boolean_fluent]])
        pddl_problem.init.append(['=', ['gravity'], problem_params["gravity"]])


        # Initial angle value to prune un-promising trajectories which only hit the ground
        closest_obj_x, closest_obj_y = get_closest_object_xy(pddl_problem)
        min_angle, max_angle = estimate_launch_angle(slingshot, Point2D(closest_obj_x, closest_obj_y), self)
        problem_params["angle"] = min_angle
        pddl_problem.init.append(['=', ['angle'], problem_params["angle"]])
        problem_params["max_angle"] = max_angle
        pddl_problem.init.append(['=', ['max_angle'], problem_params["max_angle"]])


        # Add goal
        pigs = problem_params["pigs"]
        assert len(pigs)>0
        for pig in pigs:
            pddl_problem.goal.append(['pig_dead', pig])

        return pddl_problem

    ''' Get the ground offset '''
    def get_ground_offset(self, slingshot):
        return slingshot[1]['bbox'].bounds[3]

    ''' Translate the given observed SBState to a PddlPlusState object. 
    This is designed to handle intermediate state observed during execution '''
    def create_pddl_state(self, sb_state: ProcessedSBState):
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