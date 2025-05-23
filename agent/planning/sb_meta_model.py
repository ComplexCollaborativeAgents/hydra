import copy
import math

from agent.perception.perception import ProcessedSBState
from agent.planning.pddlplus_parser import *
import settings
import logging
import random

from utils.point2D import Point2D
import worlds.science_birds as SB
from agent.perception.perception import Perception

from agent.planning.meta_model import *

logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sb_meta_model")
logger.setLevel(logging.INFO)

""" Utility functions """


def get_x_coordinate(obj):
    return round(abs(obj[1]['polygon'].bounds[2] + obj[1]['polygon'].bounds[0]) / 2)


def get_y_coordinate(obj, ground_offset):
    return abs(round(abs(obj[1]['polygon'].bounds[1] + obj[1]['polygon'].bounds[3]) / 2) - ground_offset)


# TODO: See how to merge the two functions below with the two above
def get_slingshot_x(slingshot):
    return round(
        (slingshot[1]['polygon'].bounds[0] + slingshot[1]['polygon'].bounds[2]) / 2) - 0  # TODO: Why this minus zero


def get_slingshot_y(ground_offset, slingshot):
    return round(abs(((slingshot[1]['polygon'].bounds[1] + slingshot[1]['polygon'].bounds[3]) / 2) - ground_offset) - 0)


def get_scale(slingshot):
    # sling.width + sling.height
    return abs(round(slingshot[1]['polygon'].bounds[0] - slingshot[1]['polygon'].bounds[2])) + round(
        abs(slingshot[1]['polygon'].bounds[1] - slingshot[1]['polygon'].bounds[3]))


def get_radius(obj):
    return round((abs(obj[1]['polygon'].bounds[2] - obj[1]['polygon'].bounds[0]) / 2) * 0.9)


def get_height(obj):
    return abs(obj[1]['polygon'].bounds[3] - obj[1]['polygon'].bounds[1])


def get_width(obj):
    return abs(obj[1]['polygon'].bounds[2] - obj[1]['polygon'].bounds[0])


def get_random_pig_xy(pddl_problem: PddlPlusProblem):
    """ Returns the location of the closest object to aim for """
    state = PddlPlusState(pddl_problem.init)
    target_pigs = list(state.get_objects('pig'))

    pig = random.choice(target_pigs)
    random_pig_x = state[('x_pig', pig)]
    random_pig_y = state[('y_pig', pig)]

    return random_pig_x, random_pig_y


def get_closest_object_xy(pddl_problem: PddlPlusProblem):
    """ Returns the location of the closest object to aim for """
    state = PddlPlusState(pddl_problem.init)
    target_pigs = state.get_objects('pig')
    closest_obj_x = None
    closest_obj_y = None
    # assert len(target_pigs) > 0
    for pig in target_pigs:
        x_pig = state[('x_pig', pig)]
        y_pig = state[('y_pig', pig)]
        if (closest_obj_x is None) or (
                math.sqrt(x_pig ** 2 + y_pig ** 2) < math.sqrt(closest_obj_x ** 2 + closest_obj_y ** 2)):
            closest_obj_x = x_pig
            closest_obj_y = y_pig
            # print("\n\nNEW CLOSEST TARGET: " + pig + "(" + str(closest_obj_x) + ", " + str(closest_obj_y) + ")\n")
    for plat in state.get_objects('platform'):
        x_plat = state[('x_platform', plat)]
        y_plat = state[('y_platform', plat)]
        if (y_plat - ((state[('y_height', plat)]) / 2)) >= 10.0:
            y_plat = 0
        if (closest_obj_x is None) or (
                math.sqrt(x_plat ** 2 + y_plat ** 2) < math.sqrt(closest_obj_x ** 2 + closest_obj_y ** 2)):
            closest_obj_x = x_plat
            closest_obj_y = y_plat
            # print("\n\nNEW CLOSEST TARGET: " + plat + "(" + str(closest_obj_x) + ", " + str(closest_obj_y) + ")\n")
    for blo in state.get_objects('block'):
        x_blo = state[('x_block', blo)]
        y_blo = state[('y_block', blo)]
        if (closest_obj_x is None) or (
                math.sqrt(x_blo ** 2 + y_blo ** 2) < math.sqrt(closest_obj_x ** 2 + closest_obj_y ** 2)):
            closest_obj_x = x_blo
            closest_obj_y = y_blo
            # print("\n\nNEW CLOSEST TARGET: " + blo + "(" + str(closest_obj_x) + ", " + str(closest_obj_y) + ")\n")
    return closest_obj_x, closest_obj_y


def estimate_launch_angle(slingshot, targetPoint, meta_model):
    """ Returns a min/max launch angle to hit the given target point """
    # calculate relative position of the target (normalised)
    scale_factor = meta_model.hyper_parameters['scale_factor']
    _velocity = 9.5 / scale_factor
    x_offset = 0.0
    y_offset = 0.0

    default_min_angle = 0.0
    default_max_angle = 90.0

    try:

        ground_offset = slingshot[1]['polygon'].bounds[3]

        scale = get_scale(slingshot)

        # ref = Point2D(int(slingshot.X + X_OFFSET * slingshot.width), int(slingshot.Y + Y_OFFSET * slingshot.height))
        ref = Point2D(int(get_slingshot_x(slingshot) + x_offset * get_width(slingshot)),
                      int(get_slingshot_y(ground_offset, slingshot) + y_offset * get_height(slingshot)))
        x = (targetPoint.X - ref.X)
        y = (targetPoint.Y - ref.Y)


        # gravity
        g = 0.48 * meta_model.constant_numeric_fluents['gravity_factor'] / scale_factor * scale
        # launch speed
        v = 182  # _velocity * scale

        solution_existence_factor = v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2

        # the target point cannot be reached
        if solution_existence_factor < 0:
            logger.info('estimate launch angle: NO SOLUTION!')
            return 0.0, 90.0

        # solve cos theta from projectile equation
        cos_theta_1 = math.sqrt(
            (x ** 2 * v ** 2 - x ** 2 * y * g + x ** 2 * math.sqrt(solution_existence_factor)) /
            (2 * v ** 2 * (x ** 2 + y ** 2)))
        cos_theta_2 = math.sqrt(
            (x ** 2 * v ** 2 - x ** 2 * y * g - x ** 2 * math.sqrt(solution_existence_factor)) /
            (2 * v ** 2 * (x ** 2 + y ** 2)))

        theta_1 = math.acos(cos_theta_1)
        theta_2 = math.acos(cos_theta_2)
        print(f'found angle to pig: {math.degrees(theta_1)}, {math.degrees(theta_2)}')
        return math.degrees(theta_1), math.degrees(theta_2)

    except:
        print('No trajectory to pig!')
        return default_min_angle, default_max_angle


class SBPddlObjectType(PddlObjectType):
    def _get_name(self, obj):
        obj_type = obj[1]['type']
        return '{}_{}'.format(obj_type, obj[0])


# The slingshot is currently not directly modeled in our model, so its object is currently ignored.
# TODO: Reconsider this design choice.
class SlingshotType:
    """ Populate a PDDL+ problem with details about this object """

    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params: dict):
        return  # Do nothing, slingshot is currently not directly modeled as an object

    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params: dict):
        return  # Do nothing, slingshot is currently not directly modeled as an object


class PigType(SBPddlObjectType):
    def __init__(self, life_multiplier=1.0):
        super(PigType, self).__init__()
        self.pddl_type = "pig"
        self.hyper_parameters["m_pig"] = 1
        self.hyper_parameters["pddl_type"] = "pig"
        self.hyper_parameters["pig_life_multiplier"] = life_multiplier

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)
        obj_attributes["m_pig"] = self.hyper_parameters["m_pig"]

        obj_attributes["pig_life"] = self.__compute_pig_life()
        return obj_attributes

    def __compute_pig_life(self):
        return str(math.ceil(10 + self.hyper_parameters["pig_life_multiplier"]))

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = dict()
        obj_attributes["x_pig"] = get_x_coordinate(obj)
        obj_attributes["y_pig"] = get_y_coordinate(obj, problem_params["groundOffset"])
        obj_attributes["pig_radius"] = get_radius(obj)
        obj_attributes["pig_dead"] = False

        problem_params["pigs"].add(self._get_name(obj))

        return obj_attributes


class AgentType(SBPddlObjectType):
    def __init__(self):
        super(AgentType, self).__init__()
        self.pddl_type = "external_agent"
        self.hyper_parameters["agent_types"] = {'magician': 0, 'wizard': 1, 'butterfly': 2, 'worm': 3, 'unknown': 4}
        self.hyper_parameters["agent_velocities"] = {'magician': [10.4, 0], 'wizard': [10.4, 10.4], 'butterfly': [0, 0],
                                                     'worm': [10.4, 0], 'unknown': [0, 0]}
        self.hyper_parameters["vx_agent"] = 0
        self.hyper_parameters["vy_agent"] = 0
        self.hyper_parameters["timing_agent"] = 0
        self.hyper_parameters["agent_dead"] = False
        self.hyper_parameters["pddl_type"] = "external_agent"
        self.hyper_parameters["x_max_border"] = 800
        self.hyper_parameters["x_min_border"] = 0
        self.hyper_parameters["y_max_border"] = 600
        self.hyper_parameters["y_min_border"] = -10

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)
        obj_attributes["timing_agent"] = self.hyper_parameters["timing_agent"]
        obj_attributes["agent_dead"] = self.hyper_parameters["agent_dead"]
        # obj_attributes[""] = self.hyper_parameters[""]

        return obj_attributes

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        problem_params["has_external_agent"] = True
        obj_attributes = dict()
        obj_attributes["agent_type"] = self.hyper_parameters["agent_types"][obj[1]['type'].lower()]
        obj_attributes["x_agent"] = get_x_coordinate(obj)
        obj_attributes["y_agent"] = get_y_coordinate(obj, problem_params["groundOffset"])
        obj_attributes["vx_agent"] = self.hyper_parameters["agent_velocities"][obj[1]['type'].lower()][0]
        obj_attributes["vy_agent"] = self.hyper_parameters["agent_velocities"][obj[1]['type'].lower()][1]
        obj_attributes["agent_height"] = get_height(obj)
        obj_attributes["agent_width"] = get_width(obj)

        obj_attributes["x_min_border"] = get_slingshot_x(problem_params["slingshot"])
        obj_attributes["x_max_border"] = self.hyper_parameters["x_max_border"]
        obj_attributes["y_min_border"] = self.hyper_parameters["y_min_border"]
        obj_attributes["y_min_border"] = self.hyper_parameters["y_max_border"]

        problem_params["external_agents"].add(self._get_name(obj))

        return obj_attributes


class BirdType(SBPddlObjectType):
    def __init__(self):
        super(BirdType, self).__init__()
        self.pddl_type = "bird"
        # self.hyper_parameters["v_bird"] = 270
        self.hyper_parameters["vx_bird"] = 0
        self.hyper_parameters["vy_bird"] = 0
        self.hyper_parameters["m_bird"] = {'red': 1, 'yellow': 0.75, 'black': 1.5, 'white': 0.4,
                                           'blue': 0.1}  # {"red"= 0, "yellow"= 1, "black"= 2, "white"= 3, "blue"= 4}
        self.hyper_parameters["bounce_count"] = 0
        self.hyper_parameters["bird_released"] = False
        ## TAP UPDATE
        self.hyper_parameters["bird_tapped"] = False
        self.hyper_parameters["velocity_change"] = 0

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)

        # obj_attributes["v_bird"] = self.hyper_parameters["v_bird"]
        obj_attributes["vx_bird"] = self.hyper_parameters["vx_bird"]
        obj_attributes["vy_bird"] = self.hyper_parameters["vy_bird"]
        # obj_attributes["m_bird"] = self.hyper_parameters["m_bird"]
        obj_attributes["bounce_count"] = self.hyper_parameters["bounce_count"]
        obj_attributes["bird_released"] = self.hyper_parameters["bird_released"]
        obj_attributes["bird_radius"] = get_radius(obj)

        return obj_attributes

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = dict()

        slingshot = problem_params["slingshot"]
        ground_offset = problem_params["groundOffset"]

        # In the initial state, the planner assumes the birds are on the sling
        slingshot_x = get_slingshot_x(slingshot)
        slingshot_y = get_slingshot_y(ground_offset, slingshot)

        if "initial_state" in problem_params and problem_params["initial_state"] == True:
            obj_attributes["x_bird"] = slingshot_x
            obj_attributes["y_bird"] = slingshot_y
            obj_attributes[
                "v_bird"] = 182 + self.hyper_parameters["velocity_change"]
            # round((9.5 / 2.7) * (get_scale(slingshot)) * (self.hyper_parameters["velocity_change"]/10) )
        else:
            obj_attributes["x_bird"] = get_x_coordinate(obj)
            obj_attributes["y_bird"] = get_y_coordinate(obj, ground_offset)

            # Need to separate the case where we're before shooting the bird and after.
            # Before: the bird location is considered as the location of the slingshot,
            # afterwards, it's the location of the birds bounding box TODO: Replace this hack
            if obj_attributes["x_bird"] <= slingshot_x:
                obj_attributes["x_bird"] = slingshot_x
                obj_attributes["y_bird"] = slingshot_y

        # TAP UPDATE
        # Encode bird type/colour in the PDDL+ model
        obj_attributes["bird_type"] = 0
        bird_map = {"red": 0, "yellow": 1, "black": 2, "white": 3, "blue": 4}
        for key in bird_map:
            if key in self._get_name(obj).lower():
                obj_attributes["bird_type"] = bird_map[key]
        # Encode bird type/colour in the PDDL+ model
        for key in bird_map:
            if key in self._get_name(obj).lower():
                obj_attributes["bird_type"] = bird_map[key]
                obj_attributes["m_bird"] = self.hyper_parameters["m_bird"][key]

        obj_attributes["bird_id"] = problem_params["bird_index"]
        problem_params["bird_index"] = problem_params["bird_index"] + 1

        problem_params["birds"].add(self._get_name(obj))

        return obj_attributes


class PlatformType(SBPddlObjectType):
    def __init__(self):
        super(PlatformType, self).__init__()
        self.pddl_type = "platform"

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        problem_params["has_platform"] = True

        obj_attributes = dict()

        obj_attributes["x_platform"] = get_x_coordinate(obj)
        obj_attributes["y_platform"] = get_y_coordinate(obj, problem_params["groundOffset"])

        obj_attributes["platform_height"] = get_height(obj)  # Previous versions had a 10% dialation
        obj_attributes["platform_width"] = get_width(obj)
        return obj_attributes

    def _compute_obj_attributes(self, obj, problem_params: dict):
        return self._compute_observable_obj_attributes(obj, problem_params)


class BlockType(SBPddlObjectType):
    def __init__(self, block_life_multiplier=1.0, block_mass_coeff=1.0):
        super(BlockType, self).__init__()
        self.pddl_type = "block"

        self.hyper_parameters["block_life_multiplier"] = block_life_multiplier
        self.hyper_parameters["block_mass_coeff"] = block_mass_coeff

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        problem_params["has_block"] = True

        obj_attributes = dict()

        ground_offset = problem_params["groundOffset"]
        obj_attributes["x_block"] = get_x_coordinate(obj)
        obj_attributes["y_block"] = get_y_coordinate(obj, ground_offset)
        obj_attributes["block_height"] = get_height(obj)
        obj_attributes["block_width"] = get_width(obj)
        return obj_attributes

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = self._compute_observable_obj_attributes(obj, problem_params)

        ground_offset = problem_params["groundOffset"]
        obj_attributes["block_life"] = self.__compute_block_life()
        obj_attributes["block_mass"] = self.__compute_block_mass()
        obj_attributes["block_stability"] = self.__compute_stability(obj_attributes["block_width"],
                                                                     obj_attributes["block_height"],
                                                                     obj_attributes["y_block"],
                                                                     ground_offset)
        return obj_attributes

    def __compute_block_life(self):
        return str(math.ceil(265 * self.hyper_parameters["block_life_multiplier"]))

    def __compute_block_mass(self):
        return str(self.hyper_parameters["block_mass_coeff"])

    def __compute_stability(self, bl_width, bl_height, bl_y, ground_offset):
        return round(265 * (bl_width / (bl_height + 1)) \
                     * (1 - (bl_y / (ground_offset + 1))) \
                     * self.hyper_parameters["block_mass_coeff"])


class WoodType(BlockType):
    def __init__(self):
        super(WoodType, self).__init__(1.0, 0.375 * 1.3)


class IceType(BlockType):
    def __init__(self):
        super(IceType, self).__init__(0.5, 0.125 * 2)


class StoneType(BlockType):
    def __init__(self):
        super(StoneType, self).__init__(2.0, 1.2)


class TNTType(BlockType):
    def __init__(self):
        super(TNTType, self).__init__(0.001, 0.1)

    def _compute_obj_attributes(self, obj, problem_params: dict):
        obj_attributes = super(TNTType, self)._compute_obj_attributes(obj, problem_params)
        obj_attributes["block_explosive"] = True
        return obj_attributes


""" Assumption: unknown types are assumed to be blocks """


class UnknownType(BlockType):
    def __init__(self):
        super().__init__()


class ScienceBirdsMetaModel(MetaModel):
    TWANG_ACTION = "pa-twang"  # Default action type

    """ Sets the default meta-model"""

    def __init__(self):
        super().__init__(docker_path=settings.SB_PLANNING_DOCKER_PATH,
                         domain_file_name="sb_domain.pddl",
                         delta_t=settings.SB_DELTA_T,
                         metric='minimize(total-time)',
                         repairable_constants=[
                             # 'meta_wood_multiplier',
                             # 'meta_stone_multiplier',
                             # 'meta_ice_multiplier',
                             'v_bird_change',
                             # 'meta_platform_size',
                             'base_life_pig_multiplier',
                             # 'explosion_damage',
                             # 'fall_damage',
                             'gravity_factor'

                         ],
                         repair_deltas=[
                             1, 50, 0.1  # 50, 10
                         ],
                         constant_numeric_fluents={
                             'active_bird': 0,
                             'angle_rate': 20,
                             'ground_damper': 0.3,
                             'base_life_wood_multiplier': 0.75,
                             'base_life_ice_multiplier': 0.4,
                             'base_life_stone_multiplier': 1.25,
                             'base_life_tnt_multiplier': 0.001,
                             'base_mass_wood_multiplier': 0.375,
                             'base_mass_ice_multiplier': 0.188,
                             'base_mass_stone_multiplier': 1,
                             'base_mass_tnt_multiplier': 0.1,
                             'meta_wood_multiplier': 1.0,
                             'meta_stone_multiplier': 1.0,
                             'meta_ice_multiplier': 1.0,
                             'v_bird_change': 0.0,
                             'gravity_factor': 9.81,
                             'meta_platform_size': 2,
                             'base_life_pig_multiplier': 0.0,
                             'fall_damage': 50,
                             'explosion_damage': 100,
                             'redBird_ice_damage_factor': 1.5,
                             'redBird_wood_damage_factor': 0.5,
                             'redBird_stone_damage_factor': 0.3,
                             'yellowBird_ice_damage_factor': 0.4,
                             'yellowBird_wood_damage_factor': 2,
                             'yellowBird_stone_damage_factor': 0.5,
                             'blueBird_ice_damage_factor': 2,
                             'blueBird_wood_damage_factor': 0.5,
                             'blueBird_stone_damage_factor': 0.1,
                             'blackBird_ice_damage_factor': 1,
                             'blackBird_wood_damage_factor': 1,
                             'blackBird_stone_damage_factor': 0.2,
                             'birdWhite_ice_damage_factor': 0.5,
                             'birdWhite_wood_damage_factor': 0.5,
                             'birdWhite_stone_damage_factor': 0.5,
                         },
                         constant_boolean_fluents={
                             'angle_adjusted': False,
                             'pig_killed': False})

        self.hyper_parameters['scale_factor'] = 2.7

        # Mapping of type to Pddl object. All objects of this type will be clones of this pddl object
        self.object_types = dict()
        self.object_types["pig"] = PigType()
        self.object_types["bird"] = BirdType()
        self.object_types["block"] = BlockType()
        self.object_types["wood"] = WoodType()
        self.object_types["ice"] = IceType()
        self.object_types["stone"] = StoneType()
        self.object_types["TNT"] = TNTType()
        self.object_types["magician"] = AgentType()
        self.object_types["wizard"] = AgentType()
        self.object_types["butterfly"] = AgentType()
        self.object_types["worm"] = AgentType()

        # self.object_types["wood"].hyper_parameters["block_life"] = self.constant_numeric_fluents["base_life_wood_multiplier"] * 265
        # self.object_types["stone"].hyper_parameters["block_life"] = self.constant_numeric_fluents["base_life_stone_multiplier"] * 265
        # self.object_types["ice"].hyper_parameters["block_life"] = self.constant_numeric_fluents["base_life_ice_multiplier"] * 265
        # self.object_types["TNT"].hyper_parameters["block_life"] = self.constant_numeric_fluents["base_life_tnt_multiplier"] * 265

        self.object_types["platform"] = PlatformType()
        self.object_types["slingshot"] = SlingshotType()
        self.object_types["unknown"] = UnknownType()

    def get_slingshot(self, sb_state: ProcessedSBState):
        """ Get the slingshot object """
        sling = None
        for o in sb_state.objects.items():
            if o[1]['type'] == 'slingshot':
                sling = o
        return sling

    @staticmethod
    def action_time_to_angle(action_time: float, state: PddlPlusState):
        """ Converts twang angle to the corresponding action time """
        return action_time * float(state[('angle_rate',)]) + float(state[('angle',)])

    @staticmethod
    def angle_to_action_time(angle: float, state: PddlPlusState):
        """ Converts a twang angle to the time of the twang action. """
        return (angle - float(state[('angle',)])) / float(state[('angle_rate',)])

    def create_sb_action(self, timed_action: TimedAction, processed_state: ProcessedSBState, tap_timing=3000):
        """ Creates an SB action from a PDDL action_angle_time triple outputted by the planner """
        # Compute angle from action time
        pddl_state = self.create_pddl_problem(processed_state).get_init_state()
        angle = self.action_time_to_angle(timed_action.start_at, pddl_state)

        # Convert angle to release point
        tp = SB.ScienceBirds.trajectory_planner
        ref_point = tp.get_reference_point(processed_state.sling)
        release_point_from_plan = tp.find_release_point(processed_state.sling, math.radians(angle))
        action = SB.SBShoot(release_point_from_plan.X, release_point_from_plan.Y,
                            tap_timing, ref_point.X, ref_point.Y, timed_action)
        return action

    def create_timed_action(self, sb_shoot: SB.SBShoot, sb_state: ProcessedSBState):
        """ Create a PDDL+ TimedAction object from an SB action and state """
        #
        # mag = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip((0, 0), (sb_shoot.dx, sb_shoot.dy))))
        # theta_from_x = math.acos(sb_shoot.dx / (-mag))
        # theta_from_y = math.asin(sb_shoot.dy / mag)
        # assert abs(theta_from_x - theta_from_y) < 1  # Precision TODO: Discuss this, the number 1 is arbitrary
        # action_angle = math.degrees(theta_from_x)
        #
        # pddl_state = PddlPlusState(self.create_pddl_problem(sb_state).init)
        # action_time = self.angle_to_action_time(action_angle, pddl_state)
        #
        # action_time = settings.SB_DELTA_T * round(action_time / settings.SB_DELTA_T)
        #
        # active_bird = pddl_state.get_active_bird()
        # action_name = "%s %s" % (ScienceBirdsMetaModel.TWANG_ACTION, active_bird)
        #
        # return TimedAction(action_name, action_time)
        return sb_shoot.timed_action

    """ Translate the initial SBState, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. """

    def create_pddl_problem(self, sb_state: ProcessedSBState):
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

        # we should probably use the self.sling on the object
        slingshot = self.get_slingshot(sb_state)

        # A dictionary with global problem parameters
        problem_params = dict()
        problem_params["has_platform"] = False
        problem_params["has_block"] = False
        problem_params["has_external_agent"] = False
        problem_params["bird_index"] = 0
        problem_params["slingshot"] = slingshot
        problem_params["groundOffset"] = self.get_ground_offset(slingshot)
        problem_params["gravity"] = round(0.48 * self.constant_numeric_fluents['gravity_factor'] /
                                          self.hyper_parameters['scale_factor'] * get_scale(slingshot))
        # Above line redundant since we're storing the slingshot also, but it seems easier to store it also to save computations of the offset everytime we use it.
        problem_params["meta_platform_size"] = self.constant_numeric_fluents['meta_platform_size']
        problem_params["base_life_pig_multiplier"] = self.constant_numeric_fluents['base_life_pig_multiplier']
        problem_params["fall_damage"] = self.constant_numeric_fluents['fall_damage']
        problem_params["explosion_damage"] = self.constant_numeric_fluents['explosion_damage']

        problem_params["pigs"] = set()
        problem_params["birds"] = set()
        problem_params["external_agents"] = set()
        problem_params[
            "initial_state"] = True  # This marks that SBState describes the initial state. Used for setting the bird's location in the slingshot's location. TODO: Reconsider this design choice

        self.object_types["wood"].hyper_parameters["block_life_multiplier"] = self.constant_numeric_fluents[
                                                                                  "base_life_wood_multiplier"] * \
                                                                              self.constant_numeric_fluents[
                                                                                  "meta_wood_multiplier"]
        self.object_types["stone"].hyper_parameters["block_life_multiplier"] = self.constant_numeric_fluents[
                                                                                   "base_life_stone_multiplier"] * \
                                                                               self.constant_numeric_fluents[
                                                                                   "meta_stone_multiplier"]
        self.object_types["ice"].hyper_parameters["block_life_multiplier"] = self.constant_numeric_fluents[
                                                                                 "base_life_ice_multiplier"] * \
                                                                             self.constant_numeric_fluents[
                                                                                 "meta_ice_multiplier"]
        self.object_types["TNT"].hyper_parameters["block_life_multiplier"] = self.constant_numeric_fluents[
            "base_life_tnt_multiplier"]

        self.object_types["wood"].hyper_parameters["block_mass_coeff"] = self.constant_numeric_fluents[
                                                                             "base_mass_wood_multiplier"] * \
                                                                         self.constant_numeric_fluents[
                                                                             "meta_wood_multiplier"]
        self.object_types["stone"].hyper_parameters["block_mass_coeff"] = self.constant_numeric_fluents[
                                                                              "base_mass_stone_multiplier"] * \
                                                                          self.constant_numeric_fluents[
                                                                              "meta_stone_multiplier"]
        self.object_types["ice"].hyper_parameters["block_mass_coeff"] = self.constant_numeric_fluents[
                                                                            "base_mass_ice_multiplier"] * \
                                                                        self.constant_numeric_fluents[
                                                                            "meta_ice_multiplier"]
        self.object_types["TNT"].hyper_parameters["block_mass_coeff"] = self.constant_numeric_fluents[
            "base_mass_tnt_multiplier"]

        self.object_types["pig"].hyper_parameters["pig_life_multiplier"] = self.constant_numeric_fluents[
            "base_life_pig_multiplier"]

        self.object_types["bird"].hyper_parameters["velocity_change"] = self.constant_numeric_fluents[
            "v_bird_change"]

        logger.debug("\n\n")
        ex_agent_types = {'magician', 'wizard', 'butterfly', 'worm'}
        x_max_blocks = 0
        x_min_blocks = 800
        y_min_blocks = 600
        y_max_blocks = 0

        just_in_case_platforms = []
        for objj in sb_state.objects.items():
            if 'bird' not in objj[1]['type'] and (
                    get_x_coordinate(objj) > get_slingshot_x(slingshot)) and not 'slingshot' in objj[1]['type']:
                if objj[1]['type'] not in ex_agent_types:
                    x_max_blocks = max(get_x_coordinate(objj), x_max_blocks)
                    x_min_blocks = min(get_x_coordinate(objj), x_min_blocks)
                    y_max_blocks = max(get_y_coordinate(objj, problem_params["groundOffset"]), y_max_blocks)
                    y_min_blocks = min(get_y_coordinate(objj, problem_params["groundOffset"]), y_min_blocks)
                if objj[1]['type'] == 'platform':
                    # while iterating through all objects store platforms so they can be used to define borders for the worm agent (in the subsequent loop through level objects)
                    just_in_case_platforms.append(objj)

        # Add objects to problem
        for obj in sb_state.objects.items():
            # Get type
            type_str = obj[1]['type']

            if 'bird' in type_str.lower() or (
                    get_x_coordinate(obj) <= get_slingshot_x(slingshot) and not 'slingshot' in type_str):
                obj_type = self.object_types["bird"]
                for block in sb_state.objects.items():
                    if block[1]['type'] in ['ice', 'wood', 'stone']:
                        block_str = block[1]['type'] + '_' + block[0]
                        bird_str = type_str + '_' + obj[0]
                        fluent_string = type_str + '_' + block[1]['type'] + '_damage_factor'
                        if self.constant_numeric_fluents.get(fluent_string) is None:
                            # Default assumed value is unknown bird = red, unknown block = wood
                            if type_str == 'unknown':
                                bird_type = 'redBird'
                            else:
                                bird_type = type_str
                            if block[1]['type'] == 'unknown':
                                block_type = 'wood'
                            else:
                                block_type = block[1]['type']
                            default_factor = bird_type + '_' + block_type + '_damage_factor'
                            if not self.constant_numeric_fluents.get(default_factor):
                                # This is some edge perception case like birds being identified as wood, e.g. 11:130
                                default_factor = 'redBird_wood_damage_factor'
                            self.constant_numeric_fluents[fluent_string] = self.constant_numeric_fluents[default_factor]
                            self.repairable_constants.append(fluent_string)
                        pddl_problem.init.append(['=',
                                                  ['bird_block_damage', bird_str, block_str],
                                                  self.constant_numeric_fluents[
                                                      type_str + '_' + block[1]['type'] + '_damage_factor']])
                        # TODO handle unknown bird and block types?
            else:
                if type_str in self.object_types:
                    obj_type = self.object_types[type_str]
                else:
                    logger.debug("Unknown object type: %s" % type_str)
                    # TODO Handle unknown objects in some way (Error? default object?)
                    continue

            if type_str in ex_agent_types:
                # print("(create pddl problem) Object_type: " + str(type_str) +"/" + str(type) + " [" + str(get_x_coordinate(obj)) + ", " + str(get_y_coordinate(obj, problem_params["groundOffset"])) + "] ")

                if type_str == 'magician':
                    obj_type.hyper_parameters["x_min_border"] = x_max_blocks if get_x_coordinate(
                        obj) >= x_max_blocks else get_slingshot_x(slingshot)
                    obj_type.hyper_parameters["x_max_border"] = x_min_blocks if get_x_coordinate(
                        obj) <= x_min_blocks else 800  # 800px=rightmost edge of window
                if type_str == 'wizard':
                    obj_type.hyper_parameters["x_min_border"] = get_slingshot_x(slingshot)
                    obj_type.hyper_parameters["x_max_border"] = 800  # 800px=rightmost edge of window
                    # obj_type.hyper_parameters["y_min_border"] = y_max_blocks
                    obj_type.hyper_parameters[
                        "y_min_border"] = 0  # TODO: the wizard can be at any height in the level. Currently, blocks are ignored, will update with exclusion zones soon
                    obj_type.hyper_parameters["y_max_border"] = 600  # 600px=top edge of window
                if type_str == 'butterfly':
                    obj_type.hyper_parameters["x_min_border"] = x_min_blocks * 0.75
                    obj_type.hyper_parameters["x_max_border"] = x_max_blocks * 1.25
                    obj_type.hyper_parameters["y_min_border"] = 0
                    obj_type.hyper_parameters["y_max_border"] = y_max_blocks * 1.25  # 600px=top edge of window
                if type_str == 'worm':
                    for pl in just_in_case_platforms:
                        if (abs(get_x_coordinate(obj) - get_x_coordinate(pl)) <= (get_width(pl) / 2)) and (
                                abs(get_y_coordinate(obj, problem_params["groundOffset"]) - get_y_coordinate(pl,
                                                                                                             problem_params[
                                                                                                                 "groundOffset"])) <= (
                                        get_height(pl) / 2)):
                            obj_type.hyper_parameters["x_min_border"] = get_x_coordinate(pl) - (get_width(pl) / 2)
                            obj_type.hyper_parameters["x_max_border"] = get_x_coordinate(pl) + (get_width(pl) / 2)
                            obj_type.hyper_parameters["y_min_border"] = get_y_coordinate(pl, problem_params[
                                "groundOffset"]) - (get_height(pl) / 2)
                            obj_type.hyper_parameters["y_max_border"] = get_y_coordinate(pl, problem_params[
                                "groundOffset"]) + (get_height(pl) / 2)
                            break

            # Add object of this type to the problem
            obj_type.add_object_to_problem(pddl_problem, obj, problem_params)

        # Add dummy platform and block if none exists TODO: Why do we need this?
        if not problem_params["has_platform"]:
            pddl_problem.objects.append(['dummy_platform', 'platform'])
        if not problem_params["has_block"]:
            pddl_problem.objects.append(['dummy_block', 'block'])
        if not problem_params["has_external_agent"]:
            pddl_problem.objects.append(['dummy_agent', 'external_agent'])
            pddl_problem.init.append(['agent_dead', 'dummy_agent'])

        # Add constants fluents
        for numeric_fluent in self.constant_numeric_fluents:
            pddl_problem.init.append(['=', [numeric_fluent], self.constant_numeric_fluents[numeric_fluent]])
        for boolean_fluent in self.constant_boolean_fluents:
            if self.constant_boolean_fluents[boolean_fluent]:
                pddl_problem.init.append([boolean_fluent])
            else:
                pddl_problem.init.append(['not', [boolean_fluent]])
        pddl_problem.init.append(['=', ['gravity'], problem_params["gravity"]])

        # Initial angle value to prune un-promising trajectories which only hit the ground
        # closest_obj_x, closest_obj_y = get_closest_object_xy(pddl_problem)
        # min_angle, max_angle = estimate_launch_angle(slingshot, Point2D(closest_obj_x, closest_obj_y), self)
        problem_params["angle"] = 0.0
        pddl_problem.init.append(['=', ['angle'], problem_params["angle"]])
        problem_params["max_angle"] = 81.5  # max_angle # Above angles of 81.5 the bird goes backwards.
        pddl_problem.init.append(['=', ['max_angle'], problem_params["max_angle"]])

        problem_params["points_score"] = len(problem_params["birds"])
        pddl_problem.init.append(['=', ['points_score'], problem_params["points_score"]])

        # Add goal
        pigs = problem_params["pigs"]
        if len(pigs) == 0:
            logger.info("No pigs found, taking default shot.")
            pddl_problem.goal.append(['pig_dead'])
        else:
            for pig in pigs:
                pddl_problem.goal.append(['pig_dead', pig])
        return pddl_problem

    def get_ground_offset(self, slingshot):
        """ Get the ground offset """
        return slingshot[1]['polygon'].bounds[3]

    """ Translate the given observed SBState to a PddlPlusState object. 
    This is designed to handle intermediate state observed during execution """

    def create_pddl_state(self, sb_state: ProcessedSBState):
        pddl_state = PddlPlusState()

        # we should probably use the self.sling on the object
        slingshot = self.get_slingshot(sb_state)

        # A dictionary with global problem parameters
        state_params = dict()
        state_params["has_platform"] = False
        state_params["has_block"] = False
        state_params["has_external_agent"] = False
        state_params["bird_index"] = 0
        state_params["slingshot"] = slingshot
        state_params["groundOffset"] = self.get_ground_offset(slingshot)
        # Above line redundant since we're storing the slingshot also, but it seems easier to store it also to save computations of the offset everytime we use it.
        state_params["pigs"] = set()
        state_params["birds"] = set()
        state_params["external_agents"] = set()
        state_params[
            "initial_state"] = False  # This marks that SBState describes the initial state. Used for setting the bird's location in the slingshot's location. TODO: Reconsider this design choice

        # Add objects to problem
        for obj in sb_state.objects.items():
            # Get type
            type_str = obj[1]['type']
            if 'bird' in type_str.lower() or (
                    get_x_coordinate(obj) <= get_slingshot_x(slingshot) and not 'slingshot' in type_str):
                obj_type = self.object_types["bird"]
            else:
                if type_str in self.object_types:
                    obj_type = self.object_types[type_str]
                else:
                    logger.info("Unknown object type: %s" % type_str)
                    # TODO Handle unknown objects in some way (Error? default object?)
                    continue
            # print("(create pddl state) Object_type: " + str(type_str) + "/" + str(type) + " [" + str(get_x_coordinate(obj)) + ", " + str(get_y_coordinate(obj, state_params["groundOffset"])) + "] ")

            # Add object of this type to the problem
            obj_type.add_object_to_state(pddl_state, obj, state_params)

        return pddl_state

    def create_simplified_problem(self, prob: PddlPlusProblem):
        """ Create a simplified version of the given problem, to help the planner plan
         TODO: Move this to planner"""
        prob_simplified = PddlPlusProblem()
        prob_simplified.name = copy.copy(prob.name)
        prob_simplified.domain = copy.copy(prob.domain)
        prob_simplified.objects = copy.copy(prob.objects)

        removed_list = []
        first_bird_spotted = False
        for obj in prob.objects:
            if (obj[1] == 'bird') and not first_bird_spotted:
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

    def create_super_simplified_problem(self, prob: PddlPlusProblem):
        """ Created an even more simplified version of the problem to speed up planner ?"""
        prob_super_simplified = PddlPlusProblem()
        prob_super_simplified.name = copy.copy(prob.name)
        prob_super_simplified.domain = copy.copy(prob.domain)
        prob_super_simplified.objects = copy.copy(prob.objects)

        removed_list = []
        super_removed_list = []
        first_bird_spotted = False
        for obj in prob.objects:
            if (obj[1] == 'bird') and not first_bird_spotted:
                first_bird_spotted = True
            elif obj[1] == 'bird':
                removed_list.append(obj[0])
                prob_super_simplified.objects.remove(obj)
            elif obj[1] == 'block':
                super_removed_list.append(obj[0])
                prob_super_simplified.objects.remove(obj)
            elif (obj[1] == 'magician') or (obj[1] == 'wizard') or (obj[1] == 'butterfly') or (obj[1] == 'worm'):
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
