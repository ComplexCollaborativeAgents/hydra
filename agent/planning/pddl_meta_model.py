from agent.planning.pddl_plus import *
import copy
import math


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
    return abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0])]

''' Objects for the meta model '''
class PddlObject():
    ''' Accepts an object from SBState.objects '''
    def __init__(self, obj):
        self.attributes = dict()
        self.type = obj[1]['type']
        self.name = '{}_{}'.format(self.obj_type, obj[0])


    def __getitem__(self, attribute_name):
        return self.attributes[attribute_name]

    ''' Populate a PDDL+ problem with details about this object '''
    def add_to_problem(self, prob: PddlPlusProblem):
        prob.objects.append((self.name, self.type))
        for attribute in self.attributes:
            value = self.attributes[attribute]
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value,  bool):
                if value==True:
                    prob.init.append([attribute, self.name])
                else: # value == False
                    prob.init.append(['not', [attribute, self.name]])
            else: # Attribute is a number
                prob.init.append(['=', [attribute, self.name], value])


class Pig(PddlObject):
    def __init__(self, obj, groundOffset):
        super(Pig, self).__init__(obj)
        self.groundOffset = groundOffset
        self.attributes["x_pig"] = get_x_coordinate(obj)
        self.attributes["y_pig"] = get_y_coordinate(obj, self.groundOffset)
        self.attributes["pig_radius"] = get_radius(obj)
        self.attributes["m_pig"] = 1
        self.attributes["pig_dead"] = False

    ''' Append relevant facts about this object to the PddlPlusProblem init state '''
    def add_to_problem(self, prob: PddlPlusProblem):
        super(Pig, self)

        # Pig specific code: goal is to kill pigs
        prob.goal.append(['pig_dead', self.name])

class Bird(PddlObject):
    def __init__(self, obj, slingshot, groundOffset, bird_index):
        super(Bird, self).__init__(obj)
        self.groundOffset = groundOffset
        self.attributes["x_bird"] = get_x_coordinate(slingshot)
        self.attributes["y_bird"] = get_y_coordinate(slingshot, self.groundOffset)
        self.attributes["bird_id"] = bird_index

        self.attributes["v_bird"] = 270
        self.attributes["vx_bird"] = 0
        self.attributes["vy_bird"] = 0
        self.attributes["m_bird"] = 1
        self.attributes["bound_count"] = -1
        self.attributes["bird_released"] = False


class Platform(PddlObject):
    def __init__(self, obj, groundOffset):
        super(Platform, self).__init__(obj)
        self.groundOffset = groundOffset
        self.attributes["x_platform"] = get_x_coordinate(obj)
        self.attributes["y_platform"] = get_y_coordinate(obj, self.groundOffset)

        self.attributes["platform_height"] = get_height(obj)
        self.attributes["platform_width"] = get_width(obj)

class Block(PddlObject):
    def __init__(self, obj, groundOffset, block_life_multiplier=1.0, block_mass_coeff = 1.0):
        super(Block, self).__init__(obj)
        raise ValueError("Object types mess!!! block is not a super type")

        self.groundOffset = groundOffset
        self.block_life_multiplier = block_life_multiplier
        self.block_mass_coeff = block_mass_coeff

        self.attributes["x_block"] = get_x_coordinate(obj)
        self.attributes["y_block"] = get_y_coordinate(obj, self.groundOffset)
        self.attributes["block_height"] = get_height(obj)
        self.attributes["block_width"] = get_width(obj)
        self.attributes["block_life"] = self.__compute_block_life()
        self.attributes["block_mass"] = self.__compute_block_mass()
        self.attributes["block_stability"] = self.__compute_stability()

    def __compute_block_life(self):
        return str(math.ceil(265 * self.block_life_multiplier))
    def __compute_block_mass(self):
        return str(self.block_mass_coeff)
    def __compute_stability(self):
        return 265 * (self.attributes["block_width"] / self.attributes["block_height"]) \
               * (1 - (self.attributes["y_block"] / self.groundOffset)) \
               * self.block_mass_coeff

    ''' Populate a PDDL+ problem with details about this object
     HACk to change the type of the object in PDDL to block, overriding to some extent the super.add_to_problem.
      This is because UPMurphey currently does not support type heirarchy. '''
    def add_to_problem(self, prob: PddlPlusProblem):
        super(Block,self).add_to_problem(prob)
        prob.objects.remove((self.name, self.type))
        prob.objects.append((self.name, "block"))


class Wood(Block):
    def __init__(self, obj, groundOffset):
        super(Wood, self).__init__(obj, groundOffset, 1.0, 0.375 * 1.3)

class Ice(Block):
    def __init__(self, obj, groundOffset):
        super(Ice, self).__init__(obj, groundOffset, 0.5, 0.125*2)

class Stone(Block):
    def __init__(self, obj, groundOffset):
        super(Ice, self).__init__(obj, groundOffset, 2.0, 1.2)


class TNT(Block):
    def __init__(self, obj, groundOffset):
        super(Ice, self).__init__(obj, groundOffset, 0.001, 1.2)

        self.attributes["block_explosive"] = True





class MetaModel():
    ''' Sets the default meta-model'''
    def __init__(self):
        # TODO: Read this from file instead of hard coding

        self.constant_numeric_facts = dict()
        self.constant_boolean_facts = dict()
        self.constant_numeric_facts['gravity']=134.2
        self.constant_numeric_facts['active_bird']=0
        self.constant_numeric_facts['angle']=0
        self.constant_boolean_facts['angle_adjusted']=False
        self.constant_boolean_facts['pig_killed']=False
        self.constant_numeric_facts['angle_rate'] = 10
        self.constant_numeric_facts['ground_damper'] = 0.4

        self.metric = 'minimize(total-time)'

        # Mapping of type to Pddl object. All objects of this type will be clones of this pddl object
        self.object_template = dict()
        self.object_template["pig"]=Pig()
        self.object_template["bird"]=Bird()
        self.object_template["block"]=Block()
        self.object_template["wood"]=Wood()
        self.object_template["ice"] = Ice()
        self.object_template["stone"] = Stone()
        self.object_template["TNT"] = TNT()
        self.object_template["platform"] = Platform()


    ''' Get the sling object '''
    def get_sling(self):
        sling = None
        for o in self.objects.items():
            if o[1]['type'] == 'slingshot':
                sling = o
        return sling

    ''' Translate the initial SBState, as observed, to a PddlPlusProblem object. 
    Note that in the initial state, we ignore the location of the bird and assume it is on the slingshot. '''
    def translate_sb_state_to_pddl_problem(self):
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
        slingshot = self.get_sling()

        groundOffset = slingshot[1]['bbox'].bounds[3]
        bird_index = 0

        platform = False
        block = False
        state_objects = list()
        for obj in self.objects.items():
            if obj[1]['type'] == 'pig':
                state_objects.append(Pig(obj, groundOffset))
            elif 'Bird' in obj[1]['type']:
                state_objects.append(Bird(obj, groundOffset,bird_index))
                bird_index += 1
            elif obj[1]['type'] == 'wood':
                state_objects.append(Wood(obj, groundOffset))
            elif obj[1]['type'] == 'ice':
                state_objects.append(Ice(obj, groundOffset))
            elif obj[1]['type'] == 'stone':
                state_objects.append(Stone(obj, groundOffset))
            elif obj[1]['type'] == 'TNT':
                state_objects.append(TNT(obj, groundOffset))
            elif obj[1]['type'] == 'unknown':
                state_objects.append(Block(obj,groundOffset))
            elif obj[1]['type'] == 'hill':
                state_objects.append(Platform(obj,groundOffset))
                platform = True
            elif obj[1]['type'] == 'slingshot':
                slingshot = obj
            else:
                logger.info("Unknown object type: %s" % obj[1]['type'])
            # TODO Handle unknown objects in some way (Error? default object? log?)

        # Add objects and their properties to the PDDL+ problem
        for obj in state_objects:
            obj.add_to_problem(prob)
            if isinstance(obj,Block):
                block=True
            if isinstance(obj, Platform):
                platfor=True
        if not platform:
            prob.objects.append(['dummy_platform','platform'])
        if not block:
            prob.objects.append(['dummy_block','block'])

        # Add constants
        for numeric_constant in self.constant_numeric_facts:
            prob.init.append(['=',[numeric_constant], self.constant_numeric_facts[numeric_constant]])
        for boolean_constant in self.constant_boolean_facts:
            if self.constant_boolean_facts[boolean_constant]:
                prob.init.append([boolean_constant])
            else:
                prob.init.append(['not',[boolean_constant]])


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

