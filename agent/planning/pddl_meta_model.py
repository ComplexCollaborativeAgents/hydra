from agent.planning.pddl_plus import *
import copy
import math

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
        self.defaults = dict()
        self.defaults["metric"] = 'minimize(total-time)'

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
        prob.metric = self.default["metric"]
        prob.objects = []
        prob.init = []
        prob.goal = []

        #we should probably use the self.sling on the object
        slingshot = self.get_sling()

        groundOffset = slingshot[1]['bbox'].bounds[3]
        bird_index = 0

        platform = False
        block = False
        for obj in self.objects.items():
            if obj[1]['type'] == 'pig':
                self.add_pig(obj, prob, groundOffset)
            elif 'Bird' in obj[1]['type']:
                self.add_bird(obj, prob, bird_index, groundOffset, slingshot)
                bird_index += 1
            elif obj[1]['type'] == 'wood' or obj[1]['type'] == 'ice' or obj[1]['type'] == 'stone':
                self.add_block(obj, prob, block, groundOffset)
                block = True
            elif obj[1]['type'] == 'hill':
                self.add_platform(obj, prob, groundOffset)
                platform = True
            elif obj[1]['type'] == 'slingshot':
                slingshot = obj

        self.add_constants(prob)

        if not platform:
            prob.objects.append(['dummy_platform','platform'])
        if not block:
            prob.objects.append(['dummy_block','block'])

        prob_simplified = self.create_simplified_problem(prob)

        # print("\n\nPROB: " + str(prob.goal))
        # print("\nPROB SIMPLIFIED: " + str(prob_simplified.goal))

        return prob, prob_simplified

    ''' Add constants to the created PDDL+ problem '''
    def add_constants(self, prob : PddlPlusProblem):
        for fact in [['=', ['gravity'], 134.2],
                     ['=', ['active_bird'], 0],
                     ['=', ['angle'], 0],
                     ['not', ['angle_adjusted']],
                     ['not', ['pig_killed']],
                     ['=', ['angle_rate'], 10],
                     ['=', ['ground_damper'], 0.4]
                     ]:
            prob.init.append(fact)

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

    def add_platform(self, obj, prob, groundOffset):
        obj_name = '{}_{}'.format(obj[1]['type'], obj[0])
        prob.init.append(
            ['=', ['x_platform', obj_name], round((obj[1]['bbox'].bounds[2] + obj[1]['bbox'].bounds[0]) / 2)])
        prob.init.append(['=', ['y_platform', obj_name],
                          abs(round(abs(obj[1]['bbox'].bounds[1] + obj[1]['bbox'].bounds[3]) / 2) - groundOffset)])
        prob.init.append(['=', ['platform_height', obj_name], abs(obj[1]['bbox'].bounds[3] - obj[1]['bbox'].bounds[1])])
        prob.init.append(['=', ['platform_width', obj_name], abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0])])
        prob.objects.append([obj_name, 'platform'])


    def add_block(self, obj, prob, groundOffset):
        obj_name = '{}_{} '.format(obj[1]['type'], obj[0])
        bl_x = round((obj[1]['bbox'].bounds[2] + obj[1]['bbox'].bounds[0]) / 2)
        bl_y = abs(round(abs(obj[1]['bbox'].bounds[1] + obj[1]['bbox'].bounds[3]) / 2) - groundOffset)
        prob.init.append(['=', ['x_block', obj_name], bl_x])
        prob.init.append(['=', ['y_block', obj_name], bl_y])
        # prob.init.append(['block_supporting',obj_name])
        bl_height = abs(obj[1]['bbox'].bounds[3] - obj[1]['bbox'].bounds[1])
        bl_width = abs(obj[1]['bbox'].bounds[2] - obj[1]['bbox'].bounds[0])
        prob.init.append(['=', ['block_height', obj_name], bl_height])
        prob.init.append(['=', ['block_width', obj_name], bl_width])
        block_life_multiplier = 1.0
        block_mass_coeff = 1.0
        if obj[1]['type'] == 'wood':
            block_life_multiplier = 1.0
            block_mass_coeff = 0.375 * 1.3
        elif obj[1]['type'] == 'ice':
            block_life_multiplier = 0.5
            block_mass_coeff = 0.125 * 2
        elif obj[1]['type'] == 'stone':
            block_life_multiplier = 2.0
            block_mass_coeff = 1.2
        elif obj[1]['type'] == 'TNT':
            block_life_multiplier = 0.001
            block_mass_coeff = 1.2
            prob.init.append(['block_explosive', obj_name])
        else:  # not sure how this could ever happen
            block_life_multiplier = 2.0
            block_mass_coeff = 1.2
        prob.init.append(['=', ['block_life', obj_name], str(math.ceil(265 * block_life_multiplier))])
        prob.init.append(['=', ['block_mass', obj_name], str(block_mass_coeff)])
        bl_stability = 265 * (bl_width / bl_height) * (1 - (bl_y / groundOffset)) * block_mass_coeff
        prob.init.append(['=', ['block_stability', obj_name], bl_stability])
        prob.objects.append((obj_name, 'block'))

    ''' Add a Bird to the PDDL+ problem '''
    def add_bird(self,obj, prob : PddlPlusProblem, bird_index, groundOffset, slingshot):
        obj_name = '{}_{}'.format(obj[1]['type'], obj[0])
        prob.objects.append((obj_name, 'Bird'))  # This probably needs to change
        # prob.init.append(['not',['bird_dead',obj_name]])
        prob.init.append(['not', ['bird_released', obj_name]])
        prob.init.append(['=', ['x_bird', obj_name],
                          round((slingshot[1]['bbox'].bounds[0] + slingshot[1]['bbox'].bounds[2]) / 2) - 0])
        prob.init.append(['=', ['y_bird', obj_name], round(
            abs(((slingshot[1]['bbox'].bounds[1] + slingshot[1]['bbox'].bounds[3]) / 2) - groundOffset) - 0)])
        prob.init.append(['=', ['v_bird', obj_name], 270])
        prob.init.append(['=', ['vx_bird', obj_name], 0])
        prob.init.append(['=', ['vy_bird', obj_name], 0])
        prob.init.append(['=', ['m_bird', obj_name], 1])
        prob.init.append(['=', ['bounce_count', obj_name], 0])
        prob.init.append(['=', ['bird_id', obj_name], bird_index])

    ''' Add a pig object to the problem '''
    def add_pig(self, obj, prob, groundOffset):
        obj_name = '{}_{}'.format(obj[1]['type'], obj[0])
        prob.init.append(['not', ['pig_dead', obj_name]])
        prob.init.append(['=', ['x_pig', obj_name], round(abs(obj[1]['bbox'].bounds[2] + obj[1]['bbox'].bounds[0]) / 2)])
        prob.init.append(['=', ['y_pig', obj_name],
                          abs(round(abs(obj[1]['bbox'].bounds[1] + obj[1]['bbox'].bounds[3]) / 2) - groundOffset)])
        prob.init.append(['=', ['pig_radius', obj_name], self.compute_radius(obj)])
        prob.init.append(['=', ['m_pig', obj_name], self.defaults['m_pig']])
        prob.goal.append(['pig_dead', obj_name])
        prob.objects.append((obj_name, obj[1]['type']))

    ''' Translate an intermediate state to PddlPlusState. 
    Key difference between this method and translate_init_state... method is that here we consider the location of the birds'''
    def translate_intermediate_state_to_pddl_state(self):
        state_as_list = list()

        #we should probably use the self.sling on the object
        slingshot = self.get_sling()
        groundOffset = slingshot[1]['bbox'].bounds[3]

        slingshot_x = round((slingshot[1]['bbox'].bounds[0] + slingshot[1]['bbox'].bounds[2]) / 2)
        slingshot_y = round(abs(((slingshot[1]['bbox'].bounds[1] + slingshot[1]['bbox'].bounds[3]) / 2) - groundOffset) - 0)
        bird_index = 0
        platform = False
        block = False

        for o in self.objects.items():
            if o[1]['type'] == 'pig':
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                state_as_list.append(['not', ['pig_dead',obj_name]])
                state_as_list.append(['=', ['x_pig',obj_name], self.compute_x_coordinate(o)])
                state_as_list.append(['=', ['y_pig',obj_name], self.compute_y_coordinate(groundOffset, o)])
                state_as_list.append(['=', ['pig_radius', obj_name], self.compute_radius(o)])
                state_as_list.append(['=', ['m_pig', obj_name], 1])
            elif 'bird' in o[1]['type'].lower():
                obj_name = '{}_{}'.format(o[1]['type'], o[0])
                # prob.init.append(['not',['bird_dead',obj_name]])
                # prob.init.append(['not',['bird_released',obj_name]])
                self.compute_x_coordinate(o)

                # Need to separate the case where we're before shooting the bird and after.
                # Before: the bird location is considered as the location of the slingshot,
                # afterwards, it's the location of the birds bounding box
                x_bird = self.compute_x_coordinate(o)
                if x_bird>slingshot_x:
                    state_as_list.append(['=',['x_bird',obj_name], self.compute_x_coordinate(o)])
                    state_as_list.append(['=',['y_bird',obj_name], self.compute_y_coordinate(groundOffset,o)])
                else:
                    state_as_list.append(['=', ['x_bird', obj_name], slingshot_x])
                    state_as_list.append(['=', ['y_bird', obj_name], slingshot_y])


                # prob.init.append(['=',['v_bird',obj_name], 270])  Computing velocity is more difficult
                # prob.init.append(['=',['vx_bird',obj_name], 0])
                # prob.init.append(['=',['vy_bird',obj_name], 0])
                state_as_list.append(['=',['m_bird',obj_name], 1])
                #prob.init.append(['=',['bounce_count',obj_name], 0])
                state_as_list.append(['=',['bird_id',obj_name],bird_index])
                bird_index += 1
            elif o[1]['type'] == 'wood' or o[1]['type'] == 'ice' or o[1]['type'] == 'stone':
                block = True
                obj_name = '{}_{} '.format(o[1]['type'], o[0])
                state_as_list.append(['=',['x_block', obj_name], round((o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)])
                state_as_list.append(['=', ['y_block',obj_name], self.compute_y_coordinate(groundOffset, o)])
                state_as_list.append(['=',['block_height',obj_name],abs(
                    o[1]['bbox'].bounds[3] - o[1]['bbox'].bounds[1])])
                state_as_list.append(['=',['block_width',obj_name],abs(
                    o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])])
                block_life_multiplier = 1.0
                block_mass_coeff = 1.0
                if o[1]['type'] == 'wood':
                    block_life_multiplier = 1.0
                    block_mass_coeff = 0.375
                elif o[1]['type'] == 'ice':
                    block_life_multiplier = 0.5
                    block_mass_coeff = 0.125
                elif o[1]['type'] == 'stone':
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                else: # not sure how this could ever happen
                    block_life_multiplier = 2.0
                    block_mass_coeff = 1.2
                state_as_list.append(['=',['block_life',obj_name],str(265 * block_life_multiplier)])
                state_as_list.append(['=',['block_mass',obj_name],str(block_mass_coeff)])
            elif o[1]['type'] == 'hill':
                platform = True
                obj_name ='{}_{}'.format(o[1]['type'], o[0])
                state_as_list.append(['=',['x_platform', obj_name], round((o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0])/2)])
                state_as_list.append(['=', ['y_platform', obj_name], self.compute_y_coordinate(groundOffset, o)])
                state_as_list.append(['=', ['platform_height', obj_name], abs(o[1]['bbox'].bounds[3] - o[1]['bbox'].bounds[1])])
                state_as_list.append(['=', ['platform_width', obj_name], abs(o[1]['bbox'].bounds[2] - o[1]['bbox'].bounds[0])])
            elif o[1]['type'] == 'slingshot':
                slingshot = o
        for fact in [['=',['gravity'], 134.2],
                     ['=',['active_bird'], 0],
                     ['=', ['angle'], 0],
                     ['not', ['angle_adjusted']],
                     ['=',['angle_rate'], 10],
                     ['=', ['ground_damper'], 0.4]
                     ]:
            state_as_list.append(fact)
        return PddlPlusState(state_as_list)

    ''' Computes the y coordinate of the given object as the center of its bounding box, 
    corrected for the given groundOffset '''
    def compute_y_coordinate(self, groundOffset, o):
        return abs(round(abs(o[1]['bbox'].bounds[1] + o[1]['bbox'].bounds[3]) / 2) - groundOffset)

    ''' Computes the x coordinate of the given object as the center of its boundingbox.'''
    def compute_x_coordinate(self, o):
        return round(abs(o[1]['bbox'].bounds[2] + o[1]['bbox'].bounds[0]) / 2)

    ''' Computes the radius of the given object '''

