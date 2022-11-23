from agent.planning.pddlplus_parser import *


class PddlObjectType:
    """ A generator for Pddl Objects """

    def __init__(self):
        """
        Accepts an object from SBState.objects
        """
        self.hyper_parameters = dict()
        self.pddl_type = "object"  # This the PDDL+ type of this object.

    def _compute_obj_attributes(self, obj, problem_params: dict):
        """
        Subclasses should override this setting all attributes of that object
        """
        return dict()

    def _compute_observable_obj_attributes(self, obj, problem_params: dict):
        """
        Subclasses should override this settign all attributes of that object that can be observed
        """
        return dict()

    def add_object_to_problem(self, prob: PddlPlusProblem, obj, problem_params: dict):
        """
        Populate a PDDL+ problem with details about this object
        """
        name = self._get_name(obj)
        prob.objects.append([name, self.pddl_type])
        attributes = self._compute_obj_attributes(obj, problem_params)
        for attribute, value in attributes.items():
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value, bool):
                if value:
                    prob.init.append([attribute, name])
                else:  # value == False
                    prob.init.append(['not', [attribute, name]])
            else:  # Attribute is a number
                prob.init.append(['=', [attribute, name], value])

    def add_object_to_state(self, pddl_state: PddlPlusState, obj, state_params: dict):
        """
        Populate a PDDL+ state with details about this object.
        Note that unlike a PDDL problem, states represent observations, so should only contain attributes observable in
        the world, and not internal/hidden variables used by the model.
        """
        name = self._get_name(obj)
        attributes = self._compute_observable_obj_attributes(obj, state_params)
        for attribute, value in attributes.items():
            fluent_name = (attribute, name)
            # If attribute is Boolean no need for an "=" sign
            if isinstance(value, bool):
                if value:
                    pddl_state.boolean_fluents.add(fluent_name)
                # TODO: Think how to handle boolean fluents with False value. Not as trivial as it sounds
            else:  # Attribute is a number
                pddl_state.numeric_fluents[fluent_name] = value

    def _get_name(self, obj):
        raise NotImplementedError()

class MetaModel:
    """ A meta-model used to generated PDDL+ domains and problems from observations """

    def __init__(self,
                 docker_path: str,
                 domain_file_name: str,
                 delta_t,
                 metric: str,
                 repairable_constants: list,
                 repair_deltas: list = None,
                 constant_numeric_fluents=None,
                 constant_boolean_fluents=None):

        if constant_boolean_fluents is None:
            constant_boolean_fluents = {}
        if constant_numeric_fluents is None:
            constant_numeric_fluents = {}
        self.docker_path = docker_path
        self.domain_file_name = domain_file_name
        self.delta_t = delta_t
        self.hyper_parameters = dict()  # These are parameters that do not appear in the PDDL files

        self.constant_numeric_fluents = dict(constant_numeric_fluents)
        self.constant_boolean_fluents = dict(constant_boolean_fluents)

        self.repairable_constants = list()

        for rc_list in repairable_constants:
            self.repairable_constants.append(list(rc_list))

        # self.repairable_constants = list(repairable_constants)

        self.repair_deltas = []


        if repair_deltas is None:
            for rc_list in self.repairable_constants:
                self.repair_deltas.append([1.0] * len(rc_list))
        else:
            for rd_list in repair_deltas:
                self.repair_deltas.append(list(rd_list))
        # if repair_deltas is None:
        #     self.repair_deltas = [1.0] * len(self.repairable_constants)
        # else:
        #     self.repair_deltas = repair_deltas

        self.metric = metric
        self.current_domain = None

    def create_pddl_problem(self, observed_state) -> PddlPlusProblem:
        """ Transtlate the observed state to a PDDL+ probelm for this domain in which the observed state is the
        initial state """
        raise NotImplementedError()

    def create_pddl_state(self, observed_state) -> PddlPlusState:
        """ Translate the observed state to a PDDL+ state in this domain """
        raise NotImplementedError()

    def create_pddl_domain(self, observed_state) -> PddlPlusDomain:
        """ Create a PDDL+ domain for the given observed state """
        # if self.current_domain is None:
        domain_file = "{}/{}".format(str(self.docker_path), self.domain_file_name)
        domain_parser = PddlDomainParser()
        self.current_domain = domain_parser.parse_pddl_domain(domain_file)
        return self.current_domain
