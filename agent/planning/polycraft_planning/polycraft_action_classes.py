from agent.planning.meta_model import MetaModel
from agent.planning.pddl_plus import PddlPlusWorldChange
from agent.planning.polycraft_planning.polycraft_pddl_objects_and_constants import PddlGameMapCellType
from worlds.polycraft_world import PolycraftAction, PolycraftState


class PddlPolycraftAction(PolycraftAction):
    """ Wrapper for Polycraft World Action that also stores the grounded pddl action that corresponds to this action """

    def __init__(self, poly_action, pddl_name, binding):
        super().__init__()
        self.poly_action = poly_action
        self.binding = binding
        self.pddl_name = pddl_name
        if binding:
            self.action_name = self.pddl_name + ' ' + ' '.join(
                [PddlGameMapCellType.get_cell_object_name(cell) for cell in self.binding.values()])
        else:
            self.action_name = self.pddl_name
        self.action_name = self.action_name.replace(':', '_')

    def is_success(self, result: dict):
        return self.poly_action.is_success(result)

    def __str__(self):
        if len(self.binding) > 0:
            params_str = " ".join([f"{k}={v}" for k, v in self.binding.items()])
            return f"<({self.pddl_name} {params_str}) success={self.success}>"
        else:
            return f"<({self.pddl_name}) success={self.success}>"

    __repr__ = __str__

    def do(self, state: PolycraftState, env) -> dict:
        result = self.poly_action.do(state, env)
        self.success = self.poly_action.success
        return result

    def can_do(self, state: PolycraftState, env) -> bool:
        return self.poly_action.can_do(state, env)


class PddlPolycraftActionGenerator:
    """ An object that bridges between pddl actions and polycraft actions"""

    def __init__(self, pddl_name):
        self.pddl_name = pddl_name  # The name of this PDDL action

    """ A class representing a PDDL+ action in polycraft """

    def to_pddl(self, meta_model: MetaModel) -> PddlPlusWorldChange:
        """ This method should be implemented by sublcasses and output a string representation of the corresponding
        PDDL+ action """
        raise NotImplementedError()

    def to_polycraft(self, parameter_binding: dict) -> PolycraftAction:
        """ This method should be implemented by sublcasses and output a string representation of the corresponding
        PDDL+ action """
        raise NotImplementedError()

    def to_pddl_polycraft(self, parameter_binding: dict) -> PddlPolycraftAction:
        return PddlPolycraftAction(poly_action=self.to_polycraft(parameter_binding),
                                   pddl_name=self.pddl_name,
                                   binding=parameter_binding)
