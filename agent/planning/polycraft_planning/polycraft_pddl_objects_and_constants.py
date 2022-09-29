import enum

from agent.planning.meta_model import PddlObjectType


class PddlGameMapCellType(PddlObjectType):
    """
    A cell or block in the polycraft world.
    """
    def __init__(self, type_idx=-1, relevant_attributes=None):
        super().__init__()
        if relevant_attributes is None:
            relevant_attributes = [Predicate.isAccessible.name, Function.cell_x.name, Function.cell_z.name]
        self.pddl_type = "cell"
        self.type_idx = type_idx
        self.relevant_attributes = relevant_attributes

    @staticmethod
    def get_cell_object_name(cell_id: str):
        """ Return the object name in PDDL for the given cell """
        return "cell_{}".format("_".join(cell_id.split(",")))

    def _get_name(self, obj):
        """ Return the object name in PDDL """
        (cell_id, cell_attr) = obj
        return PddlGameMapCellType.get_cell_object_name(cell_id)

    def _compute_obj_attributes(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        fluent_to_value = dict()
        fluent_to_value[Function.cell_type.name] = self.type_idx

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute in self.relevant_attributes:
                fluent_to_value[attribute] = attribute_value

        return fluent_to_value

    def _compute_observable_obj_attributes(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for observable fluents created for this object"""
        fluent_to_value = dict()
        fluent_to_value[Function.cell_type.name] = self.type_idx

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute in self.relevant_attributes:
                fluent_to_value[attribute] = attribute_value

        return fluent_to_value


class PddlDoorCellType(PddlGameMapCellType):
    """
    A polycraft cell/block subtype for door cells - for planning efficiency.
    """
    def __init__(self, type_idx=-1):
        super().__init__(type_idx, [])
        self.pddl_type = PddlType.door_cell.name

    def _compute_obj_attributes(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        fluent_to_value = dict()
        fluent_to_value[Function.door_cell_type.name] = self.type_idx

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute == Predicate.isAccessible.name:
                fluent_to_value[Predicate.door_is_accessible.name] = attribute_value
            elif attribute == Predicate.open.name:
                if attribute_value.upper() == "TRUE":
                    fluent_to_value[Predicate.door_is_accessible.name] = True

        return fluent_to_value


class PddlSafeCellType(PddlGameMapCellType):
    """
    A polycraft cell/block subtype for safes - for planning efficiency.
    """
    def __init__(self, type_idx=-1):
        super().__init__(type_idx, [])
        self.pddl_type = PddlType.safe_cell.name

    def _compute_obj_attributes(self, obj, params: dict) -> dict:
        """ Maps fluent_name to fluent_value for all fluents created for this object"""
        fluent_to_value = dict()

        (cell_id, cell_attr) = obj
        for attribute, attribute_value in cell_attr.items():
            # If attribute is Boolean no need for an "=" sign
            if attribute == Predicate.isAccessible.name:
                fluent_to_value[Predicate.safe_is_accessible.name] = attribute_value

        return fluent_to_value


class PddlType(enum.Enum):
    cell = "cell"
    door_cell = "door_cell"
    safe_cell = "safe_cell"


class Predicate(enum.Enum):
    """ Note: the first prameter in the list is needed: otherwise python will merge enum elements. """
    isAccessible = ["isAccessible", ("?c", PddlType.cell.name)]
    # adjacent = ["adjacent", ("?c1", PddlType.cell.name), ("?c2", PddlType.cell.name)]

    door_is_accessible = ["door_is_accessible", ("?c", PddlType.door_cell.name)]
    # adjacent_to_door = ["adjacent_to_door", ("?c1", PddlType.cell.name), ("?c2", PddlType.door_cell.name)]
    open = ["open", ("?c", PddlType.door_cell.name)]
    passed_door = ["passed_door", ("?c", PddlType.door_cell.name)]

    safe_is_accessible = ["safe_is_accessible", ("?c", PddlType.safe_cell.name)]
    # adjacent_to_safe = ["adjacent_to_safe", ("?c1", PddlType.cell.name), ("?c2", PddlType.safe_cell.name)]
    safe_collected = ["safe_collected", ("?c", PddlType.safe_cell.name)]
    safe_open = ["safe_open", ("?c", PddlType.safe_cell.name)]

    def to_pddl(self) -> list:
        """ Returns this predicate in a list format as expected by the pddl domain object """
        predicate_as_list = [self.name]
        for (param_name, param_type) in self.value[1:]:
            predicate_as_list.extend([param_name, "-", param_type])
        return predicate_as_list


class Function(enum.Enum):
    """ Note: the first prameter in the list is needed: otherwise python will merge enum elements. """
    cell_type = ["cell_type", ("?c", PddlType.cell.name)]
    cell_x = ["cell_x", ("?c", PddlType.cell.name)]
    cell_z = ["cell_z", ("?c", PddlType.cell.name)]
    door_cell_type = ["door_cell_type", ("?c", PddlType.door_cell.name)]
    selectedItem = ["selectedItem"]
    Steve_x = ["steve_x"]
    Steve_z = ["steve_z"]

    def to_pddl(self) -> list:
        """ Returns this function in a list format as expected by the pddl domain object """
        function_as_list = [self.name]
        for (param_name, param_type) in self.value[1:]:
            function_as_list.extend([param_name, "-", param_type])
        return function_as_list
