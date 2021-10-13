from typing import Callable

from agent.planning.nyx.compiler import JIT


class HappeningMixin:

    def __init__(self):
        self.preconditions = []
        self.effects = []

        self._preconditions_func = None
        self._preconditions_code = None
        self._effects_func = None
        self._effects_code = None

    @property
    def preconditions_func(self) -> Callable:
        if self._preconditions_func is None:
            self._compile_preconditions()
        return self._preconditions_func

    @property
    def preconditions_code(self) -> str:
        if self._preconditions_code is None:
            self._compile_preconditions()
        return self._preconditions_code

    @property
    def effects_func(self) -> Callable:
        if self._effects_func is None:
            self._compile_effects()
        return self._effects_func

    @property
    def effects_code(self) -> str:
        if self._effects_code is None:
            self._compile_effects()
        return self._effects_code

    def _compile_preconditions(self):
        declaration, func = JIT.compile_expression(self.preconditions, name='preconditions')
        self._preconditions_func = func
        self._preconditions_code = declaration

    def _compile_effects(self):
        declaration, func = JIT.compile_statements(self.effects, name='effects')
        self._effects_func = func
        self._effects_code = declaration
