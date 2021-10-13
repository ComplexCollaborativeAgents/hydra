import sys
from typing import Iterator, TextIO

from agent.planning.nyx.syntax.action import Action
from agent.planning.nyx.syntax.event import Event
from agent.planning.nyx.syntax.process import Process
from agent.planning.nyx.syntax.state import State


class Trace(list):

    def __init__(self, *args: State):
        super().__init__()
        self.extend(args)
        self.finished = False

    def iter(self, extended: bool = True) -> Iterator[State]:
        def include_state(s: State):
            return extended or isinstance(s.predecessor_action, (Action, type(None)))
        return filter(include_state, self)

    def print(self, extended: bool = True, out: TextIO = sys.stdout):
        out.write("Trace finished: {}\n".format(str(self.finished)))
        out.write('---\n')

        for state in self.iter(extended=extended):
            if state.predecessor_action:
                happening_type = 'Action'
                if isinstance(state.predecessor_action, Process):
                    happening_type = 'Process'
                elif isinstance(state.predecessor_action, Event):
                    happening_type = 'Event'
                out.write("{}: {}\n".format(happening_type, state.predecessor_action.grounded_name))
            else:
                out.write("Initial state\n")
            out.write("Time: {}\n".format(state.time))
            out.write("State: {}\n".format(str(state.state_vars)))
            out.write('---\n')

    def to_file(self, plan_file: str, extended: bool = True):
        with open(plan_file, 'w') as f:
            self.print(extended=extended, out=f)
