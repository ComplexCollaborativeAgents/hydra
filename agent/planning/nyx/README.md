# Nyx - a new PDDL+ planner written in Python

Nyx is a PDDL+ parser and planner written in python with a focus on simplicity. It is a discretization-based planner that approximates the continuous system dynamics using uniform time steps (Δt) and step functions.


This work was based on the classical parser and planner written by PUCRS (https://github.com/pucrs-automated-planning/pddl-parser).

## Source
- [nyx.py](nyx.py): Main Runner
- [planner.py](planner.py): planner and associated functions
- [PDDL.py](PDDL.py): PDDL parser
- [heuristic_functions.py](heuristic_functions.py): heuristic function definitions used in GBFS and A* Searches.
- [syntax](syntax/) folder with PDDL object classes and supporting elements:
  - [action.py](syntax/action.py) 
  - [event.py](syntax/event.py) 
  - [process.py](syntax/process.py)
  - [state.py](syntax/state.py)
  - [visited_state.py](syntax/visited_state.py)
  - [constants.py](syntax/constants.py)
- [ex](ex/) folder with PDDL domains:
  - [Car](ex/car)
  - [Sleeping Beauty](ex/sleeping_beauty/)
  - [Cartpole](ex/cartpole/)
  - [Vending Machine](ex/vending_machine/)
  - [Non-Temporal](ex/non-temporal/) folder with non-temporal PDDL domains:
	  - [Dinner](ex/non-temporal/dinner/)
	  - [Blocks World](ex/non-temporal/blocksworld/)
	  - [Dock Worker Robot](ex/non-temporal/dwr/)
	  - [Travelling Salesman Problem](ex/non-temporal/tsp/)

## Parser execution
```Shell
python -B nyx.py ex/car/car.pddl ex/car/pb01.pddl -t:1 -v
```

## Current limitations of our planner
- No support for Timed-Initial Literals/Fluents (TILs/TIFs)
- No support for durative actions (since they can be represented equivalently by instantaneous actions and a process (start-process-stop paradigm))
- No support for object subtypes
- No negated predicates in problem file's :init

