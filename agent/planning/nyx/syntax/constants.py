#!/usr/bin/env python

from agent.planning.nyx.syntax.action import Action

# -----------------------------------------------
# NYX Global Variables
# -----------------------------------------------

TIME_PASSING_ACTION = Action('time-passing', [], [], [])

# -----------------------------------------------
# NYX CommandLine Arguments
# -----------------------------------------------

DELTA_T = 0.1
TEMPORAL_DOMAIN = False
VERBOSE_OUTPUT = False
VERY_VERBOSE_OUTPUT = False
TIME_HORIZON = 1000
TIMEOUT = 1800
NUMBER_PRECISION = 10
PRINT_INFO = 10000
NO_PLAN = False
DOUBLE_EVENT_CHECK = False
PLOT_VARS = True

SEARCH_BFS = True
SEARCH_DFS = False
SEARCH_GBFS = False
SEARCH_ASTAR = False

CUSTOM_HEURISTIC_ID = 0

TRACK_G = False

SEARCH_ALGO_TXT = "BFS"

VALIDATE = False
PRINT_ALL_STATES = False

# -----------------------------------------------
# NYX CommandLine Arguments
# -----------------------------------------------

HELP_TEXT = "\n" \
            "\n===========================================================================\n\n" \
            "Nyx Release v0.1\n" \
            "PDDL+ Planner for Hybrid Domains\n" \
            "\n" \
            "Contact: Wiktor Piotrowski - wiktorpi@parc.com\n" \
            "" \
            "\n===========================================================================\n" \
            "\n\tCOMMAND-LINE OPTIONS:\n" \
            "\n\t-h\t\thelp." \
            "\n\t-t:<x>\t\ttime step duration (default t=0.1)." \
            "\n\t-th:<x>\t\ttime horizon (default th=1000)." \
            "\n\t-to:<x>\t\ttimeout limit in seconds (default to=1800)." \
            "\n\t-np:<x>\t\tnumber precision in digits after decimal point (default np=10)." \
            "\n\t-search:bfs\tsearch algorithm: breadth-first search (default)." \
            "\n\t-search:dfs\tsearch algorithm: depth-first search." \
            "\n\t-search:gbfs\tsearch algorithm: greedy best-first search." \
            "\n\t-search:astar\tsearch algorithm: A*." \
            "\n\t-pi:<x>\t\tprint ongoing search info every <x> visited states (default pi=10000)." \
            "\n\t-p\t\tprint all visited states." \
            "\n\t-noplan\t\tdo not print the plan." \
            "\n\t-dblevent\tcheck triggered events again at the end of time-passing action (in addition to the default check at the beginning)." \
            "\n\t-v\t\tverbose plan output." \
            "\n\t-vv\t\tvery verbose plan output." \
            "\n\t-h\t\thelp." \
            "\n\n===========================================================================\n" \
            "\n\n"

            # "\n\t-val\t\tuse VAL a posteriori to validate results." \
