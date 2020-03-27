import os
import pytest
import settings
import agent.planning.planner as pl

def test_planner():

    prob_test = open("%s/sb_test_prob.pddl" % settings.PLANNING_DOCKER_PATH).read()
    planner = pl.Planner()
    planner.write_problem_file(prob_test)

    assert os.stat("%s/sb_prob.pddl" % settings.PLANNING_DOCKER_PATH).st_size > 0

    actions = planner.get_plan_actions()

    assert os.stat("%s/docker_build_trace.txt" % settings.PLANNING_DOCKER_PATH).st_size > 0

    assert os.stat("%s/docker_plan_trace.txt" % settings.PLANNING_DOCKER_PATH).st_size > 0

    assert len(actions) > 0