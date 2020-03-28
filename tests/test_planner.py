import os
import pytest
import settings
import agent.planning.planner as pl

def test_planner():

    prob_test = open("%s/sb_test_prob.pddl" % settings.PLANNING_DOCKER_PATH).read()

    pddl_problem_file = open("%s/sb_prob.pddl" % str(settings.PLANNING_DOCKER_PATH), "w+")
    pddl_problem_file.write(prob_test)
    pddl_problem_file.close()

    assert os.stat("%s/sb_prob.pddl" % settings.PLANNING_DOCKER_PATH).st_size > 0

    planner = pl.Planner()
    actions = planner.get_plan_actions()

    assert os.stat("%s/docker_build_trace.txt" % settings.PLANNING_DOCKER_PATH).st_size > 0

    assert os.stat("%s/docker_plan_trace.txt" % settings.PLANNING_DOCKER_PATH).st_size > 0

    assert len(actions) > 0