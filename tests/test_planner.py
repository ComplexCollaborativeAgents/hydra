import os
import pytest
import settings
import agent.planning.sb_planner as pl

@pytest.mark.skip("MEK: Paths are wrong here. something to look at for Wiktor")
def test_planner():

    prob_test = open("%s/sb_test_prob.pddl" % settings.SB_PLANNING_DOCKER_PATH).read()

    pddl_problem_file = open("%s/sb_prob.pddl" % str(settings.SB_PLANNING_DOCKER_PATH), "w+")
    pddl_problem_file.write(prob_test)
    pddl_problem_file.close()

    assert os.stat("%s/sb_prob.pddl" % settings.SB_PLANNING_DOCKER_PATH).st_size > 0

    planner = pl.SBPlanner()
    actions, _ = planner.get_plan_actions()

    assert os.stat("%s/docker_build_trace.txt" % settings.SB_PLANNING_DOCKER_PATH).st_size > 0

    assert os.stat("%s/docker_plan_trace.txt" % settings.SB_PLANNING_DOCKER_PATH).st_size > 0

    assert os.stat("%s/docker_build_trace.txt" % settings.VAL_DOCKER_PATH).st_size > 0

    assert os.stat("%s/docker_validation_trace.txt" % settings.VAL_DOCKER_PATH).st_size > 0

    assert len(actions) > 0

if __name__ == '__main__':
    planner = pl.SBPlanner()
    actions, _ = planner.get_plan_actions()
    assert len(actions) > 0
