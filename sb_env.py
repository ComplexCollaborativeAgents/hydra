from agent.hydra_agent import HydraAgent
from agent.sb_hydra_agent import SBHydraAgent
import numpy
from agent.perception.perception import ProcessedSBState
import worlds.science_birds as sb
import worlds.science_birds_interface as SB
from agent.sb_hydra_agent import *
from worlds.science_birds_interface.trajectory_planner.trajectory_planner import *


print("check_point0")
#(1) making env
env = sb.ScienceBirds(None, launch=True, config="/home/sailor/hydra/data/science_birds/config/0_5_novelty_level_1_type6_non-novelty_type222.xml") #science_birds/level-15-novel-bird.xml")
current_level = env.sb_client.load_next_available_level()
raw_state = env.get_current_state()
##TODO: make action
angle = 10.0
SBagent = SBHydraAgent()
action = SBagent.meta_model.angle_to_action_time(10, raw_state)

perception = Perception()
processed_state = perception.process_state(raw_state)

angle = 45.0
print("checkpoint 1")
#pddl_state = SBagent.meta_model.create_pddl_problem(processed_state).get_init_state()
#default_time = SBagent.meta_model.angle_to_action_time(angle, pddl_state)
#tim_act = TimedAction("pa-twang blueBird_-336", default_time)
#sb1_action = SBagnet.meta_model.create_sb_action(tim_act, processed_state)


# Convert angle to release point
angle = 45.0
#processed_state = ProcessedSBState
#    ys = [200,250]
#    xs = [40,60]
#    sling = Rectangle([ys,xs])
#https://gitlab-external.parc.com/hydra/hydra/-/blob/master/worlds/science_birds_interface/trajectory_planner/trajectory_planner.py

tp = SimpleTrajectoryPlanner()
ref_point = tp.get_reference_point(processed_state.sling)
release_point_from_plan = tp.find_release_point(processed_state.sling, math.radians(angle))
action = SB.SBShoot(release_point_from_plan.X, release_point_from_plan.Y,
                            tap_timing, ref_point.X, ref_point.Y, timed_action)

exit()
#sb_action = 
#SBShoot
#45.0
raw_state, reward = self.env.act(sb_action)

print(env, raw_state)


