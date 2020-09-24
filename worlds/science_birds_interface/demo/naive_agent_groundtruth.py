import sys
#sys.path.append('./src')
sys.path.append('..')
from datetime import datetime
import time
import os
import numpy as np
from threading import Thread
import random
import json
import socket
from math import cos, sin, degrees, pi
try:
    from worlds.science_birds_interface.client.agent_client import AgentClient, GameState, RequestCodes
    from trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
    from computer_vision.GroundTruthReader import GroundTruthReader,NotVaildStateError
    from computer_vision.game_object import GameObjectType
    from utils.point2D import Point2D
    from computer_vision.GroundTruthTest import GroundTruthTest

except ModuleNotFoundError:
    from src.client.agent_client import AgentClient, GameState, RequestCodes
    from src.trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
    from src.computer_vision.GroundTruthReader import GroundTruthReader,NotVaildStateError
    from src.computer_vision.game_object import GameObjectType
    from src.computer_vision.GroundTruthTest import GroundTruthTest

import logging
import pathlib
SB_ROOT = pathlib.Path(__file__).parent.parent


class ClientNaiveAgent(Thread):
    """Naive agent (server/client version)"""
    def __init__(self, agent_ind, agent):

        self.agent_ind = agent_ind

        #################initalising the logger#################
        #file_handler saves all logs to a log file with agent_ind

        self.logger = logging.getLogger(str(self.agent_ind))

        formatter = logging.Formatter("%(asctime)s-Agent %(name)s-%(levelname)s : %(message)s")

        if not os.path.exists("log"):
            os.mkdir("log")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        #if agent_configs.save_logs:
        #    file_handler = logging.FileHandler(os.path.join("log","%s.log"%(self.agent_ind)))
        #    file_handler.setLevel(logging.DEBUG)
        #    file_handler.setFormatter(formatter)
        #    self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        #agent_ip = agent_configs.agent_host
        #agent_port = agent_configs.agent_port
        #observer_ip = agent_configs.observer_host
        #observer_port = agent_configs.observer_port


        ########################################################

        #Wrapper of the communicating messages

        self.ar = agent #AgentClient(agent_ip, agent_port, logger = self.logger)

        #the observer agent can only execute 6 command: configure, screenshot
        #and the four groundtruth related ones
        #self.observer_ar = AgentClient(observer_ip, observer_port)

        # try:
        #    self.ar.connect_to_server()
        #except socket.error as e:
        #    self.logger.error("Error in client-server communication: " + str(e))


#do not use observer
#        try:
#            self.observer_ar.connect_to_server()
#        except socket.error as e:
#            self.logger.error("Error in client-server communication: " + str(e))

        self.current_level = 0
        self.training_level_backup = 0;
        self.failed_counter = 0
        self.solved = []
        self.tp = SimpleTrajectoryPlanner()
        self.id = agent_ind # 28888
        self.first_shot = True
        self.prev_target = None
        self.novelty_existence = -1;
        self.sim_speed = 50
        self.prev_gt = None
        self.repeated_gt_counter = 0
        self.gt_patient = 10
        self.if_check_gt = False

        #load model coef
        self.model = np.loadtxt(SB_ROOT / "model",delimiter=",")
        self.target_class = list(map(lambda x : x.replace("\n", ""),open(SB_ROOT / 'target_class').readlines()))
        #self._logger = logging.getLogger("ClientNaiveAgent") remove for updated logger version

    def sample_state(self, request = RequestCodes.GetNoisyGroundTruthWithScreenshot, frequency = 0.5):
        """
         sample a state from the observer agent
         this method allows to be run in a different thread
         NOTE: Setting the frequency too high, i.e. <0.01 may cause lag in science birds game
               due to the calculation of the groundtruth
        """
        while (True):
            vision = None
            if request == RequestCodes.GetGroundTruthWithScreenshot:
                image, ground_truth = self.observer_ar.get_ground_truth_with_screenshot()
                #set to true to ignore invalid state and return the vision object regardless
                # of #birds and #pigs
                vision = GroundTruthReader(ground_truth,self.look_up_matrix,self.look_up_obj_type)
                vision.set_screenshot(image)

            elif request == RequestCodes.GetGroundTruthWithoutScreenshot:
                ground_truth = self.observer_ar.get_ground_truth_without_screenshot()
                vision = GroundTruthReader(ground_truth,self.look_up_matrix,self.look_up_obj_type)

            elif request == RequestCodes.GetNoisyGroundTruthWithScreenshot:
                image, ground_truth = self.observer_ar.get_noisy_ground_truth_with_screenshot()
                vision = GroundTruthReader(ground_truth,self.look_up_matrix,self.look_up_obj_type)
                vision.set_screenshot(image)

            elif request == RequestCodes.GetNoisyGroundTruthWithoutScreenshot:
                ground_truth = self.observer_ar.get_noisy_ground_truth_without_screenshot()
                vision = GroundTruthReader(ground_truth,self.look_up_matrix,self.look_up_obj_type)
            time.sleep(frequency)


    def get_next_unsolved_level(self):
        level = 0
        unsolved = False
        #all the level have been solved, then get the first unsolved level
        for i in range(len(self.solved)):
            if self.solved[i] == 0:
                unsolved = True
                level = i + 1
                if level <= self.current_level and self.current_level < len(self.solved):
                    continue
                else:
                    return level

        if unsolved:
            return level
        level = (self.current_level + 1)%len(self.solved)
        if level == 0:
            level = len(self.solved)
        return level

    def get_next_level(self):

        level = self.current_level + 1
        n_levels = self.update_no_of_levels()

        if n_levels == 0:
            level = 1
            return level

        level = level%n_levels
        if level <= 0:
            level = 1
        return level


    def check_my_score(self):
        """
         * Run the Client (Naive Agent)
        *"""
        scores = self.ar.get_all_level_scores()
        #print(" My score: ")
        level = 1
        for i in scores:
            self.logger.info(" level ", level, "  ", i)
            if i > 0:
                self.solved[level - 1] = 1
            level += 1
        return scores

    def check_current_level_score(self):
        current_score = self.ar.get_current_score()
        self.logger.info("current score is %d "%current_score)
        return current_score

    def update_no_of_levels(self):
        # check the number of levels in the game
        n_levels = self.ar.get_number_of_levels()

        # if number of levels has changed make adjustments to the solved array
        if n_levels > len(self.solved):
            for n in range(len(self.solved), n_levels):
                self.solved.append(0)

        if n_levels < len(self.solved):
            self.solved = self.solved[:n_levels]

        #self.logger.info('No of Levels: ' + str(n_levels))

        return n_levels

    def run(self):
        # self.ar.configure(self.id)
        #do not use observer
        #self.observer_ar.configure(self.id)
        self.ar.set_game_simulation_speed(self.sim_speed)
        #n_levels = self.update_no_of_levels()

        #self.solved = [0 for x in range(n_levels)]

        #load next available level
        #self.current_level = self.ar.load_next_available_level()
        #self.novelty_existence = self.ar.get_novelty_info()

        '''
        Uncomment this section to run TEST for requesting groudtruth via different thread
        '''
        #gt_thread = Thread(target=self.sample_state)
        #gt_thread.start()
        '''
        END TEST
        '''

        #indicates if the previous game level set is a training set
        change_from_training = False

        #ar.load_level((byte)9)
        while True:
            #test purpose only
            #sim_speed = random.randint(1, 50)
            #self.ar.set_game_simulation_speed(sim_speed)
            #print(‘simulation speed set to ', sim_speed)

            #test for multi-thread groundtruth reading
            state = self.solve()
            # try:
            #     state = self.solve()
            # except:
            #     self.logger.warning("Erros in solving level %s, state is set to lost"%(self.current_level))
            #     state = GameState.LOST


            #If the level is solved , go to the next level
            if state == GameState.WON:
                self.repeated_gt_counter = 0
                #check for change of number of levels in the game
                n_levels = self.update_no_of_levels()

                #/System.out.println(" loading the level " + (self.current_level + 1) )
                #self.check_current_level_score()
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

                # make a new trajectory planner whenever a new level is entered
                self.tp = SimpleTrajectoryPlanner()

            elif state == GameState.LOST:
                self.repeated_gt_counter = 0
                #check for change of number of levels in the game
                #n_levels = self.update_no_of_levels()

                self.check_current_level_score()

                #If lost, then restart the level
                self.failed_counter += 1
                if self.failed_counter > 0: #for testing , go directly to the next level

                    self.failed_counter = 0
                    self.current_level = self.ar.load_next_available_level()
                    self.novelty_existence = self.ar.get_novelty_info()

                else:
                    self.logger.info("fail level count does not reach the limit, restart the level")
                    self.ar.restart_level()

            elif state == GameState.LEVEL_SELECTION:
                self.logger.info("unexpected level selection page, go to the last current level : " \
                , self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.MAIN_MENU:
                self.repeated_gt_counter = 0
                self.logger.info("unexpected main menu page, reload the level : %s"%self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.EPISODE_MENU:
                self.logger.info("unexpected episode menu page, reload the level:  %s"%self.current_level)
                self.current_level = self.ar.load_next_available_level()
                self.novelty_existence = self.ar.get_novelty_info()

            elif state == GameState.REQUESTNOVELTYLIKELIHOOD:
                #Require report novelty likelihood and then playing can be resumed
                #dummy likelihoods:
                self.logger.info("received request of novelty likelihood")
                novelty_likelihood=0.9
                non_novelty_likelihood=0.1
                novel_obj_ids = {1,-2,-398879789}
                novelty_level = 0
                novelty_description = "";
                self.ar.report_novelty_likelihood(novelty_likelihood,non_novelty_likelihood,novel_obj_ids,novelty_level,novelty_description)

            elif state == GameState.NEWTRIAL:
                self.repeated_gt_counter = 0
                #Make a fresh agent to continue with a new trial (evaluation)
                self.logger.critical("new trial state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0

            elif state == GameState.NEWTESTSET:
                self.repeated_gt_counter = 0
                self.logger.critical("new test set state received")
                #DO something to clone a test-only agent that does not learn
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()

                if change_from_training:
                    self.training_level_backup = self.current_level
                self.current_level = 0
                change_from_training = False

            elif state == GameState.NEWTRAININGSET:
                self.repeated_gt_counter = 0
                #DO something to start a fresh agent for a new training set
                self.logger.critical("new training set state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()
                self.current_level = 0
                self.training_level_backup = 0
                change_from_training = True

            elif state == GameState.RESUMETRAINING:
                self.repeated_gt_counter = 0
                #DO something to resume the training agent to the previous training
                self.logger.critical("resume training set state received")
                (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo) = self.ar.ready_for_new_set()
                change_from_training = True
                self.current_level = self.training_level_backup

            elif state == GameState.EVALUATION_TERMINATED:
                #store info and disconnect the agent as the evaluation is finished
                self.logger.critical("Evaluation terminated.")
                return

    def _update_reader(self, dtype, if_check_gt=False):
        '''
        update the ground truth reader with 4 different types of ground truth if the ground truth is vaild
        otherwise, return the state.

        str type : ground_truth_screenshot , ground_truth, noisy_ground_truth_screenshot,noisy_ground_truth
        '''

        self.show_ground_truth = False


        if dtype == 'ground_truth_screenshot':
            image, ground_truth = self.ar.get_ground_truth_with_screenshot()
            vision = GroundTruthReader(ground_truth,self.model,self.target_class)
            vision.set_screenshot(image)
            self.show_ground_truth = True # draw the ground truth with screenshot or not

        elif dtype == 'ground_truth':
            ground_truth = self.ar.get_ground_truth_without_screenshot()

            vision = GroundTruthReader(ground_truth,self.model,self.target_class)

        elif dtype == 'noisy_ground_truth_screenshot':
            image, ground_truth = self.ar.get_noisy_ground_truth_with_screenshot()

            vision = GroundTruthReader(ground_truth,self.model,self.target_class)
            vision.set_screenshot(image)
            self.show_ground_truth = True # draw the ground truth with screenshot or not

        elif dtype == 'noisy_ground_truth':
            ground_truth = self.ar.get_noisy_ground_truth_without_screenshot()

            #check for gt
            if if_check_gt:
                tester = GroundTruthTest(ground_truth)
                try:
                    tester.check()
                except AssertionError:
                    print(ground_truth)
                    tester.check()

            #self.logger.info(ground_truth)
            #print(ground_truth)
            vision = GroundTruthReader(ground_truth,self.model,self.target_class)

        return vision

    def solve(self):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """
        self.logger.info("sovling level %s"%(self.current_level))

        ground_truth_type = 'noisy_ground_truth'


        game_state = self.ar.get_game_state()



        #####################################################################
        # for skipping the level
        # if agent receives over 10 gt truth per level, skip the level

        if game_state != GameState.PLAYING:
            return game_state

        else:
            self.repeated_gt_counter += 1
            if self.repeated_gt_counter > self.gt_patient:
                self.logger.warning("counter %s reached, game state set to lost"%(self.gt_patient))
                self.repeated_gt_counter = 0
                return GameState.LOST

        #####################################################################

        vision = self._update_reader(ground_truth_type, self.if_check_gt)
        if not vision.is_vaild():
            self.logger.info("no pig or birds found")
            return self.ar.get_game_state()

        if self.show_ground_truth:
            vision.showResult()

        sling = vision.find_slingshot_mbr()[0]
        #TODO: look into the width and height issue of Traj planner
        sling.width,sling.height = sling.height,sling.width

        # get all the pigs
        pigs = vision.find_pigs_mbr()
        self.logger.info("no of pigs: " + str(len(vision.find_pigs_mbr() )))
        for pig in pigs:
            self.logger.info("pig location: " + str(pig.get_centre_point()))
        state = self.ar.get_game_state()

        # if there is a sling, then play, otherwise skip.
        if sling != None:
            #If there are pigs, we pick up a pig randomly and shoot it.
            if pigs:
                release_point = None
                # random pick up a pig
                pig = pigs[random.randint(0,len(pigs)-1)]
                temp_pt = pig.get_centre_point()

                #TODO change computer_vision.cv_utils.Rectangle
                #to be more intuitive
                _tpt = Point2D(temp_pt[1],temp_pt[0])

                # if the target is very close to before, randomly choose a
                # point near it
                if self.prev_target != None and self.prev_target.distance(_tpt) < 10:
                    _angle = random.uniform(0, 1) * pi * 2
                    _tpt.X = _tpt.X + int(cos(_angle)) * 10
                    _tpt.Y = _tpt.Y + int(sin(_angle)) * 10
                    self.logger.info("Randomly changing to %s"%_tpt)

                self.prev_target = Point2D(_tpt.X, _tpt.Y)

                ################estimate the trajectory###################
                self.logger.info('################estimate the trajectory###################')

                pts = self.tp.estimate_launch_point(sling, _tpt)

                if not pts:
                    #Add logic to deal with unreachable target
                    self.logger.info("the target is not reachable directly with the birds")
                    self.logger.info("just shoot...")
                    release_point = Point2D(-100,450)


                elif len(pts) == 1:
                    release_point = pts[0]
                elif len(pts) == 2:
                    # System.out.println("first shot " + first_shot)
                    # randomly choose between the trajectories, with a 1 in
                    # 6 chance of choosing the high one
                    if random.randint(0,5) == 0:
                        release_point = pts[1]
                    else:
                        release_point = pts[0]

                ref_point = self.tp.get_reference_point(sling)

                # Get the release point from the trajectory prediction module
                tap_time = 0
                if release_point != None:
                    release_angle = self.tp.get_release_angle(sling,release_point)
                    self.logger.info("Release Point: %s"%release_point)
                    self.logger.info("Release Angle: %s"%degrees(release_angle))
                    tap_interval = 0

                    birds = vision.find_birds()
                    bird_on_sling = vision.find_bird_on_sling(birds,sling)
                    bird_type = bird_on_sling.type
                    self.logger.info("bird_type: %s"%bird_type)
                    if bird_type == GameObjectType.REDBIRD:
                        tap_interval = 0 # start of trajectory
                    elif bird_type == GameObjectType.YELLOWBIRD:
                        tap_interval = 65 + random.randint(0,24) # 65-90% of the way
                    elif bird_type == GameObjectType.WHITEBIRD:
                        tap_interval =  50 + random.randint(0,19) # 50-70% of the way
                    elif bird_type == GameObjectType.BLACKBIRD:
                        tap_interval =  0 #do not tap black bird
                    elif bird_type == GameObjectType.BLUEBIRD:
                        tap_interval =  65 + random.randint(0,19) # 65-85% of the way
                    else:
                        tap_interval =  60

                    tap_time = self.tp.get_tap_time(sling, release_point, _tpt, tap_interval)

                else:
                    self.logger.warning("No Release Point Found")
                    return self.ar.get_game_state()

                # check whether the slingshot is changed. the change of the slingshot indicates a change in the scale.
                #self.ar.fully_zoom_out()

                vision = self._update_reader(ground_truth_type, self.if_check_gt)
                if isinstance(vision,GameState):
                    return vision
                if self.show_ground_truth:
                    vision.showResult()

                _sling = vision.find_slingshot_mbr()[0]
                _sling.width,_sling.height = _sling.height,_sling.width

                if _sling != None:
                    scale_diff = (sling.width - _sling.width)**2 +  (sling.height - _sling.height)**2
                    if scale_diff < 25:
                        dx = int(release_point.X - ref_point.X)
                        dy = int(release_point.Y - ref_point.Y)

                        if dx < 0:
                            self.logger.info('ref point %s, %s'%(ref_point.X, ref_point.Y))
                            self.logger.info('tap time %s '%(tap_time))
                            #self.ar.shoot(ref_point.X, ref_point.Y, dx, dy, 0, tap_time, False)
                            # used batch gt shoot

                            # the science birds game will use the true ref_point for the shot
                            # only the release point is required here
                            self.ar.shoot_and_record_ground_truth(release_point.X, release_point.Y, 0, tap_time, 1)
                            time.sleep(2/self.sim_speed)
                            state = self.ar.get_game_state()
                            if state == GameState.PLAYING:
                                vision = self._update_reader(ground_truth_type, self.if_check_gt)
                                if isinstance(vision,GameState):
                                    return vision
                                if self.show_ground_truth:
                                    vision.showResult()
                    else:
                        self.logger.info("Scale is changed, can not execute the shot, will re-segement the image")
                else:
                    self.logger.info("no sling detected, can not execute the shot, will re-segement the image")
        return state


if __name__ == "__main__":
    na = ClientNaiveAgent("1", False)
    na.run()
