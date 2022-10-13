import pathlib
import sys

import settings
import argparse
import os
import uuid

from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

from agent.cartpoleplusplus_hydra_agent import CartpolePlusPlusHydraAgentObserver, RepairingCartpolePlusPlusHydraAgent, CartpolePlusPlusHydraAgent

LOG_PATH = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'log'

USE_HYDRA = True


def main(config_file_cmd = None):
    unique_run_id = uuid.uuid4().hex
    settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH = os.path.join(settings.ROOT_PATH, 'agent', 'planning', 'cartpoleplusplus_planning_{}'.format(unique_run_id))

    os.mkdir(settings.CARTPOLEPLUSPLUS_PLANNING_DOCKER_PATH)\

    # WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'configs' / 'demo-client.config'
    # WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'configs' / 'local-client.config'
    # WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'configs' / 'parc-mockn-cartpole.config'
    # WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'configs' / 'gui-client.config'

    # UNCOMMENT THE BELOW CONFIG FILE FOR WSU EVALUATION
    WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'client.config'


    if (config_file_cmd):
        arg_config_list = config_file_cmd.split('=')
        if (arg_config_list[0] == '--config'):
            dirname = os.path.dirname(__file__)
            parsed_filename = os.path.join(dirname, arg_config_list[1])
            WSU_CARTPOLE = parsed_filename

    # agent_type_arg = CartpolePlusPlusHydraAgent # this was sent to WSU for early tests
    agent_type_arg = RepairingCartpolePlusPlusHydraAgent
    if len(sys.argv) > 1:
        agent_type_arg = CartpolePlusPlusHydraAgent if sys.argv[1] == '-norepair' else RepairingCartpolePlusPlusHydraAgent
    print("\n\n" + str(agent_type_arg) + "\n\n")

    log_file = LOG_PATH / "hydra.{}.txt".format(settings.HYDRA_INSTANCE_ID)
    observer = CartpolePlusPlusHydraAgentObserver(agent_type=agent_type_arg) if USE_HYDRA else WSUObserver()
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=False, logfile=str(log_file), ignore_secret=False)
    dispatcher.run()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
