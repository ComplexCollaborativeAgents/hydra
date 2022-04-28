import pathlib
import sys

import settings
import argparse
import os

from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

from agent.cartpoleplusplus_hydra_agent import CartpolePlusPlusHydraAgentObserver, RepairingCartpolePlusPlusHydraAgent, CartpolePlusPlusHydraAgent

LOG_PATH = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'log'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'demo-client.config'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'local-client.config'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'parc-mockn-cartpole.config'

# UNCOMMENT THE BELOW CONFIG FILE FOR WSU EVALUATION
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'client.config'
USE_HYDRA = True


def main(config_file_cmd = None):

    WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'client.config'
    # WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'parc-mockn-cartpole.config'

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
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=True, ignore_secret=False,
                               logfile=str(log_file))
    dispatcher.run()


if __name__ == '__main__':
    if (len(sys.argv) > 1):
        main(sys.argv[1])
    else:
        main()
