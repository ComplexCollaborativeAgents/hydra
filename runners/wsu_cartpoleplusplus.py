import pathlib
import sys

import settings
import argparse

from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

from agent.cartpoleplusplus_hydra_agent import CartpolePlusPlusHydraAgentObserver, RepairingCartpolePlusPlusHydraAgent, CartpolePlusPlusHydraAgent

LOG_PATH = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'log'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'demo-client.config'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'local-client.config'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'parc-mockn-cartpole.config'

# UNCOMMENT THE BELOW CONFIG FILE FOR WSU EVALUATION
WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'client.config'
USE_HYDRA = True


def main():

    # agent_type_arg = CartpoleHydraAgent # this was sent to WSU for early tests
    agent_type_arg = RepairingCartpolePlusPlusHydraAgent
    if len(sys.argv) > 1:
        agent_type_arg = CartpolePlusPlusHydraAgent if sys.argv[1] == '-norepair' else RepairingCartpolePlusPlusHydraAgent
    print("\n\n" + str(agent_type_arg) + "\n\n")

    log_file = LOG_PATH / "hydra.{}.txt".format(settings.HYDRA_INSTANCE_ID)
    observer = CartpolePlusPlusHydraAgentObserver(agent_type=agent_type_arg) if USE_HYDRA else WSUObserver()
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=False,
                               logfile=str(log_file))
    dispatcher.run()


if __name__ == '__main__':
    main()
