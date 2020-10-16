import pathlib
import settings
from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

from agent.cartpole_hydra_agent import CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent

# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'demo-client.config'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'parc-mockn-cartpole.config'
WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'client.config'
USE_HYDRA = True


def main():
    observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent) if USE_HYDRA else WSUObserver()
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=False)
    dispatcher.run()


if __name__ == '__main__':
    main()
