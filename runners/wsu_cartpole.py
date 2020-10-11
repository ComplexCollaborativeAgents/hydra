import pathlib
import settings
from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

from agent.cartpole_hydra_agent import CartpoleHydraAgentObserver

WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'demo-client.config'
USE_HYDRA = False


def main():
    observer = CartpoleHydraAgentObserver() if USE_HYDRA else WSUObserver()
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=True)
    dispatcher.run()


if __name__ == '__main__':
    main()
