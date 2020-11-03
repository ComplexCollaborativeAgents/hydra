import pathlib
import settings
from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

from agent.cartpole_hydra_agent import CartpoleHydraAgentObserver, RepairingCartpoleHydraAgent

LOG_PATH = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'log'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'demo-client.config'
# WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'parc-mockn-cartpole.config'
WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'runners' / 'client.config'
USE_HYDRA = True


def main():
    log_file = LOG_PATH / "hydra.{}.txt".format(settings.HYDRA_INSTANCE_ID)
    observer = CartpoleHydraAgentObserver(agent_type=RepairingCartpoleHydraAgent) if USE_HYDRA else WSUObserver()
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=False,
                               logfile=str(log_file))
    dispatcher.run()


if __name__ == '__main__':
    main()
