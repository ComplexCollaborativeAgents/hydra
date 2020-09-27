import pathlib
import settings
from worlds.wsu.wsu_dispatcher import WSUObserver, WSUDispatcher

WSU_CARTPOLE = pathlib.Path(settings.ROOT_PATH) / 'worlds' / 'wsu' / 'demo-client.config'


def main():
    observer = WSUObserver()
    dispatcher = WSUDispatcher(observer, config_file=str(WSU_CARTPOLE), debug=True, printout=True)
    dispatcher.run()

if __name__ == '__main__':
    main()
