from lego_sorter_server.server import Server
import logging

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    Server.run()
