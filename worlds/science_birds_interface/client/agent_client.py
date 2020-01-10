from enum import Enum
from time import sleep
import socket
import json
import numpy as np
from PIL import Image

RESPONSE_BUFFER_SIZE = 102400

class GameState(Enum):
    UNKNOWN = 0
    MAIN_MENU = 1
    EPISODE_MENU = 2
    LEVEL_SELECTION = 3
    LOADING = 4
    PLAYING = 5
    WON = 6
    LOST = 7

class PlayingMode(Enum):
    COMPETITION = 0
    TRAINING = 1

class AgentClient():

    def __init__(self, configuration):
        self.server_port = int(configuration['port'])
        self.server_host = configuration['host']
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.server_socket.settimeout(100)
        self.request_bytes_size = configuration['requestbufbytes']

    # INITIALIZATION
    def connect_to_server(self):
        try:
            self.server_socket.connect((self.server_host, self.server_port))
            print('Client connected to server on port: ' + str(self.server_port))
        except socket.error as e:
            print('Client failed to connect to server. Requested HOST: '
                  + str(self.server_host) + '. Requested PORT: '
                  + str(self.server_port) + '. Error Message: ' + str(e))
            raise e

    def disconnect_from_server(self):
        try:
            self.server_socket.close()
            print('Client disconnected from server.')
        except socket.error as e:
            print('Client failed to disconnect from server. Requested HOST: '
                  + str(self.server_host) + '. Requested PORT: '
                  + str(self.server_port) + '. Error Code: '.format(e))
            raise e

    # REQUESTS
    def configure(self, id):
        bytes_request = self.__encode_request_str_to_byte("configure", 1)
        id_bytes = id.to_bytes(self.request_bytes_size, byteorder='big')
        playing_mode = PlayingMode.TRAINING.value
        playing_mode_bytes = playing_mode.to_bytes(1, byteorder= 'big')
        self.server_socket.sendall(bytes_request + id_bytes + playing_mode_bytes)
        print("Sending configure request")
        sleep(1)
        info_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)
        print('Received configuration')
        return info_bytes

    def read_image_from_stream(self):
        # Read the message head : 4-byte width and 4-byte height, respectively
        bytewidth = 4
        byteheight = 4
        screenshot_width_bytes = self.server_socket.recv(bytewidth)
        screenshot_heigth_bytes = self.server_socket.recv(byteheight)

        width = int.from_bytes(screenshot_width_bytes, byteorder='big')
        height = int.from_bytes(screenshot_heigth_bytes, byteorder='big')

        total_bytes = width * height * 3

        # Read the raw RGB data
        read_bytes = 0
        # read first bytes
        image_bytes = self.server_socket.recv(2048)
        read_bytes += image_bytes.__len__()

        # read the rest
        while (read_bytes < total_bytes):
            byte_buffer = self.server_socket.recv(2048)
            byte_buffer_length = byte_buffer.__len__()
            if (byte_buffer_length != -1):
                image_bytes += byte_buffer
            else:
                break
            read_bytes += byte_buffer_length

        rgb_image = Image.frombytes("RGB", (width, height), image_bytes) #check if  PIL is needed

        # TODO: Remove after Debug
        #rgb_image.save(os.path.join('./', 'test'), format='png')

        print('Received screenshot')

        img = np.array(rgb_image)
        # Convert RGB to BGR
        rgb_image = img[:, :, ::-1].copy()
        # cv2.imwrite('image.png',img)
        return img

    def read_ground_truth_from_stream(self):
        # read Ground Truth
        length_bytes = self.server_socket.recv(4)
        msg_length = int.from_bytes(length_bytes, byteorder='big')

        data = b''
        while len(data) < msg_length:
            packet = self.server_socket.recv(msg_length - len(data))
            if not packet:
                return None
            data+=packet
        #print ('finished receiving ...')
        data_string = data.decode("UTF-8")
        data_string = data_string[:-5]
        ground_truth = json.loads(data_string)

        return ground_truth

    def do_screenshot(self):
        print("Sending screenshot request")
        self.server_socket.sendall(self.__encode_request_str_to_byte("doScreenShot", 1))
        sleep(1)
        return self.read_image_from_stream()

    def get_game_state(self):
        print("Sending gamestate request")
        self.server_socket.sendall(self.__encode_request_str_to_byte("getState", 1))
        sleep(1)
        game_state_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)

        print('Received gamestate')
        game_state = self.__decode_game_state(game_state_bytes)
        return game_state

    def load_level(self, level_number):
        print("Sending loadLevel request")
        enc_request_bytes = self.__encode_request_str_to_byte("loadLevel", 1)
        level_number_bytes = level_number.to_bytes(4, byteorder='big')

        self.server_socket.sendall(enc_request_bytes + level_number_bytes)
        sleep(1)

        info_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)
        print('Received loadLevel')

        return info_bytes

    def restart_level(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("restartLevel", 1))
        info_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)

        return info_bytes

    def shoot(self, fx, fy, dx, dy, t1, t2, isPolar):
        if(isPolar):
            enc_request_bytes = self.__encode_request_str_to_byte("pshoot", 1)
        else:
            enc_request_bytes = self.__encode_request_str_to_byte("cshoot", 1)

        fx_bytes = fx.to_bytes(self.request_bytes_size, byteorder='big',signed=True)
        fy_bytes = fy.to_bytes(self.request_bytes_size, byteorder='big',signed=True)
        dx_bytes = dx.to_bytes(self.request_bytes_size, byteorder='big',signed=True)
        dy_bytes = dy.to_bytes(self.request_bytes_size, byteorder='big',signed=True)
        t1_bytes = t1.to_bytes(self.request_bytes_size, byteorder='big',signed=True)
        t2_bytes = t2.to_bytes(self.request_bytes_size, byteorder='big',signed=True)

        self.server_socket.sendall(enc_request_bytes +
                                   fx_bytes + fy_bytes +
                                   dx_bytes + dy_bytes +
                                   t1_bytes + t2_bytes)

        info_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)

        return info_bytes

    def get_all_level_scores(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("getAllLevelScores", 1))
        score_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)
        #print("score bytes len ", len(score_bytes))

        n_levels = int(len(score_bytes)/4)
        scores = [0 for x in range(n_levels)]
        for i in range(len(scores)):
            scores[i] = int.from_bytes(score_bytes[i * 4:i*4 + 4], byteorder='big')
        return scores

    def get_current_score(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("getCurrentLevelScore", 1))
        score_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)
        current_score = int.from_bytes(score_bytes, byteorder='big')

        return current_score

    def get_number_of_levels(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte('getNoOfLevels', 1))
        level_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)
        levels = int.from_bytes(level_bytes, byteorder='big')
        return levels

    def get_current_level(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte('getCurrentLevel', 1))
        level_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)
        #print ('cur level byte ', level_bytes)
        level = int.from_bytes(level_bytes, byteorder='big')
        return level

    def fully_zoom_in(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("fullyZoomIn", 1))
        info_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)

        return info_bytes

    def fully_zoom_out(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("fullyZoomOut", 1))
        info_bytes = self.server_socket.recv(RESPONSE_BUFFER_SIZE)

        return info_bytes

    def get_ground_truth_with_screenshot(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("getGroundTruthWithScreenshot", 1))

        ground_truth = self.read_ground_truth_from_stream()
        image = self.read_image_from_stream()

        return (image, ground_truth)

    def get_ground_truth_without_screenshot(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("getGroundTruthWithoutScreenshot", 1))

        ground_truth = self.read_ground_truth_from_stream()

        return ground_truth

    def get_noisy_ground_truth_with_screenshot(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("getNoisyGroundTruthWithScreenshot", 1))

        ground_truth = self.read_ground_truth_from_stream()
        image = self.read_image_from_stream()

        return (image, ground_truth)

    def get_noisy_ground_truth_without_screenshot(self):
        self.server_socket.sendall(self.__encode_request_str_to_byte("getNoisyGroundTruthWithoutScreenshot", 1))

        ground_truth = self.read_ground_truth_from_stream()

        return ground_truth

    # HELPERS
    def __encode_request_str_to_byte(self, string_request, bytes_to_use):
        switcher = {
            "doScreenShot": 11,
            "configure": 1,
            "loadLevel": 51,
            "restartLevel": 52,
            "cshoot": 31,
            "pshoot": 32,
            "getState": 12,
            "fullyZoomOut": 34,
            "getNoOfLevels": 15,
            "getCurrentLevel": 14,
            "getBestScores": 13,
            "shootSeq": 11,
            "cFastshoot": 41,
            "pFastshoot": 42,
            "shootSeqFast": 43,
            "getAllLevelScores": 23,
            "clickInCentre": 36,
            "fullyZoomIn": 35,
            "getGroundTruthWithScreenshot": 61,
            "getGroundTruthWithoutScreenshot": 62,
            "getNoisyGroundTruthWithScreenshot": 63,
            "getNoisyGroundTruthWithoutScreenshot": 64,
            "getCurrentLevelScore": 65
        }

        request_code = switcher.get(string_request, -1)

        assert request_code != -1, 'Invalid request received: ' + string_request

        return request_code.to_bytes(bytes_to_use, byteorder='big')

    def __decode_game_state(self, game_state_byte):
        dec_game_state = int.from_bytes(game_state_byte, byteorder='big')
        game_state = GameState(dec_game_state)

        return game_state

if __name__ == "__main__":
    """ TEST AGENT """
    with open('server_client_config.json', 'r') as config:

        sc_json_config = json.load(config)

    client = AgentClient(sc_json_config[0])
    try:
        client.connect_to_server()
        client.configure(2888)
        img = client.do_screenshot()

        game_state = client.get_game_state()

        info = client.load_level(3)
        client.do_screenshot()
        level = client.get_current_level()
        print ("current level ", level)
        score = client.get_all_level_scores()
        print ("score ", score)
        client.zoom_in()
        client.zoom_out()
        info = client.shoot(172, 276, 943, 264, 0, 0, False)

        image, ground_truth = client.get_ground_truth_with_screenshot()
        ground_truth = client.get_ground_truth_without_screenshot()
        noisy_image, noisy_truth = client.get_noisy_ground_truth_with_screenshot()
        noisy_truth = client.get_noisy_ground_truth_without_screenshot()

        info = client.restart_level()
        client.disconnect_from_server()
    except socket.error as e:
        print("Error in client-server communication: " + str(e))
