import numpy as np
import cv2

class SBObs_to_Imgs():
    def __init__(self):

        self.TYPES = ['blueBird',
            'yellowBird',
            'blackBird',
            'redBird',
            'birdWhite',
            'platform',
            'pig',
            'TNT',
            'slingshot',
            'ice',
            'stone',
            'wood',
       #     'magician',
       #     'wizard',
       #     'worm',
       #     'butterfly',
            'unknown']

        self.NO_TYPES = len(self.TYPES)

    def Unpack_SBEpisodeLog(self,obs):
        sb_state = obs.state  # SBState
        sb_action = obs.action # SBAction
        sb_intermediate_states = obs.intermediate_states
        reward = obs.reward

        return sb_state, sb_action, sb_intermediate_states, reward

    def Unpack_SBState(self, sb_state):
        objects = sb_state.objects
        image = sb_state.image
        game_state = sb_state.game_state
        sling = sb_state.sling
        return objects, image, game_state, sling


    def Unpack_SBAction(self, sb_action):
        dx = sb_action.dx
        dy = sb_action.dy
        tap = sb_action.tap
        ref_x = sb_action.ref_x
        ref_y = sb_action.ref_y
        return np.array([dx, dy, tap, ref_x, ref_y])

    def Obs_to_StateActionNextState(self, obs):
        sb_state, sb_action, sb_intermediate_states, reward = self.Unpack_SBEpisodeLog(obs)
        state, _ , _ , _ = self.Unpack_SBState(sb_state)
        current_state = state
        action = self.Unpack_SBAction(sb_action)
        inter_states = [] # list storing intermediate states in a sequential order

        for sb_inter_state in sb_intermediate_states:
            state, _ , _ , _ = self.Unpack_SBState(sb_inter_state)
            inter_states.append(state)

        return current_state, action, inter_states

    def state_to_image(self,state):
        # state is a dict
        keys = list(state.keys())

        img_state = np.zeros((480,840))
        for k in range(len(keys)):
            object = state[keys[k]]
            poly = object['polygon']
            x, y = poly.exterior.coords.xy
            poly_coords = np.int32(np.vstack((x,y)).T)
            cv2.fillPoly(img_state, pts =[poly_coords], color=(255,255,255))
        # cv2.imshow("img", img_state)
        # cv2.waitKey()

        return img_state

    def state_to_nD_img(self,state):

        keys = list(state.keys())
        img_state = np.zeros((480,840,self.NO_TYPES), dtype = np.uint8)

        for k in range(len(keys)):
            object = state[keys[k]]
            type = object['type']

            if type in self.TYPES:
                type_idx = np.where(np.array(self.TYPES) == type)[0][0]
                img = img_state[:,:,type_idx]
                img = np.array(img)
                poly = object['polygon']
                if poly.type == "Polygon":
                    x, y = poly.exterior.coords.xy
                    poly_coords = np.int32(np.vstack((x,y)).T)
                    cv2.fillPoly(img, pts =[poly_coords], color=(255,255,255))
                    img_state[:,:,type_idx] = img

                elif poly.type == "MultiPolygon":
                    for p in range(len(poly)):
                        x, y = poly[p].exterior.coords.xy
                        poly_coords = np.int32(np.vstack((x,y)).T)
                        cv2.fillPoly(img, pts =[poly_coords], color=(255,255,255))
                        img_state[:,:,type_idx] = img

            else:
                print("TYPE NOT FOUND", type)
        # tmp_img = np.max(img_state,axis = 2)
        # cv2.imshow("img", tmp_img)
        # cv2.waitKey()

        return img_state
