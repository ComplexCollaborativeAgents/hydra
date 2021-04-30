import torch
import torchvision.transforms as transforms
import numpy as np
from state_prediction.ab_fcnn import MyCNN
import pickle
from state_prediction.obs_to_imgs import  SBObs_to_Imgs
import settings

from agent.consistency.focused_anomaly_detector import FocusedAnomalyDetector


class FocusedSBAnomalyDetector(FocusedAnomalyDetector):
    def __init__(self, threshold=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_predictor = MyCNN(self.device)
        self.state_predictor.load_state_dict(torch.load('{}/state_prediction/fc_single_channel_posweight/pretrained_model.pt'.format(settings.ROOT_PATH),
                                                        map_location=self.device))
        with open('{}/state_prediction/fc_single_channel_posweight/fc_pretrained_novelty_detector.pickle'.format(settings.ROOT_PATH), 'rb') as f:
            novelty_detector_info = pickle.load(f)
        self.novelty_detector = novelty_detector_info['clf']
        self.novelty_feature_mu = novelty_detector_info['x_mu']
        self.novelty_feature_std = novelty_detector_info['x_std']

        self.transform_s = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((48,84)),
            transforms.Lambda(lambda x: (x > 0.5).float()),
        ])
        self.transform_a = transforms.Lambda(lambda x: torch.from_numpy(x)  / torch.tensor([480, 840, 3000, 480, 840]))

        self.threshold = threshold

    def detect(self, sb_ob):
        state, action, next_state = self.convert_to_images(sb_ob)
        state = self.transform_s(state)
        action = self.transform_a(action)
        next_state = self.transform_s(next_state[-1])

        state_resized = (state.sum(axis=0).reshape(-1) > 0).type(torch.float32)
        next_state_resized = (next_state.sum(axis=0).reshape(-1) > 0).type(torch.float32)
        y_hat = self.state_predictor(state_resized.unsqueeze(0), action.unsqueeze(0)).squeeze()
        x = self.novelty_features(state_resized, y_hat, next_state_resized)
        prob = self.novelty_detector.predict_proba(x).squeeze()[1]
        y_hat = self.novelty_detector.predict(x).squeeze()
        # returning empty or non-empty list to keep with cart-pole assumption
        if prob > self.threshold:
            return [True], prob
        return [], prob

    def novelty_features(self, state, y_hat, next_state):
        mask_changed = torch.ne(state, next_state)   # this will consider only points that change
        mask_predicted_changed = torch.ne(state, y_hat > 0)   # this will consider only points that change

        num_points_changed = mask_changed.sum()
        num_points_unchanged = (~mask_changed).sum()
        num_points_predicted_changed = mask_predicted_changed.sum()
        num_points_predicted_unchanged = (~mask_predicted_changed).sum()

        err_changed = torch.ne(y_hat > 0, next_state) * mask_changed
        err_unchanged = torch.ne(y_hat > 0, next_state) * (~mask_changed)
        err_predicted_changed = torch.ne(y_hat > 0, next_state) * mask_predicted_changed
        err_predicted_unchanged = torch.ne(y_hat > 0, next_state) * (~mask_predicted_changed)

        x = np.r_[err_changed, err_unchanged, err_predicted_changed, err_predicted_unchanged]
        x[np.isnan(x)] = 0
        x = (x - self.novelty_feature_mu) / self.novelty_feature_std
        
        return x.reshape(1, -1)

    def convert_to_images(self, sb_obs):
        obs_conv =  SBObs_to_Imgs()
        state, action, inter_states = obs_conv.Obs_to_StateActionNextState(sb_obs)
        state_img = obs_conv.state_to_nD_img(state)
        inter_states_img = []
            #
        for s in inter_states:
            next_img = obs_conv.state_to_nD_img(s)
            inter_states_img.append(next_img)

        sb_state = {'state': state_img, 'action': action, 'next_state':inter_states_img}

        return sb_state['state'], sb_state['action'], sb_state['next_state']
