import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import os
import glob
import numpy as np
from PIL import Image

class ABDataset(TensorDataset):
    def __init__(self, path, transform_s=None, transform_a=None, mode='train'):
        self.transform_s = transform_s
        self.transform_a = transform_a
        self.path = path
        assert mode in ['train', 'test', 'novelty', 'novelty_new']
        self.mode = mode
        states, actions, next_states = self._load_tensors()
        print('loaded tensors')
        super().__init__(states, actions, next_states)


    def _load_tensors(self):
        # use obs_data_all2 for first frame and obs_data_all for last frame
        if not os.path.isfile(os.path.join(self.path, self.mode + '_data.npz')):
            num_obs = len(glob.glob1(os.path.join(self.path, self.mode), '*.npz'))    # https://stackoverflow.com/questions/1320731/count-number-of-files-with-certain-extension-in-python
            drop_idx = []
            print('one by one')
            # for idx in range(num_obs):
            for idx, file in enumerate(glob.glob1(os.path.join(self.path, self.mode), '*.npz')):
                if idx % 100 == 0:
                    print(idx)
                img_name = os.path.join(self.path, self.mode, file)
                obs = np.load(img_name, allow_pickle=True)#['obs_list'][()]
                obs = dict(obs)
                try:
                    obs['next_states'] = obs['next_states'][-1]
                except IndexError:
                    drop_idx.append(idx)
                    continue
                if self.transform_s:
                    obs['state'] = self.transform_s(obs['state'])
                    # for i in range(len(obs['next_states'])):
                    #     obs['next_states'][i] = self.transform(obs['next_states'][i])
                    # it's unclear how we'll deal with the next_states sequence since it is of varying lengths and PyTorch doesn't like that
                    obs['next_states'] = self.transform_s(obs['next_states'])      
                if self.transform_a: 
                    obs['action'] = self.transform_a(obs['action'])

                if idx == 0:
                    all_state = torch.empty(num_obs, *(obs['state'].shape))
                    all_action = torch.empty(num_obs, *(obs['action'].shape))
                    all_next_states = torch.empty(num_obs, *(obs['next_states'].shape))

                all_state[idx] = obs['state']
                all_action[idx] = obs['action']
                all_next_states[idx] = obs['next_states']

            all_state = torch.from_numpy(np.delete(all_state.data.numpy(), drop_idx, axis=0))
            all_action = torch.from_numpy(np.delete(all_action.data.numpy(), drop_idx, axis=0))
            all_next_states = torch.from_numpy(np.delete(all_next_states.data.numpy(), drop_idx, axis=0))
            np.savez_compressed(os.path.join(self.path, self.mode + '_data.npz'), state=all_state, action=all_action, next_states=all_next_states)
        
        else:
            print('all together')
            data = np.load(os.path.join(self.path, self.mode + '_data.npz'))
            all_state = torch.from_numpy(data['state'])
            all_action = torch.from_numpy(data['action']) #* torch.tensor([480, 840, 3000, 480, 840]).view(1, -1)
            all_next_states = torch.from_numpy(data['next_states'])
        
        return all_state, all_action, all_next_states


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform_s = transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1)))
    transform_a = transforms.Lambda(lambda x: torch.from_numpy(x)  / torch.tensor([480, 840, 3000, 480, 840]))

    torch_dataset = ABDataset('data_test_train_preproc', transform_s, transform_a, mode='novelty_new')
    dataloader = DataLoader(
        torch_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    for states, actions, next_states in dataloader:
        print(states.shape, states.unique())
        print(actions.shape, actions.unique())
        print(next_states.shape, next_states.unique())
        break
