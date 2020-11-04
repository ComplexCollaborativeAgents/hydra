import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from PIL import Image

class ABDataset(Dataset):
    def __init__(self, path, transform=None):
        self.num_obs = len(glob.glob1(path, '*.npz')) - 1   # https://stackoverflow.com/questions/1320731/count-number-of-files-with-certain-extension-in-python
        self.transform = transform
        self.path = path



    def __len__(self):
        return self.num_obs

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.path, 'obs_data_{}.npz'.format(idx + 1))
        obs = np.load(img_name, allow_pickle=True)#['obs_list'][()]
        obs = dict(obs)
        # print(obs)
        # print(list(obs.keys()))
        # exit()
        try:
            obs['next_states'] = obs['next_states'][-1]
        except:
            return None

        if self.transform:
            obs['state'] = self.transform(obs['state'])
            # for i in range(len(obs['next_states'])):
            #     obs['next_states'][i] = self.transform(obs['next_states'][i])
            # it's unclear how we'll deal with the next_states sequence since it is of varying lengths and PyTorch doesn't like that
            obs['next_states'] = self.transform(obs['next_states'])
        return obs

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 84))
    ]
    )
    torch_dataset = ABDataset('obs_data', transform)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    for obs in dataloader:
        print(obs.keys())
        states, actions, next_states = obs['state'], obs['action'], obs['next_states']
        print(states.shape, states.unique())
        print(actions.shape, actions.unique())
        print(next_states.shape, next_states.unique())
